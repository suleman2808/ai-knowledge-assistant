"""The single gateway to the language model.

Nothing else in this project imports an LLM SDK. Agents, the router and the
API layer all call `complete()` or `complete_json()` from here. That gives
us three things:

1. **Swappable provider.** Changing from Groq to OpenAI, Anthropic or a
   local Ollama server means rewriting `_chat()` in this file. No agent, no
   node and no test changes.
2. **One place for failure policy.** Rate limits, timeouts and missing keys
   are handled once, consistently, and are turned into a single exception
   type the rest of the app understands.
3. **One place for observability.** Latency, token counts and model choice
   are recorded here rather than sprinkled across a dozen call sites.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# How many times to retry a call that failed for a transient reason.
MAX_ATTEMPTS = 3
# Base seconds for exponential backoff: 0.5s, 1s, 2s (plus jitter).
BACKOFF_BASE = 0.5


class LLMError(RuntimeError):
    """Any failure to obtain a completion.

    Callers catch this one type. Whether the cause was a missing key, a rate
    limit or a network drop, the agent's job is the same: degrade
    gracefully. `user_message` carries text that is safe to show a customer.
    """

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.user_message = user_message or (
            "I'm having trouble reaching my language service right now. "
            "Please try again in a moment, or call the clinic directly."
        )


class LLMNotConfigured(LLMError):
    """Raised when no API key is present.

    Separate from `LLMError` so startup checks and `/health` can report a
    setup problem distinctly from a runtime outage.
    """


@dataclass
class LLMResponse:
    """A completion plus the metadata we want for analytics."""

    text: str
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: Any = field(default=None, repr=False)


# --------------------------------------------------------------------------
# Provider wiring — the only part that knows the word "Groq"
# --------------------------------------------------------------------------

_client: Any = None


def _get_client() -> Any:
    """Return a lazily constructed, reused provider client.

    Built on first use rather than at import so that the app can start,
    serve `/health` and return a clear error message even with no key set.
    """
    global _client
    if _client is not None:
        return _client

    if not settings.llm_configured:
        raise LLMNotConfigured(
            "GROQ_API_KEY is missing or still set to the placeholder value. "
            "Copy .env.example to .env and add a key from console.groq.com.",
            user_message=(
                "The assistant isn't fully configured yet. "
                "Please contact the site administrator."
            ),
        )

    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise LLMError(f"Groq SDK is not installed: {exc}") from exc

    _client = Groq(api_key=settings.groq_api_key, timeout=30.0)
    return _client


def _is_retryable(exc: Exception) -> bool:
    """Decide whether a failed call is worth trying again.

    Rate limits and transient network/server errors are retried. A bad API
    key or a malformed request will fail identically every time, so those
    are surfaced immediately instead of wasting the user's time.
    """
    name = type(exc).__name__
    if name in {"AuthenticationError", "PermissionDeniedError", "BadRequestError"}:
        return False
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and (status == 429 or status >= 500)


def _chat(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
) -> LLMResponse:
    """Send a chat request to the provider, with retries.

    This is the seam. To change providers, reimplement this function so it
    accepts the same arguments and returns an `LLMResponse`.
    """
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        # Provider-side constrained decoding: the model is forced to emit
        # syntactically valid JSON. Cheaper and far more reliable than
        # asking politely in the prompt and parsing whatever comes back.
        kwargs["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        started = time.perf_counter()
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:  # provider SDKs raise a wide variety of types
            last_error = exc
            if not _is_retryable(exc) or attempt == MAX_ATTEMPTS:
                break
            delay = BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, 0.25)
            logger.warning(
                "LLM call failed (%s), retrying in %.2fs [attempt %d/%d]",
                type(exc).__name__, delay, attempt, MAX_ATTEMPTS,
            )
            time.sleep(delay)
            continue

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(completion, "usage", None)
        return LLMResponse(
            text=(completion.choices[0].message.content or "").strip(),
            model=model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            raw=completion,
        )

    raise LLMError(
        f"LLM call failed after {MAX_ATTEMPTS} attempt(s): "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


# --------------------------------------------------------------------------
# Public API — what the rest of the app uses
# --------------------------------------------------------------------------


def complete(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Return the model's reply to `prompt` as plain text.

    Args:
        prompt: The user-turn content.
        system: Optional system instruction setting role and constraints.
        model: Override the configured model (the router passes the small
            one here).
        temperature: Override sampling temperature. Lower is more
            deterministic; the default of 0.2 suits grounded answering.
        max_tokens: Cap on the reply length.

    Returns:
        The reply text, stripped.

    Raises:
        LLMError: The call could not be completed. Callers should catch this
            and fall back to a safe response.
    """
    return complete_verbose(
        prompt,
        system=system,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    ).text


def complete_verbose(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Like `complete()`, but returns latency and token usage too.

    Used by the analytics layer, which wants the metadata, and by anything
    that needs to report cost or speed.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    return _chat(
        messages,
        model=model or settings.llm_model,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        json_mode=False,
    )


def complete_json(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """Return the model's reply parsed as a JSON object.

    Used wherever the app needs a decision rather than prose — intent
    classification, complaint severity, extracting a date from a booking
    request. Temperature defaults to 0.0 because these are classification
    tasks, where we want the same input to produce the same output.

    Raises:
        LLMError: The call failed, or the reply was not valid JSON.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _chat(
        messages,
        model=model or settings.llm_model,
        temperature=temperature,
        max_tokens=max_tokens or settings.llm_max_tokens,
        json_mode=True,
    )

    try:
        parsed = json.loads(response.text)
    except json.JSONDecodeError as exc:
        # Should be unreachable with json_mode on, but a provider swap might
        # not support constrained decoding, so we fail loudly and clearly.
        raise LLMError(
            f"Expected JSON from the model but parsing failed: {exc}. "
            f"Raw reply: {response.text[:200]!r}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


def health() -> dict[str, Any]:
    """Report whether the LLM layer is usable, without making a call.

    Consumed by `/health` so an operator can distinguish "not configured"
    from "configured but the provider is down".
    """
    return {
        "configured": settings.llm_configured,
        "provider": "groq",
        "model": settings.llm_model,
        "router_model": settings.router_model,
    }
