"""Verify that configuration and the LLM layer work on this machine.

Run this straight after setup, before building anything on top:

    python -m scripts.smoke_test

It checks configuration loading, then makes one real (tiny) call to the
provider in both text and JSON mode. Exits non-zero on failure so it can be
used in CI later.
"""

from __future__ import annotations

import sys

from app.config import PROJECT_ROOT, settings
from app.llm import LLMError, LLMNotConfigured, complete_json, complete_verbose

OK = "[ok]"
FAIL = "[fail]"


def check_config() -> bool:
    """Print the resolved configuration and confirm a key is present."""
    print("Configuration")
    print(f"  project root     : {PROJECT_ROOT}")
    print(f"  agent model      : {settings.llm_model}")
    print(f"  router model     : {settings.router_model}")
    print(f"  embedding model  : {settings.embedding_model}")
    print(f"  chroma dir       : {settings.abs_path(settings.chroma_dir)}")
    print(f"  calendar ready   : {settings.calendar_configured}")

    if not settings.llm_configured:
        print(f"{FAIL} GROQ_API_KEY is missing or still the placeholder.")
        print("      Copy .env.example to .env and add a key from console.groq.com")
        return False
    print(f"{OK} GROQ_API_KEY is present.")
    return True


def check_text_completion() -> bool:
    """Make one plain-text call and report latency and token usage."""
    try:
        response = complete_verbose(
            "Reply with exactly the word: pong",
            system="You are a terse test harness. Obey literally.",
            max_tokens=10,
        )
    except LLMError as exc:
        print(f"{FAIL} Text completion failed: {exc}")
        return False

    print(
        f"{OK} Text completion: {response.text!r} "
        f"({response.latency_ms} ms, "
        f"{response.prompt_tokens}+{response.completion_tokens} tokens)"
    )
    return True


def check_json_completion() -> bool:
    """Make one JSON-mode call, which is what the router will rely on."""
    try:
        parsed = complete_json(
            'Classify this message: "Can I book a cleaning on Tuesday?"\n'
            'Reply as JSON: {"intent": "booking" | "inquiry" | "complaint"}',
            system="You are an intent classifier. Reply with JSON only.",
            model=settings.router_model,
            max_tokens=64,
        )
    except LLMError as exc:
        print(f"{FAIL} JSON completion failed: {exc}")
        return False

    print(f"{OK} JSON completion: {parsed}")
    return True


def main() -> int:
    print("=" * 62)
    print("AI Knowledge Assistant — step 1 smoke test")
    print("=" * 62)

    if not check_config():
        return 1

    print("\nProvider calls")
    results = [check_text_completion(), check_json_completion()]

    print()
    if all(results):
        print("All checks passed. The LLM layer is ready.")
        return 0
    print("Some checks failed. See the messages above.")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LLMNotConfigured as exc:
        print(f"{FAIL} {exc}")
        sys.exit(1)
