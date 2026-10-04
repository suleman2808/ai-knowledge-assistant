"""Shared contract for the specialist agents.

Agents are plain functions: query in, `AgentResponse` out. They do not know
they will be run inside a LangGraph, do not touch FastAPI, and do not read
global state. That is what makes each one testable on its own, and it is
why step 5 can wire them into a graph without changing a line of agent
code.

Every agent obeys the same two rules:

1. **Always return an `AgentResponse`.** Never raise for an expected
   condition — a missing API key, an unreachable calendar, a question the
   documents cannot answer. The response carries `success=False` and a
   sentence that is safe to show a customer.
2. **Never leak internals to the user.** Technical detail goes in
   `metadata` for logs and analytics; `answer` is what a patient reads.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from app import llm


# Characters models insert that are invisible or near-invisible but that
# corrupt the text when copied. Soft hyphens inside a phone number are the
# motivating case: the number renders correctly and then fails to dial.
_INVISIBLE = {
    "­": "",   # soft hyphen
    "​": "",   # zero-width space
    "‌": "",   # zero-width non-joiner
    "‍": "",   # zero-width joiner
    "﻿": "",   # byte-order mark
    "‑": "-",  # non-breaking hyphen
    " ": " ",  # non-breaking space
    " ": " ",  # narrow non-breaking space
}

# Dashes the model reaches for and this interface does not want. An em
# dash is correct typography and still the wrong character here: it is
# indistinguishable from a minus at small sizes, it does not survive
# every copy-paste, and the house style for these three screens is a
# plain hyphen. Converted at the same boundary as the invisibles, so it
# covers model prose as well as the text written in this repository.
_DASHES = {
    "—": "-",  # em dash
    "–": "-",  # en dash
    "‒": "-",  # figure dash
    "―": "-",  # horizontal bar
}


def normalise(text: str) -> str:
    """Strip invisible characters models sometimes emit.

    Observed in practice: the reply "(503) 555-0142" came back with a
    soft hyphen instead of a hyphen-minus, so the number looked right on
    screen and broke when copied. Cheap to fix at the boundary, and the
    boundary is the only place that catches every agent.

    Long dashes are folded to hyphens here for the same reason: it is the
    one place that catches prose the model wrote as well as prose this
    repository wrote.
    """
    for character, replacement in {**_INVISIBLE, **_DASHES}.items():
        text = text.replace(character, replacement)
    return text


@dataclass
class AgentResponse:
    """What every specialist agent returns."""

    answer: str
    """Customer-facing text. Always populated, even on failure."""

    agent: str
    """Which agent produced this, for logging and for the UI."""

    success: bool = True
    """False when the agent could not do its job. The answer still stands."""

    sources: list[dict[str, Any]] = field(default_factory=list)
    """Citations, for the Inquiry Agent. Empty for the others."""

    needs_followup: bool = False
    """True when the agent asked the user a question and expects a reply.

    The Booking Agent sets this when required details are missing, so the
    graph knows the conversation is mid-flow rather than complete.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Operator-facing detail: statuses, scores, ids, error strings."""

    def __post_init__(self) -> None:
        # Applied here rather than in each agent, so no agent can forget.
        self.answer = normalise(self.answer)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for the API response."""
        return {
            "answer": self.answer,
            "agent": self.agent,
            "success": self.success,
            "sources": self.sources,
            "needs_followup": self.needs_followup,
            "metadata": self.metadata,
        }


# A single place for the wording used when something breaks. Kept together
# so the voice stays consistent and so it can be reviewed without reading
# the agent code.
FALLBACK_MESSAGES = {
    "llm_unavailable": (
        "I'm having trouble reaching my language service at the moment, so I "
        "can't answer reliably. Please try again shortly, or call the laboratory "
        "on (503) 555-0142 and someone will help you straight away."
    ),
    "unexpected": (
        "Something went wrong at my end and I'd rather not guess. Please call "
        "the laboratory on (503) 555-0142 and the team will sort this out for you."
    ),
}


def failure(agent: str, kind: str, detail: str, **metadata: Any) -> AgentResponse:
    """Build a graceful failure response.

    Args:
        agent: The agent reporting the failure.
        kind: Key into `FALLBACK_MESSAGES`.
        detail: Technical explanation. Logged, never shown.
        **metadata: Anything else worth recording.
    """
    return AgentResponse(
        answer=FALLBACK_MESSAGES.get(kind, FALLBACK_MESSAGES["unexpected"]),
        agent=agent,
        success=False,
        metadata={"error_kind": kind, "error_detail": detail, **metadata},
    )


# Punctuation a model wraps a bare sentinel in: quotes, emphasis, a full
# stop, surrounding whitespace.
_SENTINEL_WRAPPERS = "\"'`*. \n\r\t"


def _looks_like(text: str, sentinels: tuple[str, ...]) -> bool:
    """Whether `text` is starting to spell out one of the sentinels.

    Matches the tolerance of the agents' own checks: a model that wraps
    the word in quotes, asterisks or a full stop still means it.
    """
    probe = text.strip().strip(_SENTINEL_WRAPPERS).upper()
    return any(probe.startswith(s.upper()) for s in sentinels)


@contextmanager
def guarded_tokens(*sentinels: str) -> Iterator[None]:
    """Stream tokens to the reader, but hold the first few back.

    An agent can reply with a sentinel instead of prose — the Inquiry
    Agent answers `INSUFFICIENT_CONTEXT` when the retrieved material does
    not cover the question, and the code turns that into a decline. If
    tokens went straight to the screen, the reader would watch the word
    INSUFFICIENT_CONTEXT appear and then be replaced, which is worse than
    waiting.

    So nothing is forwarded until enough characters have arrived to rule
    every sentinel out. That costs the length of the longest sentinel in
    latency, about twenty characters, and only on the first chunk.
    """
    downstream = llm.active_sink()
    if downstream is None:
        yield
        return

    longest = max((len(s) for s in sentinels), default=0)
    held: list[str] = []
    state = {"decided": False, "suppress": False}

    def forward(piece: str | None) -> None:
        # None means the attempt was abandoned; drop whatever was held and
        # tell the reader to start over.
        if piece is None:
            held.clear()
            state["decided"] = False
            state["suppress"] = False
            downstream(None)
            return

        if state["decided"]:
            if not state["suppress"]:
                downstream(piece)
            return

        held.append(piece)
        joined = "".join(held)
        if len(joined.lstrip()) < longest:
            return

        state["decided"] = True
        state["suppress"] = _looks_like(joined, sentinels)
        if not state["suppress"]:
            downstream(joined)

    with llm.token_sink(forward):
        yield

    # A reply shorter than the sentinel never reached the length that
    # triggers a decision, so it is still sitting in the buffer.
    if not state["decided"] and held:
        joined = "".join(held)
        if not _looks_like(joined, sentinels):
            downstream(joined)
