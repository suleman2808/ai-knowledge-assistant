"""The state that flows through the graph.

LangGraph passes a single mutable state object between nodes. Each node
receives the whole state and returns a partial update, which LangGraph
merges in. Declaring that shape explicitly, in one place, is most of what
makes a graph readable six months later: you can see exactly what a node
is allowed to know and what it is expected to produce.

The state is deliberately flat and JSON-serialisable. Nothing in it is an
object with behaviour, because state that can be printed, logged and
replayed is state you can debug.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# The four destinations the router can choose. `other` exists because a
# real chat box receives "hi", "thanks" and "are you a robot", and routing
# those to a RAG agent produces a refusal where a greeting belongs.
Intent = Literal["booking", "inquiry", "complaint", "other"]

INTENTS: tuple[Intent, ...] = ("booking", "inquiry", "complaint", "other")


class AssistantState(TypedDict, total=False):
    """Everything a turn needs, and everything it produces.

    Grouped by who writes it, because that ordering is what a reader needs
    in order to follow the flow.
    """

    # --- Written by the caller -------------------------------------------
    message: str
    """The patient's message, verbatim."""

    history: list[dict[str, str]]
    """Prior turns as {"role", "content"} dicts, oldest first."""

    session_id: str
    """Groups turns into a conversation, for analytics."""

    # --- Written by the router -------------------------------------------
    intent: Intent
    """Where this message was dispatched."""

    secondary_intent: Intent | None
    """A second intent present in the message but not acted on.

    A patient can be angry about a wait *and* want to rebook in the same
    sentence. The primary intent is handled properly; the secondary is
    acknowledged rather than silently dropped.
    """

    confidence: float
    """Router's confidence in the primary intent, 0-1."""

    routing_reason: str
    """Short explanation of the classification, for logs and debugging."""

    routed_by: str
    """"keyword" or "llm" — which path made the decision."""

    # --- Written by the specialist agent ----------------------------------
    answer: str
    sources: list[dict[str, Any]]
    success: bool
    needs_followup: bool
    agent_metadata: dict[str, Any]

    # --- Written by the finalise node -------------------------------------
    turn: dict[str, Any]
    """The complete record of this turn, ready for the analytics layer."""


def new_state(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
    session_id: str = "",
) -> AssistantState:
    """Build the initial state for a turn."""
    return AssistantState(
        message=message,
        history=history or [],
        session_id=session_id,
    )
