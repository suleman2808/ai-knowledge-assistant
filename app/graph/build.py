"""Assemble the LangGraph.

    entry -> router -> [booking | inquiry | complaint | other] -> finalise -> END

## Why a graph rather than an if/else

The dispatch itself could be four lines of `if`. What the graph buys is
everything around it:

- **The structure is inspectable.** `compiled.get_graph().draw_mermaid()`
  emits the diagram in this project's README. It is generated from the
  running code, so it cannot drift from what actually executes — unlike
  a diagram drawn by hand in a document.
- **State is explicit and typed.** Every node declares what it reads and
  returns a partial update. With if/else, state is whatever local
  variables happen to be in scope, and tracing what a branch mutated
  means reading the whole function.
- **Nodes are independently testable.** Each is a pure
  `state -> partial state` function, runnable without the graph.
- **It extends without rewriting.** Adding human handoff, a retry loop,
  a clarification cycle, or persistence between turns means adding a node
  and an edge. In an if/else, each of those becomes a nested branch
  inside a function that grows until nobody wants to touch it.
- **Execution is observable.** Streaming per-node updates, which step 8's
  UI uses to show which specialist is working, is a property of the graph
  rather than something to be hand-rolled.

The honest version of the trade-off: for exactly three static branches,
if/else is simpler, and choosing LangGraph here is a bet that the system
grows. That bet is why the abstraction is worth its cost — and the moment
a fourth intent or a handoff loop appears, the if/else version starts
paying interest.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base import AgentResponse
from app.agents.booking import handle_booking
from app.agents.complaint import handle_complaint
from app.agents.inquiry import answer_inquiry
from app.graph.router import classify, select_agent
from app.graph.state import AssistantState, new_state

logger = logging.getLogger(__name__)

GREETING = (
    "Hello — I'm the assistant for Riverbend Dental Care. I can answer "
    "questions about our services, prices, hours and policies, book you an "
    "appointment, or pass on a complaint. What can I help with?"
)

FAREWELL = (
    "You're welcome — take care. If you need anything else, just ask, or "
    "call the clinic on (503) 555-0142."
)

OUT_OF_SCOPE = (
    "That's outside what I can help with, I'm afraid — I only handle things "
    "to do with Riverbend Dental Care. I can answer questions about our "
    "services, prices, hours and policies, book an appointment, or pass on a "
    "complaint."
)

# What to append when a second intent was present but not acted on. These
# offer rather than assert: the secondary genuinely has not been handled,
# and claiming otherwise would be a lie the patient discovers later.
HANDOFF_NOTES = {
    "complaint": (
        "\n\nYou also mentioned something that went wrong. I haven't logged "
        "that as a complaint yet — say the word and I will, or I can pass it "
        "to our practice manager."
    ),
    "booking": (
        "\n\nYou also asked about an appointment. Tell me what you'd like and "
        "when, and I'll get that booked."
    ),
    "inquiry": (
        "\n\nYou asked something else in there too — ask me again on its own "
        "and I'll answer it properly."
    ),
}


def _apply(state: AssistantState, response: AgentResponse) -> dict[str, Any]:
    """Convert an `AgentResponse` into a state update."""
    return {
        "answer": response.answer,
        "sources": response.sources,
        "success": response.success,
        "needs_followup": response.needs_followup,
        "agent_metadata": response.metadata,
    }


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------


def booking_node(state: AssistantState) -> dict[str, Any]:
    """Run the Booking Agent."""
    return _apply(
        state,
        handle_booking(state["message"], history=state.get("history")),
    )


def inquiry_node(state: AssistantState) -> dict[str, Any]:
    """Run the Inquiry Agent."""
    return _apply(
        state,
        answer_inquiry(state["message"], history=state.get("history")),
    )


def complaint_node(state: AssistantState) -> dict[str, Any]:
    """Run the Complaint Agent."""
    return _apply(
        state,
        handle_complaint(state["message"], history=state.get("history")),
    )


def other_node(state: AssistantState) -> dict[str, Any]:
    """Handle greetings, thanks and anything outside the clinic's scope.

    A small node rather than a fourth agent, because there is nothing to
    retrieve, book or record. Routing "hi" to the RAG agent would produce
    "I don't have that in the clinic's information", which is technically
    true and a terrible first impression.

    `other` covers two quite different cases, and answering them the same
    way is wrong. A greeting deserves a greeting; a question about a
    cinema deserves to be told plainly that it is out of scope. The
    keyword fast-path only ever fires on pleasantries, so how the message
    was routed tells us which case this is.
    """
    reason = state.get("routing_reason", "")
    routed_by = state.get("routed_by", "")

    if routed_by == "keyword":
        kind = "farewell" if ("farewell" in reason or "thanks" in reason) else "greeting"
        answer = FAREWELL if kind == "farewell" else GREETING
    else:
        # Classified as `other` by the model: a real message about
        # something the clinic does not do.
        kind = "out_of_scope"
        answer = OUT_OF_SCOPE

    return {
        "answer": answer,
        "sources": [],
        "success": True,
        "needs_followup": False,
        "agent_metadata": {"handled_by": "other", "kind": kind, "reason": reason},
    }


def finalise(state: AssistantState) -> dict[str, Any]:
    """Append any secondary-intent note and assemble the turn record.

    A single exit node means every path — including failures — produces
    the same shaped output, which is what lets step 7 log analytics in
    one place instead of at five call sites.
    """
    answer = state.get("answer", "")
    secondary = state.get("secondary_intent")

    # Only offer the note when the primary agent actually finished. Adding
    # "you also mentioned a complaint" underneath "I couldn't reach my
    # language service" would be absurd.
    if secondary and state.get("success", True) and secondary in HANDOFF_NOTES:
        answer = f"{answer}{HANDOFF_NOTES[secondary]}"

    turn = {
        "session_id": state.get("session_id", ""),
        "message": state.get("message", ""),
        "intent": state.get("intent", "inquiry"),
        "secondary_intent": secondary,
        "confidence": state.get("confidence", 0.0),
        "routed_by": state.get("routed_by", ""),
        "routing_reason": state.get("routing_reason", ""),
        "answer": answer,
        "success": state.get("success", True),
        "needs_followup": state.get("needs_followup", False),
        "sources": state.get("sources", []),
        "agent_metadata": state.get("agent_metadata", {}),
    }

    return {"answer": answer, "turn": turn}


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

_compiled: Any = None


def build_graph() -> Any:
    """Construct and compile the graph.

    Returns:
        A compiled LangGraph, invokable with `.invoke(state)`.
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(AssistantState)

    builder.add_node("router", classify)
    builder.add_node("booking", booking_node)
    builder.add_node("inquiry", inquiry_node)
    builder.add_node("complaint", complaint_node)
    builder.add_node("other", other_node)
    builder.add_node("finalise", finalise)

    builder.add_edge(START, "router")

    # The conditional edge is the dispatch. The mapping is explicit rather
    # than relying on the returned string matching a node name by
    # convention, so a typo fails at build time instead of at runtime.
    builder.add_conditional_edges(
        "router",
        select_agent,
        {
            "booking": "booking",
            "inquiry": "inquiry",
            "complaint": "complaint",
            "other": "other",
        },
    )

    for node in ("booking", "inquiry", "complaint", "other"):
        builder.add_edge(node, "finalise")

    builder.add_edge("finalise", END)

    return builder.compile()


def get_graph() -> Any:
    """Return the compiled graph, building it once per process."""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def run(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Run one turn through the graph.

    The convenience entry point used by the CLI, the tests and, in step 8,
    the API. Returns the `turn` record: everything about what happened,
    ready to serialise or log.
    """
    final = get_graph().invoke(
        new_state(message, history=history, session_id=session_id)
    )
    return final.get("turn", {"answer": final.get("answer", ""), "intent": "unknown"})
