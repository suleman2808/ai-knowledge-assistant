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
import queue
import re
import threading
import time
from typing import Any

from app import llm
from app.agents.base import AgentResponse
from app.agents.booking import handle_booking
from app.agents.complaint import handle_complaint
from app.agents.inquiry import answer_inquiry
from app.graph.router import classify, select_agent
from app.graph.state import AssistantState, new_state

logger = logging.getLogger(__name__)

# Words that commonly surround a phone number when someone is simply
# identifying themselves, and carry no request of their own.
PLEASANTRIES_AROUND_A_NUMBER = frozenset(
    """
    hi hello hey this is my number its it's i'm im here thanks thank you
    please mobile cell phone and the me a on at speaking calling again back
    """.split()
)


GREETING = (
    "Hello - I'm the assistant for Riverbend Diagnostics. I can answer "
    "questions about our services, prices, hours and policies, book you an "
    "appointment, or pass on a complaint. What can I help with?"
)

FAREWELL = (
    "You're welcome - take care. If you need anything else, just ask, or "
    "call the laboratory on (503) 555-0142."
)

OUT_OF_SCOPE = (
    "That's outside what I can help with, I'm afraid - I only handle things "
    "to do with Riverbend Diagnostics. I can answer questions about our "
    "services, prices, hours and policies, book an appointment, or pass on a "
    "complaint."
)

# What to append when a second intent was present but not acted on. These
# offer rather than assert: the secondary genuinely has not been handled,
# and claiming otherwise would be a lie the patient discovers later.
# Asking for a person is not a question about the laboratory, so no
# agent owns it and the router has nowhere to put it. Left alone, the
# assistant answers whatever else was in the message and ignores the
# request entirely - which is the worst thing a customer-facing system
# can do, and exactly what it did when asked "can I talk with a human
# also how much is a full blood count".
#
# Matched in code rather than by the model: wanting a person is not a
# judgement call, and the cost of missing it is someone deciding the
# company is hiding behind a robot.
# Phrases rather than a pattern: these are the things people actually
# type, they are readable by whoever maintains this, and there is no
# clever matching to get wrong.
_PERSON_WORDS = (
    "human", "person", "someone", "somebody", "a real agent",
    "receptionist", "staff member", "advisor", "adviser",
)
_REACH_VERBS = (
    "talk to", "talk with", "speak to", "speak with", "chat to", "chat with",
    "connect me to", "connect me with", "put me through to", "transfer me to",
    "pass me to", "get me",
)


def wants_a_person(message: str) -> bool:
    """Whether the message asks to be put in touch with a human being."""
    text = " ".join(message.lower().split())

    if any(f"{verb} a {word}" in text or f"{verb} an {word}" in text
           or f"{verb} the {word}" in text or f"{verb} {word}" in text
           for verb in _REACH_VERBS for word in _PERSON_WORDS):
        return True

    return any(
        phrase in text
        for phrase in (
            "real person", "actual person", "a human please",
            "human instead", "not a bot", "not a robot",
            "is there anyone i can", "can i call someone",
        )
    )


PERSON_NOTE = (
    "\n\nAnd yes - you can always speak to someone. Call us on (503) 555-0142 "
    "during collection hours and a member of the team will help you directly."
)

PHONE = "(503) 555-0142"

HANDOFF_NOTES = {
    "complaint": (
        "\n\nYou also mentioned something that went wrong. I haven't logged "
        "that as a complaint yet - say the word and I will, or I can pass it "
        "to our practice manager."
    ),
    "booking": (
        "\n\nYou also asked about an appointment. Tell me what you'd like and "
        "when, and I'll get that booked."
    ),
    "inquiry": (
        "\n\nYou asked something else in there too - ask me again on its own "
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


def identify(state: AssistantState) -> dict[str, Any]:
    """Recognise a returning customer from a phone number in the conversation.

    Runs before the router and costs nothing: a regular expression and one
    indexed lookup, no model call. Everything downstream can then assume
    `state["patient"]` is either a known customer or None.

    The whole conversation is searched, not just the latest message,
    because someone gives their number once and expects to be known for
    the rest of the exchange — not only in the turn where they typed it.
    """
    from app.integrations.patients import find, phone_in

    message = state.get("message", "")
    key = phone_in(message)

    if not key:
        for turn in reversed(state.get("history") or []):
            if turn.get("role") != "user":
                continue
            key = phone_in(turn.get("content", ""))
            if key:
                break

    if not key:
        return {"patient": None}

    patient = find(key)
    if patient:
        logger.info(
            "Recognised returning customer (%d previous booking(s))",
            patient["visit_count"],
        )
    return {"patient": patient}


def booking_node(state: AssistantState) -> dict[str, Any]:
    """Run the Booking Agent."""
    return _apply(
        state,
        handle_booking(
            state["message"],
            history=state.get("history"),
            patient=state.get("patient"),
            # Cancelling needs the event id of what was booked, and the
            # session is how that is found again.
            session_id=state.get("session_id", ""),
        ),
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


def _spoken_phone(digits: str) -> str:
    """Format a stored number the way it is written on a card."""
    bare = "".join(c for c in digits if c.isdigit())
    if len(bare) == 10:
        return f"({bare[:3]}) {bare[3:6]}-{bare[6:]}"
    return digits


def _own_details_answer(state: AssistantState) -> str:
    """Answer "do you already have my number?" from what is actually held.

    Everything here comes from this conversation or from the patient
    record the identify node matched - never from the documents, which is
    where this question used to go and why it was refused.
    """
    from app.integrations.analytics import latest_booking

    message = state.get("message", "").lower()
    patient = state.get("patient") or {}
    booking = latest_booking(state.get("session_id", "")) or {}

    name = patient.get("name") or booking.get("patient_name") or ""
    # The booking stores the number as the patient typed it; the patient
    # record keeps a normalised key for matching. Reading the key back to
    # them ("5035550122") looks like a database field, so the typed form
    # wins and the key is only ever shown formatted.
    phone = booking.get("phone") or _spoken_phone(patient.get("phone") or "")

    wants_appointment = any(
        word in message for word in ("appointment", "booking", "slot")
    )
    wants_phone = any(word in message for word in ("number", "phone", "mobile"))
    wants_name = "name" in message

    known: list[str] = []
    if wants_appointment and booking.get("starts_at"):
        from app.agents.booking import _describe_booking

        described = _describe_booking(booking).strip()
        if described:
            known.append(f"you're booked in for {described}")
    if wants_phone and phone:
        known.append(f"I have {phone} as your contact number")
    if wants_name and name:
        known.append(f"you're down as {name}")

    if not known:
        # Asked about an appointment there isn't one of. Listing their
        # name and number instead would answer a question nobody asked.
        if wants_appointment:
            return (
                "You don't have an appointment booked in this conversation. "
                "If you'd like one, tell me what you need and which day suits "
                "you, and I'll get it arranged."
            )

        # Nothing specific was asked for: say what there is, which is the
        # honest answer to "what do you have".
        if phone or name:
            held = " and ".join(
                part for part in (
                    f"your name as {name}" if name else "",
                    f"your number as {phone}" if phone else "",
                ) if part
            )
            return (
                f"Yes - I have {held} from this conversation. I'll use those "
                "unless you tell me otherwise."
            )
        return (
            "Not yet - you haven't given me a name or a number in this "
            "conversation, so there's nothing on file. Tell me either and "
            "I'll keep it with your booking."
        )

    # "Yes" answers a yes/no question. "When is my appointment" is not
    # one, and "Yes - you're booked in for..." reads as a non sequitur.
    answering_yes_no = any(
        opener in message
        for opener in ("do you", "have you", "you already", "you know",
                       "right?", "correct?", "did you")
    )
    sentence = ", and ".join(known)
    if answering_yes_no:
        return f"Yes - {sentence}."
    return sentence[:1].upper() + sentence[1:] + "."


def other_node(state: AssistantState) -> dict[str, Any]:
    """Handle greetings, thanks and anything outside the laboratory's scope.

    A small node rather than a fourth agent, because there is nothing to
    retrieve, book or record. Routing "hi" to the RAG agent would produce
    "I don't have that in the laboratory's information", which is technically
    true and a terrible first impression.

    `other` covers two quite different cases, and answering them the same
    way is wrong. A greeting deserves a greeting; a question about a
    cinema deserves to be told plainly that it is out of scope. The
    keyword fast-path only ever fires on pleasantries, so how the message
    was routed tells us which case this is.
    """
    reason = state.get("routing_reason", "")
    routed_by = state.get("routed_by", "")

    patient = state.get("patient")

    if "own details" in reason:
        kind = "own_details"
        answer = _own_details_answer(state)
    elif routed_by == "keyword":
        kind = "farewell" if ("farewell" in reason or "thanks" in reason) else "greeting"
        answer = FAREWELL if kind == "farewell" else GREETING
        # Greet a returning customer by name. Only on a greeting: opening
        # "Welcome back, Sarah" in the middle of a conversation reads as a
        # glitch rather than a courtesy.
        if kind == "greeting" and patient and patient.get("name"):
            first_name = patient["name"].split()[0]
            answer = f"Welcome back, {first_name} - " + GREETING[len("Hello - "):]
    elif patient and _is_mostly_contact_details(state.get("message", "")):
        # Someone handing over their number is identifying themselves, not
        # asking about something off-topic. Answering "that's outside what
        # I can help with" to "hi, this is 503-555-0180" is the worst
        # possible reply: they have just told us who they are.
        kind = "identified"
        first_name = (patient.get("name") or "").split()
        greeting = f"Thanks, {first_name[0]} - " if first_name else "Thanks - "
        previous = patient.get("visit_count") or 0
        seen = (
            f"good to see you again. I have {previous} previous booking"
            f"{'s' if previous != 1 else ''} on file. "
            if previous
            else "I've found your details. "
        )
        answer = f"{greeting}{seen}What can I help with today?"
    else:
        # Classified as `other` by the model: a real message about
        # something the laboratory does not do.
        kind = "out_of_scope"
        answer = OUT_OF_SCOPE

    return {
        "answer": answer,
        "sources": [],
        "success": True,
        "needs_followup": False,
        "agent_metadata": {
            "handled_by": "other",
            "kind": kind,
            "reason": reason,
            "recognised": bool(patient),
        },
    }


def _is_mostly_contact_details(message: str) -> bool:
    """Whether a message is someone giving their number and little else.

    "hi, this is 503-555-0180" qualifies; "my number is 503-555-0180, and
    what time does the cinema open?" does not — that still deserves the
    out-of-scope answer. The test is whether anything substantial remains
    once the number and the usual pleasantries are taken out.

    Word matching rather than pattern substitution, deliberately: an
    earlier version built the patterns by string interpolation and wrote
    literal control characters into the file instead of word boundaries,
    so nothing was ever removed and the check silently always failed.
    """
    from app.integrations.phone import PHONE_IN_TEXT

    if not PHONE_IN_TEXT.search(message or ""):
        return False

    remainder = PHONE_IN_TEXT.sub(" ", message or "").lower()
    words = re.findall(r"[a-z']+", remainder)
    substantial = [w for w in words if w not in PLEASANTRIES_AROUND_A_NUMBER]
    return sum(len(w) for w in substantial) <= 3



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

    # Appended rather than substituted: the message usually carries a real
    # question alongside the request, and answering it and offering the
    # phone number are both right. Skipped when the answer already gives
    # the number, which the refusals and the complaint replies do.
    asked_for_a_person = wants_a_person(state.get("message", ""))
    if asked_for_a_person and PHONE not in answer:
        answer = f"{answer}{PERSON_NOTE}"

    turn = {
        "session_id": state.get("session_id", ""),
        "message": state.get("message", ""),
        "intent": state.get("intent", "inquiry"),
        "secondary_intent": secondary,
        "confidence": state.get("confidence", 0.0),
        "routed_by": state.get("routed_by", ""),
        "routing_reason": state.get("routing_reason", ""),
        "patient": state.get("patient"),
        "answer": answer,
        "success": state.get("success", True),
        "needs_followup": state.get("needs_followup", False),
        "sources": state.get("sources", []),
        "agent_metadata": {
            **state.get("agent_metadata", {}),
            # Worth seeing on the dashboard: a run of these is the
            # assistant failing at something, not patients being odd.
            **({"asked_for_a_person": True} if asked_for_a_person else {}),
        },
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

    builder.add_node("identify", identify)
    builder.add_node("router", classify)
    builder.add_node("booking", booking_node)
    builder.add_node("inquiry", inquiry_node)
    builder.add_node("complaint", complaint_node)
    builder.add_node("other", other_node)
    builder.add_node("finalise", finalise)

    builder.add_edge(START, "identify")
    builder.add_edge("identify", "router")

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
    log: bool = True,
) -> dict[str, Any]:
    """Run one turn through the graph and record it.

    The entry point used by the CLI and the API. Returns the `turn`
    record: everything about what happened, ready to serialise.

    Logging happens here rather than inside a node, which keeps the graph
    itself free of side effects beyond what the agents do. The graph can
    then be invoked in tests, notebooks or evaluations without polluting
    the analytics store — pass `log=False`, or call `get_graph().invoke()`
    directly.

    Args:
        log: Record the turn in analytics. Scripts that replay test
            probes pass False so evaluation traffic does not skew the
            real numbers.
    """
    started = time.perf_counter()
    final = get_graph().invoke(
        new_state(message, history=history, session_id=session_id)
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    turn = final.get("turn", {"answer": final.get("answer", ""), "intent": "unknown"})
    turn["latency_ms"] = latency_ms

    if log:
        from app.integrations.analytics import log_turn

        log_turn(turn, latency_ms=latency_ms)
    return turn


def stream(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
    session_id: str = "",
    log: bool = True,
) -> Any:
    """Run a turn, yielding progress and the answer as it is written.

    Two kinds of event come out of here. Node events say which stage is
    running — routing, then searching, then writing — which is what fills
    the seconds before any text exists. Token events carry the answer
    itself, a few characters at a time, for the agents that produce prose
    a reader sees verbatim.

    The graph runs on a worker thread. It has to: tokens arrive *during* a
    node, and a generator that is itself driving the graph cannot yield
    anything until that node returns. The thread pushes events onto a
    queue and this generator drains it, so the first characters reach the
    browser while the model is still writing the rest.

    Yields:
        `{"event": "node", ...}` and `{"event": "token", "text": ...}`,
        then one `{"event": "done", "turn": {...}}`.
    """
    started = time.perf_counter()
    events: queue.Queue[dict[str, Any] | None] = queue.Queue()
    outcome: dict[str, Any] = {}

    def on_token(piece: str | None) -> None:
        # None means the attempt was abandoned mid-reply and is about to
        # be retried, so whatever the reader has seen is wrong.
        events.put({"event": "reset"} if piece is None else
                   {"event": "token", "text": piece})

    def work() -> None:
        final: dict[str, Any] = {}
        try:
            with llm.token_sink(on_token):
                for update in get_graph().stream(
                    new_state(message, history=history, session_id=session_id),
                    stream_mode="updates",
                ):
                    for node, delta in update.items():
                        delta = delta or {}
                        final.update(delta)
                        event = {"event": "node", "node": node}
                        for key in ("intent", "secondary_intent", "confidence",
                                    "routed_by"):
                            if key in delta:
                                event[key] = delta[key]
                        events.put(event)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            outcome["error"] = exc
        else:
            outcome["final"] = final
        finally:
            events.put(None)

    worker = threading.Thread(target=work, name="assistant-turn", daemon=True)
    worker.start()

    while True:
        event = events.get()
        if event is None:
            break
        yield event

    worker.join()
    if "error" in outcome:
        raise outcome["error"]

    final = outcome.get("final", {})
    latency_ms = int((time.perf_counter() - started) * 1000)
    turn = final.get("turn", {"answer": final.get("answer", ""), "intent": "unknown"})
    turn["latency_ms"] = latency_ms

    yield {"event": "done", "turn": turn}

    if log:
        from app.integrations.analytics import log_turn

        log_turn(turn, latency_ms=latency_ms)
