"""Intent classification — the entry node of the graph.

Two paths, tried in order:

1. **A keyword fast-path** for messages whose intent is unambiguous from
   their form alone: "hi", "thanks", "bye". These cost nothing, resolve
   in microseconds, and are the most common messages a public chat box
   receives. Spending an LLM call and 400ms on "thanks" is waste.
2. **An LLM classifier** for everything else, using the small model at
   low reasoning effort.

The fast-path is deliberately narrow. It only fires on whole-message
matches, never on substrings, because keyword routing is where naive
implementations break: "I want to cancel my complaint about the
cancellation fee" contains three trigger words and means one thing. Any
message with real content goes to the model.

When classification fails entirely, the fallback is `inquiry` rather than
`other`. Inquiry is the only agent that can answer a general question, and
it refuses safely when it cannot — so a misroute there degrades into "I
don't know", while a misroute to booking or complaint produces a
confusing interrogation or a spurious complaint record.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.graph.state import INTENTS, AssistantState, Intent
from app.llm import LLMError, complete_json
from app.prompts import render

logger = logging.getLogger(__name__)

# Whole messages, normalised, that need no model call. Matched exactly
# after stripping punctuation and case — never as substrings.
PLEASANTRIES: dict[str, str] = {
    **{k: "greeting" for k in (
        "hi", "hey", "hello", "yo", "hiya", "good morning", "good afternoon",
        "good evening", "hi there", "hello there",
    )},
    **{k: "thanks" for k in (
        "thanks", "thank you", "thanks a lot", "thank you so much", "ta",
        "cheers", "much appreciated", "thankyou",
    )},
    **{k: "farewell" for k in (
        "bye", "goodbye", "bye bye", "see you", "see ya", "cheerio",
        "that's all", "thats all", "nothing else", "no thanks", "no thank you",
    )},
}

# Longest pleasantry worth checking. Anything longer has content in it.
MAX_PLEASANTRY_WORDS = 3

# "I'd like to speak to a human being please" is nine. Longer than this
# and there is something else in the message worth routing on.
MAX_PERSON_REQUEST_WORDS = 10

# Questions about the patient's own details. These are not questions
# about the laboratory, and sending them to the documents produces "I
# don't have that in the laboratory's information" about a phone number
# the patient typed thirty seconds earlier - which reads as the system
# not listening rather than as it being careful.
_OWN_DETAILS = (
    "my number", "my phone", "my mobile", "my name", "my details",
    "my contact", "my booking", "my appointment", "my slot",
)
_ASKING_ABOUT = (
    "you already", "you know", "do you have", "have you got", "what is",
    "what's", "whats", "when is", "when's", "whens", "remember",
    "on file", "got my", "know my", "have my", "confirm",
)


def _asks_about_their_own_details(message: str) -> bool:
    """Whether the message asks what the assistant holds about them."""
    text = " ".join(message.lower().split())
    return (any(subject in text for subject in _OWN_DETAILS)
            and any(verb in text for verb in _ASKING_ABOUT))


def _normalise(message: str) -> str:
    """Lower-case and strip punctuation for exact matching."""
    return "".join(
        c for c in message.lower().strip() if c.isalnum() or c.isspace()
    ).strip()


def _fast_path(message: str) -> tuple[Intent, str, str] | None:
    """Classify without a model call, or return None to defer.

    Returns:
        `(intent, reason, kind)` when confident, otherwise None.
    """
    normalised = _normalise(message)
    if not normalised:
        return ("other", "empty message", "empty")

    if _asks_about_their_own_details(message):
        return ("other", "asking about their own details", "own_details")

    # Only when that is the whole message. "Can I talk to a human, and how
    # much is a full blood count" has a real question in it and belongs
    # with the Inquiry Agent; the offer is added at the exit node either
    # way.
    from app.graph.build import wants_a_person

    if wants_a_person(message) and len(normalised.split()) <= MAX_PERSON_REQUEST_WORDS:
        return ("other", "asking for a person", "person")

    if len(normalised.split()) > MAX_PLEASANTRY_WORDS:
        return None

    kind = PLEASANTRIES.get(normalised)
    if kind:
        return ("other", f"{kind} with no request", kind)
    return None


def _coerce_intent(value: object, default: Intent = "inquiry") -> Intent:
    """Validate an intent from the model, falling back safely."""
    text = str(value or "").strip().lower()
    return text if text in INTENTS else default  # type: ignore[return-value]


def classify(state: AssistantState) -> AssistantState:
    """Route a message to a specialist. The graph's entry node.

    Returns a partial state update carrying `intent`, `secondary_intent`,
    `confidence`, `routing_reason` and `routed_by`.
    """
    message = state.get("message", "")
    history = state.get("history") or []

    fast = _fast_path(message)
    if fast is not None:
        intent, reason, _kind = fast
        return {
            "intent": intent,
            "secondary_intent": None,
            "confidence": 1.0,
            "routing_reason": reason,
            "routed_by": "keyword",
        }

    lines = []
    for turn in history[-6:]:
        role = "Patient" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    rendered_history = "\n".join(lines) if lines else "(no previous messages)"

    try:
        result = complete_json(
            render("router", message=message, history=rendered_history),
            model=settings.router_model,
            reasoning_effort=settings.router_reasoning_effort,
            max_tokens=300,
        )
    except LLMError as exc:
        # Inquiry is the safe default: it is the only agent that can
        # refuse gracefully when it turns out to be the wrong choice.
        logger.error("Router classification failed, defaulting to inquiry: %s", exc)
        return {
            "intent": "inquiry",
            "secondary_intent": None,
            "confidence": 0.0,
            "routing_reason": f"classification unavailable: {exc}",
            "routed_by": "fallback",
        }

    intent = _coerce_intent(result.get("intent"))

    secondary_raw = result.get("secondary_intent")
    secondary: Intent | None = None
    if secondary_raw:
        candidate = _coerce_intent(secondary_raw, default="other")
        # A secondary equal to the primary is noise, and "other" as a
        # secondary carries no information worth acting on.
        if candidate != intent and candidate != "other":
            secondary = candidate

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(result.get("reason") or "").strip()[:120]

    logger.info(
        "Routed to %s (confidence=%.2f, secondary=%s): %s",
        intent, confidence, secondary, reason,
    )

    return {
        "intent": intent,
        "secondary_intent": secondary,
        "confidence": confidence,
        "routing_reason": reason,
        "routed_by": "llm",
    }


def select_agent(state: AssistantState) -> str:
    """Conditional edge: name the node to run next.

    Separated from `classify` because LangGraph edges must be pure
    functions of state. Keeping the decision and the dispatch apart means
    the classification can be tested without a graph, and the graph's
    wiring can be tested without a model.
    """
    return state.get("intent", "inquiry")
