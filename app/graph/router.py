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
# after stripping punctuation, case and filler - never as substrings.
#
# This used to be four short lists, and "thanks man" fell through all of
# them to the model, which said `other`, which got the out-of-scope
# speech: a polite customer told the assistant could not help them. People
# say thanks, ok and how are you constantly, and none of it needs a model.
PLEASANTRIES: dict[str, str] = {
    **{k: "greeting" for k in (
        "hi", "hey", "hello", "yo", "hiya", "hii", "heyy", "howdy", "sup",
        "good morning", "good afternoon", "good evening", "good day",
        "hi there", "hello there", "hey there", "hi hi", "hello hello",
    )},
    **{k: "thanks" for k in (
        "thanks", "thank you", "thankyou", "thx", "ty", "ta", "cheers",
        "much appreciated", "many thanks", "thanks again", "appreciate it",
        "i appreciate it", "ok thanks", "okay thanks", "ok thank you",
        "okay thank you", "great thanks", "cool thanks", "perfect thanks",
        "alright thanks", "thanks ok", "thanks great", "thanks for your help",
        "thanks for the help", "thank you for your help",
    )},
    **{k: "farewell" for k in (
        "bye", "goodbye", "bye bye", "see you", "see ya", "see you later",
        "cya", "ttyl", "talk later", "take care", "have a good day",
        "have a nice day", "cheerio", "that's all", "thats all",
        "that is all", "that will be all", "nothing else", "nothing more",
        "all done", "all good", "im good", "i'm good", "no more questions",
        "no thanks", "no thank you", "no thats all", "no that's all",
    )},
    # Said to acknowledge an answer. Not "yes"/"yeah"/"sure": those answer
    # a question, and without the question they mean nothing.
    **{k: "ack" for k in (
        "ok", "okay", "k", "kk", "cool", "great", "nice", "alright",
        "got it", "understood", "noted", "perfect", "awesome", "fine",
        "good", "lol", "haha", "hmm", "i see", "makes sense", "sounds good",
        "oh ok", "oh okay", "ok cool", "okay cool", "ok great", "oh i see",
    )},
    **{k: "howareyou" for k in (
        "how are you", "how are you doing", "how r u", "how are u",
        "hows it going", "how's it going", "how are things", "whats up",
        "what's up", "how is your day", "how is it going", "you good",
    )},
    **{k: "identity" for k in (
        "who are you", "what are you", "are you a bot", "are you a robot",
        "are you human", "are you real", "are you ai", "are you an ai",
        "are you chatgpt", "what is your name", "whats your name",
        "what's your name", "who am i talking to", "who is this",
        "who am i speaking to", "are you a real person",
    )},
    **{k: "sorry" for k in (
        "sorry", "my bad", "oops", "apologies", "sorry about that",
        "im sorry", "i'm sorry", "sorry man", "sorry again",
    )},
}

# Words that add warmth and no meaning. "Thanks man" is "thanks".
FILLER_WORDS = frozenset({
    "man", "bro", "mate", "dude", "buddy", "pal", "friend", "sir", "maam",
    "madam", "guys", "please", "pls", "again", "heaps", "very", "so",
    "really", "just", "then", "there",
})

# Longest pleasantry worth checking, after filler is removed. Anything
# longer has content in it.
MAX_PLEASANTRY_WORDS = 5

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

    # "a lot" and "so much" are filler too, but they are two words that
    # only mean anything together, so they go before the words are split.
    squeezed = f" {normalised} "
    for phrase in (" a lot ", " so much ", " very much ", " a ton "):
        squeezed = squeezed.replace(phrase, " ")
    words = [w for w in squeezed.split() if w not in FILLER_WORDS]
    if not words or len(words) > MAX_PLEASANTRY_WORDS:
        return None

    kind = PLEASANTRIES.get(" ".join(words)) or PLEASANTRIES.get(normalised)
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
    if fast is not None and fast[2] == "ack":
        # "Ok" after "Would 10am suit you?" confirms a slot. It is only an
        # acknowledgement when the last thing said was not a question
        # about a booking, so that case is left to the model, which can
        # see the conversation.
        from app.agents.booking import _looks_mid_booking

        if _looks_mid_booking(history):
            fast = None
    if fast is not None:
        intent, reason, _kind = fast
        return {
            "intent": intent,
            "secondary_intent": None,
            "confidence": 1.0,
            "routing_reason": reason,
            "routed_by": "keyword",
        }

    # Leaving a booking is part of the booking, not small talk. "Nah
    # sorry, I'll book later" names no service, no day and no appointment,
    # so left to the model it came back as `other` and got the
    # out-of-scope speech, a few seconds after being asked for a name.
    # Decided in code: it needs no judgement, only the last thing said.
    from app.agents.booking import _looks_mid_booking, backs_out

    mid_booking = _looks_mid_booking(history)
    if mid_booking and backs_out(message, booking_in_progress=True):
        return {
            "intent": "booking",
            "secondary_intent": None,
            "confidence": 1.0,
            "routing_reason": "backing out of a booking in progress",
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
