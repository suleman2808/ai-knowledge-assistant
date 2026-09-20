"""Tests for the router and the graph wiring.

Split in two, matching the design:

- Router tests check classification logic — the fast-path, validation of
  the model's output, and the fallback when classification fails.
- Graph tests check that dispatch, convergence and the finalise step do
  what the wiring claims, with every agent stubbed.

Nothing here calls a model or a database.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.graph import build as build_module
from app.graph.build import build_graph, finalise, other_node, run
from app.graph.router import classify, select_agent
from app.graph.state import new_state
from app.llm import LLMError


# ---------------------------------------------------------------------------
# Router: the keyword fast-path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message", ["hi", "Hello!", "  thanks  ", "Thank you.", "bye", "cheers", "no thanks"]
)
def test_pleasantries_skip_the_model(message: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The commonest messages a public chat box gets must be free."""

    def _never(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("the fast-path should have handled this")

    monkeypatch.setattr("app.graph.router.complete_json", _never)

    result = classify(new_state(message))

    assert result["intent"] == "other"
    assert result["routed_by"] == "keyword"
    assert result["confidence"] == 1.0


@pytest.mark.parametrize(
    "message",
    [
        "I want to cancel my complaint about the cancellation fee",
        "thanks, but I need to book an appointment",
        "hi, how much is a crown",
        "bye the way what are your hours",
    ],
)
def test_fast_path_defers_anything_with_content(
    message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyword routing must never fire on a substring.

    This is where naive keyword routers break: a message containing
    "cancel", "complaint" and "thanks" means one thing, and only reading
    it properly reveals which.
    """
    called = False

    def _classifier(*_a, **_k) -> dict[str, Any]:  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        return {"intent": "inquiry", "secondary_intent": None,
                "confidence": 0.9, "reason": "stub"}

    monkeypatch.setattr("app.graph.router.complete_json", _classifier)

    classify(new_state(message))

    assert called is True, "should have deferred to the model"


def test_empty_message_is_handled_without_a_model_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.graph.router.complete_json",
        lambda *_a, **_k: pytest.fail("should not be called"),
    )

    result = classify(new_state("   "))

    assert result["intent"] == "other"


# ---------------------------------------------------------------------------
# Router: validating the model's output
# ---------------------------------------------------------------------------


def stub_router(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    monkeypatch.setattr("app.graph.router.complete_json", lambda *_a, **_k: payload)


def test_unknown_intent_falls_back_to_inquiry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid JSON does not mean a valid intent."""
    stub_router(monkeypatch, {"intent": "refund_request", "confidence": 0.9})

    result = classify(new_state("something"))

    assert result["intent"] == "inquiry"


def test_secondary_equal_to_primary_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_router(
        monkeypatch,
        {"intent": "booking", "secondary_intent": "booking", "confidence": 0.9},
    )

    result = classify(new_state("book me in"))

    assert result["secondary_intent"] is None


def test_secondary_other_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """"Also said hello" is not a second intent worth acting on."""
    stub_router(
        monkeypatch,
        {"intent": "booking", "secondary_intent": "other", "confidence": 0.9},
    )

    assert classify(new_state("hi, book me in"))["secondary_intent"] is None


def test_genuine_secondary_intent_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_router(
        monkeypatch,
        {"intent": "complaint", "secondary_intent": "booking", "confidence": 0.9},
    )

    result = classify(new_state("furious about the wait, anyway book me in"))

    assert result["intent"] == "complaint"
    assert result["secondary_intent"] == "booking"


@pytest.mark.parametrize(
    "raw,expected", [(1.5, 1.0), (-0.2, 0.0), ("high", 0.0), (None, 0.0), (0.75, 0.75)]
)
def test_confidence_is_clamped_and_coerced(
    raw: Any, expected: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_router(monkeypatch, {"intent": "inquiry", "confidence": raw})

    assert classify(new_state("q"))["confidence"] == expected


def test_classification_failure_defaults_to_inquiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inquiry is the safe default: it is the only agent that can refuse.

    Defaulting to booking would interrogate a confused user; defaulting
    to complaint would manufacture a record that never happened.
    """

    def _fail(*_a, **_k):  # noqa: ANN002, ANN003
        raise LLMError("router model unreachable")

    monkeypatch.setattr("app.graph.router.complete_json", _fail)

    result = classify(new_state("how much is a crown"))

    assert result["intent"] == "inquiry"
    assert result["routed_by"] == "fallback"
    assert result["confidence"] == 0.0


def test_select_agent_reads_the_intent() -> None:
    assert select_agent({"intent": "complaint"}) == "complaint"
    assert select_agent({}) == "inquiry", "missing intent must not crash dispatch"


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_agents(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Replace every agent with a recorder, so dispatch can be observed."""
    calls: dict[str, list[str]] = {"booking": [], "inquiry": [], "complaint": []}

    def recorder(name: str):  # noqa: ANN202
        def _run(message: str, **_kwargs):  # noqa: ANN003, ANN202
            calls[name].append(message)
            from app.agents.base import AgentResponse

            return AgentResponse(answer=f"{name} handled it", agent=name)

        return _run

    monkeypatch.setattr(build_module, "handle_booking", recorder("booking"))
    monkeypatch.setattr(build_module, "answer_inquiry", recorder("inquiry"))
    monkeypatch.setattr(build_module, "handle_complaint", recorder("complaint"))
    return calls


@pytest.mark.parametrize(
    "intent,expected",
    [("booking", "booking"), ("inquiry", "inquiry"), ("complaint", "complaint")],
)
def test_each_intent_reaches_its_agent(
    intent: str, expected: str, stub_agents: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_router(monkeypatch, {"intent": intent, "confidence": 0.9, "reason": "stub"})
    build_module._compiled = None

    turn = run("some message")

    assert stub_agents[expected] == ["some message"]
    assert turn["intent"] == intent
    assert turn["answer"] == f"{expected} handled it"
    # Exactly one agent ran.
    assert sum(len(v) for v in stub_agents.values()) == 1


def test_other_intent_reaches_no_agent(
    stub_agents: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    build_module._compiled = None

    turn = run("hi")

    assert sum(len(v) for v in stub_agents.values()) == 0
    assert turn["intent"] == "other"


def test_every_path_produces_a_turn_record(
    stub_agents: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """One exit node means analytics logs in one place, not five."""
    stub_router(monkeypatch, {"intent": "booking", "confidence": 0.8, "reason": "r"})
    build_module._compiled = None

    turn = run("book me in", session_id="abc")

    for key in (
        "session_id", "message", "intent", "secondary_intent", "confidence",
        "routed_by", "routing_reason", "answer", "success", "needs_followup",
        "sources", "agent_metadata",
    ):
        assert key in turn, f"turn record is missing {key}"
    assert turn["session_id"] == "abc"


# ---------------------------------------------------------------------------
# finalise: secondary-intent handling
# ---------------------------------------------------------------------------


def test_secondary_intent_is_acknowledged() -> None:
    result = finalise(
        {"answer": "Logged as CMP-1.", "secondary_intent": "booking", "success": True}
    )

    assert result["answer"].startswith("Logged as CMP-1.")
    assert "appointment" in result["answer"].lower()


def test_secondary_note_does_not_claim_work_it_did_not_do() -> None:
    """The note must offer, not assert.

    Saying "I've also logged your complaint" when nothing was logged is a
    lie the patient discovers when nobody calls back.
    """
    result = finalise(
        {"answer": "Booked.", "secondary_intent": "complaint", "success": True}
    )

    lowered = result["answer"].lower()
    assert "haven't logged" in lowered or "have not logged" in lowered
    assert "i've logged your complaint" not in lowered


def test_no_secondary_note_when_the_agent_failed() -> None:
    """Do not bolt a cheerful offer onto an apology for being broken."""
    result = finalise(
        {"answer": "I can't reach my language service.",
         "secondary_intent": "booking", "success": False}
    )

    assert result["answer"] == "I can't reach my language service."


def test_no_note_when_there_is_no_secondary_intent() -> None:
    result = finalise({"answer": "Here you go.", "secondary_intent": None, "success": True})

    assert result["answer"] == "Here you go."


# ---------------------------------------------------------------------------
# other_node
# ---------------------------------------------------------------------------


def test_greeting_and_out_of_scope_get_different_answers() -> None:
    """`other` covers two cases that deserve different replies."""
    greeting = other_node({"routed_by": "keyword", "routing_reason": "greeting with no request"})
    farewell = other_node({"routed_by": "keyword", "routing_reason": "thanks with no request"})
    out_of_scope = other_node({"routed_by": "llm", "routing_reason": "asks about a cinema"})

    assert "I can answer questions" in greeting["answer"]
    assert "You're welcome" in farewell["answer"]
    assert "outside what I can help with" in out_of_scope["answer"]
    assert out_of_scope["agent_metadata"]["kind"] == "out_of_scope"


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_graph_has_the_expected_shape() -> None:
    """The diagram in the README is generated from this structure."""
    graph = build_graph().get_graph()
    nodes = set(graph.nodes)

    for expected in ("router", "booking", "inquiry", "complaint", "other", "finalise"):
        assert expected in nodes

    mermaid = build_graph().get_graph().draw_mermaid()
    assert "router" in mermaid
    assert "finalise" in mermaid
