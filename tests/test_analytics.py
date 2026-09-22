"""Tests for analytics logging and the summary.

Each test runs against its own temporary SQLite file (see conftest.py).
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from app.agents.complaint import ComplaintRecord, _persist, handle_complaint
from app.integrations.analytics import (
    SQLiteComplaintStore,
    connect,
    db_path,
    log_turn,
    redact,
    summary,
)


def turn(**overrides) -> dict:  # noqa: ANN003
    base = {
        "session_id": "s1",
        "message": "how much is a filling",
        "intent": "inquiry",
        "secondary_intent": None,
        "confidence": 0.95,
        "routed_by": "llm",
        "success": True,
        "needs_followup": False,
        "agent_metadata": {"grounded": True, "retrieval_status": "ok", "best_score": 0.6},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,absent",
    [
        ("Sarah Chen, 503-555-0180", "555-0180"),
        ("call me on (503) 555 0180 please", "555 0180"),
        ("my number is +1 503 555 0180", "0180"),
        ("email sarah.chen@example.com", "sarah.chen@example.com"),
    ],
)
def test_contact_details_are_redacted(text: str, absent: str) -> None:
    assert absent not in redact(text)


@pytest.mark.parametrize(
    "text",
    [
        "is a $1,250 root canal at 2pm on the 23rd covered",
        "can I come on 2026-09-29",
        "is it 10.30 or 11.00",
        # Found by probing: an early version turned this into
        # CMP-[phone]-4F2A, destroying the reference staff trace by.
        "my reference is CMP-20260920-4F2A",
    ],
)
def test_redaction_leaves_non_contact_numbers_alone(text: str) -> None:
    """Prices, dates, times and references carry the meaning."""
    assert redact(text) == text


def test_redaction_is_applied_before_writing() -> None:
    """The guarantee is about what reaches disk, so check the disk."""
    log_turn(turn(message="book me in, Sarah Chen 503-555-0180"))

    with sqlite3.connect(db_path()) as raw:
        stored = raw.execute("SELECT message FROM turns").fetchone()[0]

    assert "555-0180" not in stored
    assert "[phone]" in stored


# ---------------------------------------------------------------------------
# Writing turns
# ---------------------------------------------------------------------------


def test_turn_is_recorded() -> None:
    assert log_turn(turn(), latency_ms=820) is True

    with connect() as c:
        row = c.execute("SELECT * FROM turns").fetchone()

    assert row["intent"] == "inquiry"
    assert row["grounded"] == 1
    assert row["latency_ms"] == 820


def test_logging_failure_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An analytics outage must not break a conversation."""

    def _broken(*_a, **_k):  # noqa: ANN002, ANN003
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr("app.integrations.analytics.connect", _broken)

    assert log_turn(turn()) is False


def test_booking_and_escalation_are_derived_from_metadata() -> None:
    log_turn(turn(intent="booking", agent_metadata={"stage": "booked"}))
    log_turn(turn(intent="complaint", agent_metadata={"escalated": True}))

    with connect() as c:
        rows = {r["intent"]: r for r in c.execute("SELECT * FROM turns")}

    assert rows["booking"]["booked"] == 1
    assert rows["complaint"]["escalated"] == 1


def test_non_inquiry_turns_record_grounded_as_null() -> None:
    """Grounding only means something for inquiries; 0 would be a lie."""
    log_turn(turn(intent="booking", agent_metadata={}))

    with connect() as c:
        assert c.execute("SELECT grounded FROM turns").fetchone()[0] is None


# ---------------------------------------------------------------------------
# Complaints
# ---------------------------------------------------------------------------


def make_complaint(ref: str = "CMP-1", **overrides) -> ComplaintRecord:  # noqa: ANN003
    values = dict(
        reference=ref,
        received_at=datetime.now(),
        message="charged twice, call me on 503-555-0180",
        summary="Patient reports double billing.",
        category="billing",
        severity="high",
        escalated=True,
        escalation_reasons=["high-severity billing complaint"],
    )
    values.update(overrides)
    return ComplaintRecord(**values)


def test_complaints_persist_across_store_instances() -> None:
    """The point of moving off memory: a restart must not lose them."""
    SQLiteComplaintStore().record(make_complaint())

    reloaded = SQLiteComplaintStore().all()

    assert len(reloaded) == 1
    assert reloaded[0]["reference"] == "CMP-1"
    assert reloaded[0]["escalated"] is True
    assert reloaded[0]["escalation_reasons"] == ["high-severity billing complaint"]
    assert "555-0180" not in reloaded[0]["message"]


def test_complaint_survives_a_store_failure() -> None:
    """Losing a complaint is the one unacceptable failure."""
    from app.agents import complaint as complaint_module

    class BrokenStore:
        def record(self, _complaint) -> None:  # noqa: ANN001
            raise sqlite3.OperationalError("disk full")

    before = len(complaint_module._fallback.all())

    stored = _persist(BrokenStore(), make_complaint("CMP-FALLBACK"))

    assert stored is False
    assert len(complaint_module._fallback.all()) == before + 1


def test_agent_reports_when_persistence_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenStore:
        def record(self, _complaint) -> None:  # noqa: ANN001
            raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(
        "app.agents.complaint.complete_json",
        lambda *_a, **_k: {
            "summary": "s", "category": "billing", "severity": "medium",
            "requires_escalation": False, "patient_appears_distressed": False,
            "mentions_legal_action": False, "mentions_harm": False,
            "is_actually_a_complaint": True,
        },
    )
    monkeypatch.setattr("app.agents.complaint.complete", lambda *_a, **_k: "Sorry.")

    response = handle_complaint("charged twice", store=BrokenStore())

    assert response.metadata["persisted"] is False
    assert response.success is True, "the patient still gets a reference and a reply"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_of_an_empty_database() -> None:
    data = summary()

    assert data["turns"] == 0
    assert data["inquiries"]["answer_rate_pct"] is None, "no data is not 0%"
    assert data["latency_ms"]["p50"] is None


def test_summary_counts_intents_and_outcomes() -> None:
    log_turn(turn(), latency_ms=500)
    log_turn(turn(), latency_ms=700)
    log_turn(turn(intent="booking", agent_metadata={"stage": "booked"}), latency_ms=900)
    log_turn(turn(intent="other", routed_by="keyword", agent_metadata={}), latency_ms=5)

    data = summary()

    assert data["turns"] == 4
    assert data["intents"]["inquiry"] == 2
    assert data["bookings_made"] == 1
    assert data["fast_path_pct"] == 25.0
    assert data["latency_ms"]["p50"] in (500, 700)


def test_knowledge_gaps_list_declined_questions() -> None:
    """The most useful output: what patients asked that nobody wrote down."""
    for _ in range(3):
        log_turn(turn(message="do you offer botox",
                      agent_metadata={"grounded": False, "best_score": 0.31}))
    log_turn(turn(message="is there wifi",
                  agent_metadata={"grounded": False, "best_score": 0.2}))
    log_turn(turn(message="how much is a filling"))

    data = summary()
    gaps = data["knowledge_gaps"]

    assert gaps[0] == {"question": "do you offer botox", "times": 3, "best_score": 0.31}
    assert "how much is a filling" not in [g["question"] for g in gaps]
    assert data["inquiries"]["answer_rate_pct"] == 20.0


def test_failed_turns_are_not_reported_as_knowledge_gaps() -> None:
    """An outage is not a gap in the documents."""
    log_turn(turn(message="how much is a crown", success=False,
                  agent_metadata={"grounded": False, "error_kind": "llm_unavailable"}))

    data = summary()

    assert data["knowledge_gaps"] == []
    assert data["failure_rate_pct"] == 100.0


def test_complaint_breakdown_appears_in_summary() -> None:
    store = SQLiteComplaintStore()
    store.record(make_complaint("CMP-1", severity="high", category="billing"))
    store.record(make_complaint("CMP-2", severity="medium", category="waiting_time"))

    data = summary()

    assert data["complaints_by_severity"] == {"high": 1, "medium": 1}
    assert data["complaints_by_category"]["billing"] == 1


def test_graph_run_logs_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a turn through the graph lands in the database."""
    from app.graph import build as build_module

    build_module._compiled = None
    build_module.run("hi", session_id="e2e")

    with connect() as c:
        row = c.execute("SELECT session_id, intent, routed_by FROM turns").fetchone()

    assert tuple(row) == ("e2e", "other", "keyword")


def test_graph_run_can_skip_logging() -> None:
    from app.graph import build as build_module

    build_module._compiled = None
    build_module.run("hi", log=False)

    assert summary()["turns"] == 0
