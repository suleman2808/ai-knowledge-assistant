"""Tests for analytics logging and the summary.

Each test runs against its own temporary SQLite file (see conftest.py).
"""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# Retrieval trace, bookings and returning customers
# ---------------------------------------------------------------------------


def booked_turn(**overrides) -> dict:  # noqa: ANN003
    """A turn representing a completed booking."""
    base = turn(
        intent="booking",
        message="book a cleaning next Tuesday at 2pm",
        answer="You're booked in, Sarah Chen — cleaning on Tuesday at 2:00 PM.",
        agent_metadata={
            "stage": "booked",
            "event_id": "evt-123",
            "start": "2026-10-06T14:00:00",
            "end": "2026-10-06T14:40:00",
            "calendar_backend": "google",
            "extracted": {
                "service": "cleaning",
                "patient_name": "Sarah Chen",
                "phone": "503-555-0180",
                "notes": "anxious patient",
            },
        },
    )
    base.update(overrides)
    return base


def test_answer_and_sources_are_stored() -> None:
    """The dashboard's retrieval trace depends on these being kept."""
    log_turn(
        turn(
            answer="A root canal on a molar costs $1,250.",
            sources=[{"breadcrumb": "Services > Restorative", "source": "s.md",
                      "score": 0.62}],
            agent_metadata={"grounded": True, "search_query": "price of a root canal"},
        )
    )

    with connect() as c:
        row = c.execute("SELECT answer, sources, search_query FROM turns").fetchone()

    assert row["answer"].startswith("A root canal")
    assert json.loads(row["sources"])[0]["breadcrumb"] == "Services > Restorative"
    assert row["search_query"] == "price of a root canal"


def test_a_completed_booking_is_recorded() -> None:
    log_turn(booked_turn())

    with connect() as c:
        row = c.execute("SELECT * FROM bookings").fetchone()

    assert row["patient_name"] == "Sarah Chen"
    assert row["phone"] == "503-555-0180"
    assert row["service"] == "cleaning"
    assert row["event_id"] == "evt-123"
    assert row["starts_at"] == "2026-10-06T14:00:00"
    assert row["calendar_backend"] == "google"


def test_a_booking_keeps_contact_details_that_turns_redact() -> None:
    """The two tables hold deliberately different things.

    Analytics needs what was asked, not who asked. A booking needs the
    opposite, or nobody can be told their appointment moved.
    """
    log_turn(booked_turn())

    with connect() as c:
        stored_message = c.execute("SELECT message FROM turns").fetchone()[0]
        stored_phone = c.execute("SELECT phone FROM bookings").fetchone()[0]

    assert "[phone]" not in stored_phone
    assert stored_phone == "503-555-0180"
    assert "503-555-0180" not in stored_message


def test_only_booked_turns_create_a_booking() -> None:
    log_turn(turn(intent="booking", agent_metadata={"stage": "collecting_details"}))
    log_turn(turn(intent="inquiry"))

    with connect() as c:
        assert c.execute("SELECT COUNT(*) FROM bookings").fetchone()[0] == 0


def test_a_returning_customer_is_recognised_by_phone() -> None:
    log_turn(booked_turn())
    log_turn(booked_turn())

    with connect() as c:
        row = c.execute("SELECT * FROM patients").fetchone()
        count = c.execute("SELECT COUNT(*) FROM patients").fetchone()[0]

    assert count == 1, "the same number must not create a second customer"
    assert row["visit_count"] == 2
    assert row["name"] == "Sarah Chen"


def test_a_later_booking_without_a_name_does_not_erase_the_name() -> None:
    log_turn(booked_turn())

    anonymous = booked_turn()
    anonymous["agent_metadata"]["extracted"]["patient_name"] = ""
    log_turn(anonymous)

    with connect() as c:
        assert c.execute("SELECT name FROM patients").fetchone()[0] == "Sarah Chen"


def test_first_seen_is_preserved_across_visits() -> None:
    log_turn(booked_turn())
    with connect() as c:
        first = c.execute("SELECT first_seen FROM patients").fetchone()[0]

    log_turn(booked_turn())
    with connect() as c:
        row = c.execute("SELECT first_seen, last_seen FROM patients").fetchone()

    assert row["first_seen"] == first


def test_a_booking_without_a_phone_creates_no_customer() -> None:
    anonymous = booked_turn()
    anonymous["agent_metadata"]["extracted"]["phone"] = ""
    log_turn(anonymous)

    with connect() as c:
        assert c.execute("SELECT COUNT(*) FROM bookings").fetchone()[0] == 1
        assert c.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 0


def test_an_older_database_gains_the_new_columns() -> None:
    """A database created before these columns existed must keep working.

    CREATE TABLE IF NOT EXISTS silently does nothing to an existing table,
    so without a migration the insert would fail on every turn.
    """
    legacy = db_path()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(legacy) as raw:
        raw.execute(
            """
            CREATE TABLE turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL, session_id TEXT, message TEXT NOT NULL,
                intent TEXT NOT NULL, secondary_intent TEXT, confidence REAL,
                routed_by TEXT, success INTEGER, grounded INTEGER,
                retrieval_status TEXT, best_score REAL, needs_followup INTEGER,
                escalated INTEGER, booked INTEGER, latency_ms INTEGER,
                error_kind TEXT
            )
            """
        )

    assert log_turn(turn(answer="still works")) is True

    with connect() as c:
        assert c.execute("SELECT answer FROM turns").fetchone()[0] == "still works"
