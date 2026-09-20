"""Tests for the three specialist agents.

Every LLM call is stubbed. These tests are about the logic *around* the
model — grounding enforcement, escalation rules, availability checking,
failure handling — which is the part we wrote and the part that must not
regress. Whether the model writes a nice sentence is not a unit test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.agents import complaint as complaint_module
from app.agents.base import AgentResponse, normalise
from app.agents.booking import handle_booking
from app.agents.complaint import ComplaintStore, handle_complaint
from app.agents.inquiry import NO_ANSWER, answer_inquiry
from app.integrations.calendar import (
    Appointment,
    CalendarError,
    InMemoryCalendar,
    TimeSlot,
    duration_for,
    is_open,
)
from app.llm import LLMError
from app.rag.retriever import (
    RetrievalResult,
    RetrievalStatus,
    RetrievedChunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def grounded_result(text: str = "A root canal on a molar costs $1,250.") -> RetrievalResult:
    return RetrievalResult(
        query="q",
        status=RetrievalStatus.OK,
        chunks=[
            RetrievedChunk(
                chunk_id="services::2",
                text=text,
                score=0.62,
                breadcrumb="Services and Pricing > Restorative Treatment",
                source="services-and-pricing.md",
            )
        ],
        best_score=0.62,
    )


def next_weekday(target: int, *, weeks_ahead: int = 1) -> date:
    """A future date falling on `target` weekday (Monday = 0)."""
    today = date.today()
    ahead = (target - today.weekday()) % 7 or 7
    return today + timedelta(days=ahead + 7 * (weeks_ahead - 1))


# ---------------------------------------------------------------------------
# Inquiry Agent
# ---------------------------------------------------------------------------


def test_inquiry_answers_from_retrieved_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agents.inquiry.retrieve", lambda _q: grounded_result())
    monkeypatch.setattr(
        "app.agents.inquiry.complete_verbose",
        lambda *_a, **_k: type(
            "R", (), {"text": "$1,250.", "latency_ms": 500, "prompt_tokens": 10,
                      "completion_tokens": 5, "model": "test"}
        )(),
    )

    response = answer_inquiry("how much is a root canal")

    assert response.answer == "$1,250."
    assert response.metadata["grounded"] is True
    assert response.sources[0]["breadcrumb"].startswith("Services and Pricing")


def test_inquiry_refuses_when_the_model_says_context_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second line of defence.

    Retrieval scored above the threshold, so stage one let it through.
    The model judged the context not to answer the question. That must
    become a refusal, not a raw token shown to a patient.
    """
    monkeypatch.setattr("app.agents.inquiry.retrieve", lambda _q: grounded_result())
    monkeypatch.setattr(
        "app.agents.inquiry.complete_verbose",
        lambda *_a, **_k: type(
            "R", (), {"text": "INSUFFICIENT_CONTEXT", "latency_ms": 90,
                      "prompt_tokens": 10, "completion_tokens": 3, "model": "test"}
        )(),
    )

    response = answer_inquiry("what time does the cinema open")

    assert response.answer == NO_ANSWER
    assert "INSUFFICIENT" not in response.answer
    assert response.metadata["grounded"] is False
    assert response.metadata["refused_by"] == "model"
    assert response.sources == []


@pytest.mark.parametrize(
    "raw",
    ["INSUFFICIENT_CONTEXT", "INSUFFICIENT_CONTEXT.", '"INSUFFICIENT_CONTEXT"',
     "  insufficient_context  ", "**INSUFFICIENT_CONTEXT**"],
)
def test_inquiry_refusal_detection_tolerates_formatting(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Models decorate the token despite instructions; all forms must count."""
    monkeypatch.setattr("app.agents.inquiry.retrieve", lambda _q: grounded_result())
    monkeypatch.setattr(
        "app.agents.inquiry.complete_verbose",
        lambda *_a, **_k: type(
            "R", (), {"text": raw, "latency_ms": 1, "prompt_tokens": 1,
                      "completion_tokens": 1, "model": "t"}
        )(),
    )

    assert answer_inquiry("q").answer == NO_ANSWER


def test_inquiry_does_not_call_the_model_when_retrieval_is_weak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No grounding means no call — cheaper, and structurally safer."""
    called = False

    def _never(*_a, **_k):  # noqa: ANN002, ANN003
        nonlocal called
        called = True
        raise AssertionError("the model must not be called without context")

    monkeypatch.setattr(
        "app.agents.inquiry.retrieve",
        lambda _q: RetrievalResult(
            query="q", status=RetrievalStatus.LOW_CONFIDENCE, best_score=0.12
        ),
    )
    monkeypatch.setattr("app.agents.inquiry.complete_verbose", _never)

    response = answer_inquiry("what is the capital of France")

    assert called is False
    assert response.answer == NO_ANSWER
    assert response.success is True, "not knowing is correct behaviour, not a failure"


def test_inquiry_reports_an_unbuilt_index_as_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator error must be distinguishable from ignorance."""
    monkeypatch.setattr(
        "app.agents.inquiry.retrieve",
        lambda _q: RetrievalResult(query="q", status=RetrievalStatus.EMPTY_INDEX),
    )

    response = answer_inquiry("anything")

    assert response.success is False
    assert response.metadata["retrieval_status"] == "empty_index"


def test_inquiry_degrades_when_the_model_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.agents.inquiry.retrieve", lambda _q: grounded_result())

    def _fail(*_a, **_k):  # noqa: ANN002, ANN003
        raise LLMError("groq is unreachable")

    monkeypatch.setattr("app.agents.inquiry.complete_verbose", _fail)

    response = answer_inquiry("how much is a root canal")

    assert response.success is False
    assert "555-0142" in response.answer
    assert "groq is unreachable" not in response.answer
    assert response.metadata["error_detail"] == "groq is unreachable"


# ---------------------------------------------------------------------------
# Booking Agent
# ---------------------------------------------------------------------------


def stub_extraction(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    monkeypatch.setattr("app.agents.booking.complete_json", lambda *_a, **_k: payload)


def test_booking_asks_only_for_what_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_extraction(
        monkeypatch,
        {"service": "check-up", "date": next_weekday(0).isoformat(), "time": None,
         "time_preference": "morning", "patient_name": None, "phone": None},
    )

    response = handle_booking("check-up monday morning", backend=InMemoryCalendar())

    assert response.needs_followup is True
    assert set(response.metadata["missing"]) == {"patient_name", "phone"}
    assert "full name" in response.answer
    assert "come in for" not in response.answer, "should not re-ask for known details"


def test_booking_does_not_show_iso_dates_to_patients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = next_weekday(0)
    stub_extraction(
        monkeypatch,
        {"service": "check-up", "date": target.isoformat(), "time": None,
         "time_preference": "morning", "patient_name": None, "phone": None},
    )

    response = handle_booking("check-up", backend=InMemoryCalendar())

    assert target.isoformat() not in response.answer


def test_booking_creates_an_appointment_when_details_are_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = next_weekday(1)  # a Tuesday
    stub_extraction(
        monkeypatch,
        {"service": "cleaning", "date": target.isoformat(), "time": "14:00",
         "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "503-555-0180", "notes": None},
    )
    calendar = InMemoryCalendar(appointments=[])

    response = handle_booking("book a cleaning", backend=calendar)

    assert response.success is True
    assert response.needs_followup is False
    assert response.metadata["stage"] == "booked"
    assert len(calendar.appointments) == 1
    booked = calendar.appointments[0]
    assert booked.patient_name == "Sarah Chen"
    assert booked.start.hour == 14
    # Cleaning is a 40-minute appointment.
    assert booked.end - booked.start == timedelta(minutes=40)


def test_booking_refuses_a_time_outside_opening_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = next_weekday(6)  # Sunday, closed
    stub_extraction(
        monkeypatch,
        {"service": "check-up", "date": target.isoformat(), "time": "10:00",
         "time_preference": None, "patient_name": "Ana Silva",
         "phone": "503-555-0111", "notes": None},
    )
    calendar = InMemoryCalendar(appointments=[])

    response = handle_booking("sunday please", backend=calendar)

    assert calendar.appointments == [], "must not book when the clinic is closed"
    assert response.needs_followup is True
    assert "closed on Sunday" in response.answer


def test_booking_offers_alternatives_when_the_slot_is_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = next_weekday(1)
    clash_start = datetime.combine(target, datetime.min.time()).replace(hour=14)
    calendar = InMemoryCalendar(
        appointments=[
            Appointment(
                event_id="existing",
                start=clash_start,
                end=clash_start + timedelta(minutes=60),
                summary="Crown fitting",
            )
        ]
    )
    stub_extraction(
        monkeypatch,
        {"service": "cleaning", "date": target.isoformat(), "time": "14:00",
         "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "503-555-0180", "notes": None},
    )

    response = handle_booking("2pm please", backend=calendar)

    assert response.metadata["stage"] == "offered_alternatives"
    assert len(response.metadata["offered"]) > 0
    assert len(calendar.appointments) == 1, "no booking should have been made"


def test_booking_rejects_a_date_in_the_past(monkeypatch: pytest.MonkeyPatch) -> None:
    """A past date means the model mis-resolved a relative reference."""
    stub_extraction(
        monkeypatch,
        {"service": "cleaning", "date": "2020-01-01", "time": "14:00",
         "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "503-555-0180", "notes": None},
    )
    calendar = InMemoryCalendar(appointments=[])

    response = handle_booking("last january", backend=calendar)

    assert calendar.appointments == []
    assert "date" in response.metadata.get("missing", [])


def test_booking_rejects_an_implausible_phone_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_extraction(
        monkeypatch,
        {"service": "cleaning", "date": next_weekday(1).isoformat(), "time": "14:00",
         "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "call me maybe", "notes": None},
    )

    response = handle_booking("book it", backend=InMemoryCalendar(appointments=[]))

    assert "phone" in response.metadata["missing"]


def test_booking_survives_a_calendar_outage(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenCalendar:
        name = "broken"

        def busy_periods(self, day):  # noqa: ANN001, ARG002
            raise CalendarError("calendar API returned 503")

        def create_appointment(self, *a, **k):  # noqa: ANN002, ANN003, ARG002
            raise CalendarError("calendar API returned 503")

    stub_extraction(
        monkeypatch,
        {"service": "cleaning", "date": next_weekday(1).isoformat(), "time": "14:00",
         "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "503-555-0180", "notes": None},
    )

    response = handle_booking("book it", backend=BrokenCalendar())

    assert response.success is False
    assert "503" not in response.answer or "555-0142" in response.answer
    assert "CalendarError" not in response.answer


def test_booking_degrades_when_extraction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_a, **_k):  # noqa: ANN002, ANN003
        raise LLMError("rate limited")

    monkeypatch.setattr("app.agents.booking.complete_json", _fail)

    response = handle_booking("book me in", backend=InMemoryCalendar())

    assert response.success is False
    assert "rate limited" not in response.answer


def test_service_durations_prefer_the_longest_match() -> None:
    assert duration_for("deep cleaning please") == 60
    assert duration_for("just a cleaning") == 40
    assert duration_for("something unheard of") == 30


def test_last_appointment_is_an_hour_before_closing() -> None:
    friday = next_weekday(4)
    at_two = datetime.combine(friday, datetime.min.time()).replace(hour=14)
    at_half_two = at_two + timedelta(minutes=30)

    assert is_open(at_two) is True, "Friday closes at 3pm, so 2pm is bookable"
    assert is_open(at_half_two) is False, "2:30pm leaves under an hour"


# ---------------------------------------------------------------------------
# Complaint Agent
# ---------------------------------------------------------------------------


def stub_complaint(monkeypatch: pytest.MonkeyPatch, assessment: dict) -> None:
    monkeypatch.setattr(
        "app.agents.complaint.complete_json", lambda *_a, **_k: assessment
    )
    monkeypatch.setattr(
        "app.agents.complaint.complete", lambda *_a, **_k: "We're sorry. Ref given."
    )


BASE_ASSESSMENT = {
    "summary": "Patient waited 45 minutes.",
    "category": "waiting_time",
    "severity": "medium",
    "requires_escalation": False,
    "patient_appears_distressed": False,
    "mentions_legal_action": False,
    "mentions_harm": False,
    "is_actually_a_complaint": True,
}


def test_complaint_is_logged_with_a_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_complaint(monkeypatch, dict(BASE_ASSESSMENT))
    store = ComplaintStore()

    response = handle_complaint("I waited 45 minutes", store=store)

    assert len(store.all()) == 1
    assert response.metadata["reference"].startswith("CMP-")
    assert store.all()[0].severity == "medium"


def test_every_high_severity_complaint_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caught in testing: a double-billing complaint with three ignored
    calls was rated high and not escalated, because the model said so."""
    stub_complaint(
        monkeypatch,
        {**BASE_ASSESSMENT, "category": "billing", "severity": "high",
         "requires_escalation": False},
    )
    store = ComplaintStore()

    response = handle_complaint("charged twice, three calls ignored", store=store)

    assert response.metadata["escalated"] is True
    assert store.all()[0].escalated is True


def test_reported_harm_escalates_regardless_of_severity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Code overrides the model upward, never downward."""
    stub_complaint(
        monkeypatch,
        {**BASE_ASSESSMENT, "severity": "low", "mentions_harm": True,
         "requires_escalation": False},
    )

    response = handle_complaint("my gums bled for two days", store=ComplaintStore())

    assert response.metadata["escalated"] is True
    assert "patient reports harm" in response.metadata["escalation_reasons"]


def test_legal_threat_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_complaint(
        monkeypatch,
        {**BASE_ASSESSMENT, "severity": "low", "mentions_legal_action": True,
         "requires_escalation": False},
    )

    response = handle_complaint("I'm contacting my solicitor", store=ComplaintStore())

    assert response.metadata["escalated"] is True


def test_unrecognised_severity_defaults_to_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid JSON does not mean sensible values."""
    stub_complaint(monkeypatch, {**BASE_ASSESSMENT, "severity": "catastrophic"})

    response = handle_complaint("something happened", store=ComplaintStore())

    assert response.metadata["severity"] == "high"
    assert response.metadata["escalated"] is True


def test_praise_is_not_logged_as_a_complaint(monkeypatch: pytest.MonkeyPatch) -> None:
    stub_complaint(
        monkeypatch, {**BASE_ASSESSMENT, "is_actually_a_complaint": False}
    )
    store = ComplaintStore()

    response = handle_complaint("your receptionist was lovely", store=store)

    assert store.all() == []
    assert response.metadata["logged"] is False


def test_complaint_is_recorded_even_when_assessment_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing a complaint is the unacceptable failure."""

    def _fail(*_a, **_k):  # noqa: ANN002, ANN003
        raise LLMError("model unavailable")

    monkeypatch.setattr("app.agents.complaint.complete_json", _fail)
    store = ComplaintStore()

    response = handle_complaint("something went badly wrong", store=store)

    assert len(store.all()) == 1, "the complaint must survive an LLM outage"
    assert store.all()[0].escalated is True
    assert store.all()[0].severity == "high"
    assert response.metadata["reference"] in response.answer


def test_complaint_is_logged_before_the_reply_is_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reply generation is courtesy; the record has regulatory weight."""
    stub_complaint(monkeypatch, dict(BASE_ASSESSMENT))

    def _fail(*_a, **_k):  # noqa: ANN002, ANN003
        raise LLMError("died after assessment")

    monkeypatch.setattr("app.agents.complaint.complete", _fail)
    store = ComplaintStore()

    response = handle_complaint("I waited 45 minutes", store=store)

    assert len(store.all()) == 1
    assert response.metadata["reference"] in response.answer


# ---------------------------------------------------------------------------
# Shared behaviour
# ---------------------------------------------------------------------------


def test_invisible_characters_are_stripped_from_answers() -> None:
    """Observed: a soft hyphen inside a phone number broke copy-paste."""
    response = AgentResponse(answer="Call (503) 555­0142​ now", agent="t")

    assert response.answer == "Call (503) 5550142 now"
    assert normalise("a‑b") == "a-b"
    assert normalise("a b") == "a b"
