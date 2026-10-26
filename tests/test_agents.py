"""Tests for the three specialist agents.

Every LLM call is stubbed. These tests are about the logic *around* the
model — grounding enforcement, escalation rules, availability checking,
failure handling — which is the part we wrote and the part that must not
regress. Whether the model writes a nice sentence is not a unit test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from app.agents import complaint as complaint_module
from app.agents.base import AgentResponse, normalise
from app.agents.booking import handle_booking
from app.agents.complaint import ComplaintStore, handle_complaint
from app.agents.inquiry import NO_ANSWER, answer_inquiry
from app.config import settings
from app.integrations.analytics import log_turn
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
        {"service": "basic health check", "date": next_weekday(0).isoformat(), "time": None,
         "time_preference": "morning", "patient_name": None, "phone": None},
    )

    response = handle_booking("health check monday morning", backend=InMemoryCalendar())

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
        {"service": "basic health check", "date": target.isoformat(), "time": None,
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
        {"service": "full blood count", "date": target.isoformat(), "time": "14:00",
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
    # A full blood count is a ten-minute collection.
    assert booked.end - booked.start == timedelta(minutes=10)


def test_booking_refuses_a_time_outside_opening_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every day is a collection day, so the out-of-hours case is a time,
    # not a day: 9pm on a weekday.
    target = next_weekday(2)
    stub_extraction(
        monkeypatch,
        {"service": "basic health check", "date": target.isoformat(), "time": "21:00",
         "time_preference": None, "patient_name": "Ana Silva",
         "phone": "503-555-0111", "notes": None},
    )
    calendar = InMemoryCalendar(appointments=[])

    response = handle_booking("9pm please", backend=calendar)

    assert calendar.appointments == [], "must not book when the laboratory is closed"
    assert response.needs_followup is True
    assert "hours are" in response.answer


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
        {"service": "full blood count", "date": target.isoformat(), "time": "14:00",
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
        {"service": "full blood count", "date": "2020-01-01", "time": "14:00",
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
        {"service": "full blood count", "date": next_weekday(1).isoformat(), "time": "14:00",
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
        {"service": "full blood count", "date": next_weekday(1).isoformat(), "time": "14:00",
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
    """"glucose tolerance" must beat nothing, and an unknown test still
    gets a slot rather than failing."""
    assert duration_for("glucose tolerance test please") == 150
    assert duration_for("just a full blood count") == 10
    assert duration_for("something unheard of") == 15


def test_collection_stops_before_the_doors_close() -> None:
    """A sample has to be taken, labelled and logged before closing."""
    saturday = next_weekday(5)
    at_one_fifteen = datetime.combine(saturday, datetime.min.time()).replace(hour=13, minute=15)
    at_one_forty_five = at_one_fifteen + timedelta(minutes=30)

    assert is_open(at_one_fifteen) is True, "Saturday closes at 2pm"
    assert is_open(at_one_forty_five) is False, "under the 30-minute buffer"


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


def test_demo_booking_does_not_promise_reminders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Found on the hosted demo: a booking confirmed with "we'll send a
    reminder", when the deployed instance uses the in-memory calendar and
    nothing in this project sends email at all."""
    target = next_weekday(1)
    stub_extraction(
        monkeypatch,
        {"service": "full blood count", "date": target.isoformat(), "date_phrase": None,
         "time": "14:00", "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "503-555-0180", "notes": None},
    )

    response = handle_booking("book it", backend=InMemoryCalendar(appointments=[]))

    answer = response.answer.lower()
    assert response.metadata["stage"] == "booked"
    assert "we'll send a reminder" not in answer, "promises an action nothing performs"
    # And it must not break the illusion either: a customer is not told
    # they are looking at a demonstration. Where the booking is held is
    # recorded in metadata for the dashboard, not announced to them.
    assert "demonstration" not in answer
    assert "in memory" not in answer
    assert response.metadata["calendar_backend"] == "in_memory"


def test_real_calendar_booking_keeps_the_clinic_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Against a connected calendar the reminder wording is accurate."""

    class RealishCalendar(InMemoryCalendar):
        name = "google"

    target = next_weekday(1)
    stub_extraction(
        monkeypatch,
        {"service": "full blood count", "date": target.isoformat(), "date_phrase": None,
         "time": "14:00", "time_preference": None, "patient_name": "Sarah Chen",
         "phone": "503-555-0180", "notes": None},
    )

    response = handle_booking("book it", backend=RealishCalendar(appointments=[]))

    assert "reminder" in response.answer.lower()


def test_the_reply_only_claims_an_email_that_was_actually_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three different endings, each true of the booking it describes."""

    class NotifyingCalendar(InMemoryCalendar):
        name = "google"

        def create_appointment(self, slot, **kwargs):  # noqa: ANN001, ANN003
            appointment = super().create_appointment(slot, **kwargs)
            appointment.notified = True
            return appointment

    class SilentCalendar(InMemoryCalendar):
        name = "google"

    target = next_weekday(1)
    payload = {
        "service": "full blood count", "date": target.isoformat(), "date_phrase": None,
        "time": "14:00", "time_preference": None, "patient_name": "Sarah Chen",
        "phone": "503-555-0180", "notes": None,
    }

    stub_extraction(monkeypatch, payload)
    notified = handle_booking("book it", backend=NotifyingCalendar(appointments=[]))
    assert "emailed to the laboratory" in notified.answer

    stub_extraction(monkeypatch, payload)
    silent = handle_booking("book it", backend=SilentCalendar(appointments=[]))
    assert "emailed" not in silent.answer
    assert "reminder" in silent.answer

    stub_extraction(monkeypatch, payload)
    demo = handle_booking("book it", backend=InMemoryCalendar(appointments=[]))
    assert "emailed" not in demo.answer
    assert "demonstration" not in demo.answer


# ---------------------------------------------------------------------------
# Cancelling an appointment
# ---------------------------------------------------------------------------


def _a_weekday_soon() -> str:
    """A bookable date a few days out, as an ISO string.

    Written rather than hard-coded: a fixed date in a booking test passes
    until the morning it goes past, and then fails for a reason that has
    nothing to do with the code under test.
    """
    day = date.today() + timedelta(days=3)
    while day.weekday() >= 5:  # the laboratory's weekend hours differ
        day += timedelta(days=1)
    return day.isoformat()


@pytest.mark.parametrize(
    "message",
    [
        "cancel it, my mind changed",
        "please cancel my appointment",
        "actually, I've changed my mind",
        "forget it",
        "I no longer need that appointment",
    ],
)
def test_a_cancellation_is_recognised(message: str) -> None:
    from app.agents.booking import _wants_to_cancel

    assert _wants_to_cancel(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "what is your cancellation policy?",
        "can I cancel an appointment later?",
        "how do I cancel if something comes up",
        "is there a late cancellation fee",
        "book me a blood test on Tuesday",
    ],
)
def test_a_question_about_cancelling_is_not_a_cancellation(message: str) -> None:
    """Asking about the policy must not call off the patient's appointment."""
    from app.agents.booking import _wants_to_cancel

    assert _wants_to_cancel(message) is False


def test_cancelling_removes_the_appointment_from_the_calendar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bug: 'cancel it' was extracted as a booking request with no
    date, so the agent asked which day they would like - and a following
    'yes' re-confirmed the appointment they were trying to call off."""
    from app.agents.booking import handle_booking

    monkeypatch.setattr(settings, "analytics_db", tmp_path / "analytics.db")
    stub_extraction(
        monkeypatch,
        {"service": "blood test", "date": _a_weekday_soon(), "time": "09:00",
         "time_preference": None, "patient_name": "Sarah Chen", "phone": "503-555-0180"},
    )

    calendar = InMemoryCalendar(appointments=[])
    booked = handle_booking(
        "book a blood test, Sarah Chen, 503-555-0180",
        backend=calendar,
        session_id="s1",
    )
    assert booked.metadata["stage"] == "booked"
    assert len(calendar.appointments) == 1

    # The graph logs every turn, and logging a booked turn is what writes
    # the row the cancellation later reads. Doing it here keeps the test
    # on the agent while still exercising the real lookup.
    log_turn(
        {
            "session_id": "s1",
            "message": "book a blood test",
            "answer": booked.answer,
            "intent": "booking",
            "agent_metadata": booked.metadata,
        },
        latency_ms=10,
    )

    cancelled = handle_booking("cancel it, my mind changed", backend=calendar, session_id="s1")

    assert cancelled.metadata["stage"] == "cancelled"
    assert calendar.appointments == []
    assert "cancelled" in cancelled.answer.lower()
    assert "booked in" not in cancelled.answer.lower()


def test_cancelling_with_nothing_booked_says_so(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents.booking import handle_booking

    monkeypatch.setattr(settings, "analytics_db", tmp_path / "analytics.db")

    response = handle_booking(
        "cancel it", backend=InMemoryCalendar(appointments=[]), session_id="empty"
    )

    assert response.metadata["stage"] == "nothing_to_cancel"
    assert response.success is True


@pytest.mark.parametrize(
    "message",
    [
        "nah leave it for now, ill book other time",
        "ill book some other time, not now",
        "never mind",
        "not today thanks",
        "maybe later",
        "leave it for now",
    ],
)
def test_backing_out_of_a_booking_is_recognised(message: str) -> None:
    """Reported: told "leave it for now", the agent re-sent the same
    request for a name and number, verbatim, three times."""
    from app.agents.booking import _is_deferral

    assert _is_deferral(message) is True


@pytest.mark.parametrize(
    "message",
    [
        # "another time" means a different slot when a day comes with it.
        "can I book another time slot on friday?",
        "book another time on tuesday at 9am",
        "book a blood test next tuesday",
        "i want to book an appointment",
    ],
)
def test_arranging_a_different_slot_is_not_backing_out(message: str) -> None:
    from app.agents.booking import _is_deferral

    assert _is_deferral(message) is False


def test_backing_out_mid_booking_does_not_re_ask_for_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "analytics_db", tmp_path / "analytics.db")

    response = handle_booking(
        "nah leave it for now, ill book other time",
        backend=InMemoryCalendar(appointments=[]),
        session_id="s-defer",
    )

    assert response.metadata["stage"] == "deferred"
    assert "full name" not in response.answer.lower()
    assert "contact number" not in response.answer.lower()
    assert response.needs_followup is False


def test_a_soft_no_after_a_confirmed_booking_keeps_the_appointment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"Leave it" with an appointment on the books means leave it alone.
    Reading it as a cancellation would lose someone their slot."""
    monkeypatch.setattr(settings, "analytics_db", tmp_path / "analytics.db")
    stub_extraction(
        monkeypatch,
        {"service": "blood test", "date": _a_weekday_soon(), "time": "09:00",
         "time_preference": None, "patient_name": "Sarah Chen", "phone": "503-555-0180"},
    )

    calendar = InMemoryCalendar(appointments=[])
    booked = handle_booking(
        "book a blood test, Sarah Chen, 503-555-0180",
        backend=calendar,
        session_id="s-keep",
    )
    log_turn(
        {
            "session_id": "s-keep",
            "message": "book a blood test",
            "answer": booked.answer,
            "intent": "booking",
            "agent_metadata": booked.metadata,
        },
        latency_ms=10,
    )

    response = handle_booking("leave it for now", backend=calendar, session_id="s-keep")

    assert response.metadata["stage"] == "deferred"
    assert len(calendar.appointments) == 1
    assert "cancel" in response.answer.lower()  # offered, not performed


# ---------------------------------------------------------------------------
# Found testing the live demo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "service",
    ["haircut", "MRI scan", "X-ray", "dental cleaning", "massage"],
)
def test_a_service_the_laboratory_does_not_do_is_refused(service: str) -> None:
    """Reported from the live demo: "You're booked in, Lee Park - haircut on
    Monday at 2:00 PM", at a medical laboratory. The model extracts whatever
    the patient says and nothing checked it against what is offered."""
    from app.agents.booking import is_a_service_we_offer

    assert is_a_service_we_offer(service) is False


@pytest.mark.parametrize(
    "service",
    [
        "blood test", "lipid profile", "full blood count", "thyroid profile",
        "health check", "urine test", "HbA1c", "glucose tolerance test",
        "vitamin D", "covid test", "pregnancy test", "liver function tests",
        # Nothing specific named yet: asking for it is the next step.
        "a test", "appointment", None,
    ],
)
def test_a_real_service_is_never_refused(service: str | None) -> None:
    """Refusing a real test loses a patient; accepting an odd one is caught
    by whoever reads the diary. The check leans permissive."""
    from app.agents.booking import is_a_service_we_offer

    assert is_a_service_we_offer(service) is True


def test_booking_a_haircut_is_declined_before_anything_else_is_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_extraction(
        monkeypatch,
        {"service": "haircut", "date": next_weekday(0).isoformat(), "time": "14:00",
         "time_preference": None, "patient_name": "Lee Park", "phone": "503-555-0155"},
    )
    calendar = InMemoryCalendar(appointments=[])

    reply = handle_booking("book a haircut", backend=calendar, session_id="h")

    assert reply.metadata["stage"] == "unknown_service"
    assert "can't book a haircut" in reply.answer
    assert calendar.appointments == []


def test_the_catalogue_is_read_from_the_document_not_copied_into_code() -> None:
    """A test added to the catalogue becomes bookable with no other change."""
    from app.agents.booking import _catalogue_vocabulary

    vocabulary = _catalogue_vocabulary()

    assert {"haemoglobin", "lipid", "glucose"} <= vocabulary or "lipid" in vocabulary
    assert "haircut" not in vocabulary


def test_a_date_that_has_passed_is_acknowledged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reported: "book a blood test yesterday at 10am" was answered with "which
    day suits you?" as though the date had never been said."""
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    stub_extraction(
        monkeypatch,
        {"service": "blood test", "date": yesterday, "time": "10:00",
         "time_preference": None, "patient_name": "Mia Clarke", "phone": "503-555-0199"},
    )

    reply = handle_booking("book a blood test yesterday at 10am",
                           backend=InMemoryCalendar(appointments=[]), session_id="p")

    assert "already passed" in reply.answer
    assert "Yesterday" in reply.answer


def test_a_long_dash_between_words_does_not_glue_them_together() -> None:
    """Reported: "lipid profile-typically for 9 to 12 hours"."""
    assert normalise("lipid profile\u2014typically for 9 to 12 hours") == (
        "lipid profile - typically for 9 to 12 hours"
    )


def test_a_dash_between_digits_stays_a_tight_range() -> None:
    assert normalise("fast for 9\u201312 hours") == "fast for 9-12 hours"


@pytest.mark.parametrize(
    "message",
    [
        "My haemoglobin is 9.2, is that bad?",
        "my glucose is 130 is that normal",
        "is a cholesterol of 6.1 dangerous?",
        "TSH 5.2 high?",
        "my result says 14 g/dL, should I worry?",
    ],
)
def test_a_request_to_interpret_a_result_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch, message: str
) -> None:
    """Reported: "9.2 g/dL is not in the critical range". The prompt forbade
    it and the model said it anyway, through two rewordings, so the rule is
    now enforced in code. The suite's guard would fail this test if a model
    call were made."""
    from app.agents.inquiry import RESULT_INTERPRETATION_ANSWER

    monkeypatch.setattr(
        "app.agents.inquiry.retrieve",
        lambda _q: RetrievalResult(query="q", status=RetrievalStatus.LOW_CONFIDENCE),
    )

    reply = answer_inquiry(message)

    assert reply.answer == RESULT_INTERPRETATION_ANSWER
    assert reply.metadata["handled"] == "result_interpretation"


def test_the_interpretation_refusal_makes_no_clinical_claim() -> None:
    from app.agents.inquiry import RESULT_INTERPRETATION_ANSWER

    lowered = RESULT_INTERPRETATION_ANSWER.lower()
    for word in ("critical", "threshold", "normal range", "reference range", "below", "above"):
        assert word not in lowered


@pytest.mark.parametrize(
    "message",
    [
        "How much is a glucose test?",
        "do I need to fast for 12 hours?",
        "can I come at 8 ok?",
        "is my haemoglobin result ready?",
        "what is a normal haemoglobin test",
    ],
)
def test_ordinary_questions_are_not_mistaken_for_a_request_to_interpret(message: str) -> None:
    from app.agents.inquiry import asks_to_interpret_a_result

    assert asks_to_interpret_a_result(message) is False


def test_each_half_of_a_compound_question_is_anchored_to_the_business() -> None:
    """"Where are your branches" scored 0.28 and was rejected; with the
    company's name in it, 0.62 and the right section."""
    from app.agents.inquiry import _anchor

    assert _anchor("where are your branches") == "where are your branches Riverbend Diagnostics"
    assert _anchor("hours at Riverbend Diagnostics") == "hours at Riverbend Diagnostics"


# ---------------------------------------------------------------------------
# The test catalogue, looked up by name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "test_name"),
    [
        # Reported: this was declined, because retrieval never returned the
        # table holding the row, so the model had nothing to answer from.
        ("how much is a lipid profile?", "Lipid profile"),
        ("how much does a full blood count cost", "Full blood count"),
        ("price of HbA1c", "HbA1c"),
        ("what is the LFT price", "Liver function tests"),
        ("how much is an ESR", "Erythrocyte sedimentation rate"),
        ("do I need to fast for a glucose tolerance test", "Glucose tolerance test"),
    ],
)
def test_a_test_named_in_the_question_is_found_in_the_catalogue(
    question: str, test_name: str
) -> None:
    from app.rag.catalogue import rows_named_in

    assert [r.name for r in rows_named_in(question)] == [test_name]


@pytest.mark.parametrize(
    "question",
    ["how much is a blood test", "do you open on sunday", "how much is a haircut"],
)
def test_a_question_that_names_no_test_matches_no_row(question: str) -> None:
    from app.rag.catalogue import rows_named_in

    assert rows_named_in(question) == []


def test_the_catalogue_row_reaches_the_model_even_when_retrieval_misses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure in one test: retrieval returns nothing useful, the named
    test's row is supplied anyway, and the model answers from it."""
    seen: dict[str, str] = {}

    def _answer(prompt: str, **_kw):  # noqa: ANN003, ANN202
        seen["prompt"] = prompt
        from app.llm import LLMResponse
        return LLMResponse(text="A lipid profile costs $62.", model="m", latency_ms=1,
                           prompt_tokens=0, completion_tokens=0, raw=None)

    monkeypatch.setattr("app.agents.inquiry.complete_verbose", _answer)
    monkeypatch.setattr(
        "app.agents.inquiry.retrieve",
        lambda _q: RetrievalResult(query="q", status=RetrievalStatus.LOW_CONFIDENCE),
    )

    reply = answer_inquiry("how much is a lipid profile?")

    assert "| Lipid profile | $62 |" in seen["prompt"]
    assert reply.answer == "A lipid profile costs $62."
    assert reply.metadata["grounded"] is True


def test_an_answer_repeated_by_two_parts_is_said_once() -> None:
    from app.agents.inquiry import _without_repeats

    same = "Yes - a lipid profile needs a 9-12 hour fast. It costs $62."
    assert _without_repeats([same, same.replace("Yes - ", "Yes, ")]) == [same]
    assert _without_repeats(["Fasting is needed.", "It costs $62."]) == [
        "Fasting is needed.", "It costs $62.",
    ]


def test_a_pronoun_only_part_is_given_what_it_refers_to() -> None:
    """"How much does it cost" means nothing without the other half."""
    from app.agents.inquiry import _part_question

    assert "lipid profile" in _part_question("how much does it cost", "do I need to fast for a lipid profile")
    # A part with its own subject is left exactly as asked: appending the
    # rest of the message made the model refuse the half it could answer.
    assert _part_question("what are your hours", "do you do allergy testing") == "what are your hours"


def test_folding_a_dash_does_not_merge_the_lines_of_a_markdown_list() -> None:
    """The first version collapsed every double space in an answer, and
    markdown reads two trailing spaces as a line break - so a list written
    that way would have run together into one paragraph."""
    text = "Monday to Friday: 6:30 am - 6:00 pm  \nSaturday: 7:00 am - 2:00 pm  \nSunday: 8:00 am"

    assert normalise(text) == text


def test_a_dash_next_to_a_space_does_not_double_it() -> None:
    assert normalise("word \u2014word") == "word - word"
    assert normalise("word\u2014 word") == "word - word"
    assert normalise("a \u2014 b") == "a - b"


def test_an_unanswered_part_is_quoted_back_only_to_a_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A part is whatever the patient typed. Echoing it unbounded would
    repeat an arbitrarily long or hostile string back to them."""
    from app.agents.inquiry import MAX_QUOTED_PART

    assert MAX_QUOTED_PART <= 100
