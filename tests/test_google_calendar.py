"""Tests for the Google Calendar backend.

Run against a fake API client, so they need no credentials, no network
and no Google project. What is tested is our translation layer — time
zones, which events count as busy, clash re-checking, error wrapping —
which is where the bugs live. Google's API is not ours to test.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from app.integrations.calendar import CalendarError, TimeSlot
from app.integrations.google_calendar import (
    GoogleCalendar,
    from_rfc3339,
    to_rfc3339,
)


# ---------------------------------------------------------------------------
# A fake Google API client
# ---------------------------------------------------------------------------


class FakeExecutable:
    def __init__(self, result, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def execute(self):  # noqa: ANN201
        if self._error:
            raise self._error
        return self._result


class FakeEvents:
    def __init__(self, items: list[dict], error: Exception | None = None) -> None:
        self.items = items
        self.error = error
        self.list_kwargs: dict = {}
        self.inserted: list[dict] = []

    def list(self, **kwargs):  # noqa: ANN201
        self.list_kwargs = kwargs
        return FakeExecutable({"items": self.items}, self.error)

    def insert(self, *, calendarId, body):  # noqa: ANN001, N803, ANN201
        self.inserted.append({"calendarId": calendarId, "body": body})
        return FakeExecutable({"id": "evt-created"}, self.error)


class FakeService:
    def __init__(self, items: list[dict] | None = None, error: Exception | None = None):
        self._events = FakeEvents(items or [], error)

    def events(self):  # noqa: ANN201
        return self._events


def make_calendar(items: list[dict] | None = None, error: Exception | None = None):  # noqa: ANN201
    """Build a GoogleCalendar with a fake service, bypassing OAuth."""
    calendar = object.__new__(GoogleCalendar)
    calendar._interactive = False
    calendar._calendar_id = "primary"
    calendar._service = FakeService(items, error)
    return calendar


def timed_event(start: str, end: str, **extra) -> dict:  # noqa: ANN003
    return {"start": {"dateTime": start}, "end": {"dateTime": end}, **extra}


# ---------------------------------------------------------------------------
# Time zone conversion
# ---------------------------------------------------------------------------


def test_wall_clock_times_survive_a_round_trip() -> None:
    naive = datetime(2026, 9, 29, 14, 0)

    assert from_rfc3339(to_rfc3339(naive)) == naive


def test_daylight_saving_is_handled() -> None:
    """The same wall-clock time is a different absolute time in summer.

    Getting this wrong books patients an hour out, which is worse than
    not booking them at all — they turn up.
    """
    winter = to_rfc3339(datetime(2026, 1, 15, 14, 0))
    summer = to_rfc3339(datetime(2026, 7, 15, 14, 0))

    assert winter.endswith("-08:00"), "January is PST"
    assert summer.endswith("-07:00"), "July is PDT"


def test_utc_input_is_converted_to_clinic_local() -> None:
    assert from_rfc3339("2026-07-15T21:00:00Z") == datetime(2026, 7, 15, 14, 0)


def test_all_day_events_are_read_as_midnight() -> None:
    assert from_rfc3339("2026-09-29") == datetime(2026, 9, 29, 0, 0)


# ---------------------------------------------------------------------------
# Reading busy periods
# ---------------------------------------------------------------------------


def test_busy_periods_are_returned_in_clinic_time() -> None:
    calendar = make_calendar(
        [timed_event("2026-09-29T14:00:00-07:00", "2026-09-29T15:00:00-07:00")]
    )

    periods = calendar.busy_periods(date(2026, 9, 29))

    assert periods == [TimeSlot(datetime(2026, 9, 29, 14, 0), datetime(2026, 9, 29, 15, 0))]


def test_recurring_events_are_expanded() -> None:
    """Without singleEvents, a weekly meeting blocks only its first week."""
    calendar = make_calendar([])

    calendar.busy_periods(date(2026, 9, 29))

    assert calendar._service.events().list_kwargs["singleEvents"] is True


def test_events_marked_free_do_not_block_the_diary() -> None:
    calendar = make_calendar(
        [
            timed_event(
                "2026-09-29T14:00:00-07:00", "2026-09-29T15:00:00-07:00",
                transparency="transparent",
            )
        ]
    )

    assert calendar.busy_periods(date(2026, 9, 29)) == []


def test_cancelled_events_do_not_block_the_diary() -> None:
    calendar = make_calendar(
        [
            timed_event(
                "2026-09-29T14:00:00-07:00", "2026-09-29T15:00:00-07:00",
                status="cancelled",
            )
        ]
    )

    assert calendar.busy_periods(date(2026, 9, 29)) == []


def test_malformed_events_are_skipped_not_fatal() -> None:
    """One broken event must not make the whole day unreadable."""
    calendar = make_calendar(
        [
            {"start": {}, "end": {}},
            timed_event("2026-09-29T09:00:00-07:00", "2026-09-29T09:40:00-07:00"),
        ]
    )

    periods = calendar.busy_periods(date(2026, 9, 29))

    assert len(periods) == 1
    assert periods[0].start == datetime(2026, 9, 29, 9, 0)


def test_api_failure_becomes_a_calendar_error() -> None:
    """Callers handle CalendarError; they must not see Google's types."""
    calendar = make_calendar([], error=RuntimeError("503 backend error"))

    with pytest.raises(CalendarError, match="Could not read the calendar"):
        calendar.busy_periods(date(2026, 9, 29))


# ---------------------------------------------------------------------------
# Writing appointments
# ---------------------------------------------------------------------------


def slot_at(hour: int, minutes: int = 40) -> TimeSlot:
    start = datetime(2026, 9, 29, hour, 0)
    return TimeSlot(start, start + timedelta(minutes=minutes))


def test_creating_an_appointment_sends_an_explicit_timezone() -> None:
    calendar = make_calendar([])

    calendar.create_appointment(
        slot_at(14), summary="Cleaning", patient_name="Sarah Chen",
        phone="503-555-0180",
    )

    body = calendar._service.events().inserted[0]["body"]
    assert body["start"]["timeZone"] == "America/Los_Angeles"
    assert body["start"]["dateTime"].startswith("2026-09-29T14:00:00")
    assert "Sarah Chen" in body["summary"]
    assert "503-555-0180" in body["description"]


def test_created_events_are_tagged_as_assistant_bookings() -> None:
    """Clinic staff must be able to tell where a booking came from."""
    calendar = make_calendar([])

    calendar.create_appointment(slot_at(14), summary="Cleaning", patient_name="A B")

    body = calendar._service.events().inserted[0]["body"]
    assert body["extendedProperties"]["private"]["source"] == "ai-assistant"
    assert body["colorId"]


def test_clash_is_rechecked_immediately_before_writing() -> None:
    """Google has no conditional insert, so the window must be narrowed.

    A conversation takes seconds; re-checking here reduces the race to
    milliseconds. The Booking Agent turns this error into an offer of
    alternatives rather than a failure.
    """
    calendar = make_calendar(
        [timed_event("2026-09-29T14:00:00-07:00", "2026-09-29T15:00:00-07:00")]
    )

    with pytest.raises(CalendarError, match="taken between checking and booking"):
        calendar.create_appointment(slot_at(14), summary="Cleaning", patient_name="A B")

    assert calendar._service.events().inserted == [], "must not have written"


def test_a_free_slot_adjacent_to_a_busy_one_is_bookable() -> None:
    """Touching intervals do not overlap; off-by-one here loses slots."""
    calendar = make_calendar(
        [timed_event("2026-09-29T13:00:00-07:00", "2026-09-29T14:00:00-07:00")]
    )

    appointment = calendar.create_appointment(
        slot_at(14), summary="Cleaning", patient_name="Sarah Chen"
    )

    assert appointment.event_id == "evt-created"


def test_write_failure_becomes_a_calendar_error() -> None:
    class FailOnInsert(FakeService):
        def events(self):  # noqa: ANN201
            events = super().events()

            def _insert(*, calendarId, body):  # noqa: ANN001, N803, ANN202
                return FakeExecutable(None, RuntimeError("quota exceeded"))

            events.insert = _insert  # type: ignore[method-assign]
            return events

    calendar = object.__new__(GoogleCalendar)
    calendar._interactive = False
    calendar._calendar_id = "primary"
    calendar._service = FailOnInsert([])

    with pytest.raises(CalendarError, match="Could not create the appointment"):
        calendar.create_appointment(slot_at(14), summary="Cleaning", patient_name="A B")


def test_returned_appointment_carries_the_google_event_id() -> None:
    calendar = make_calendar([])

    appointment = calendar.create_appointment(
        slot_at(9), summary="Check-up", patient_name="Ana Silva", notes="anxious"
    )

    assert appointment.event_id == "evt-created"
    assert appointment.start == datetime(2026, 9, 29, 9, 0)
    assert appointment.notes == "anxious"


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------


def test_missing_token_raises_rather_than_prompting(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A browser prompt inside a web request would hang the server."""
    from app.config import settings

    monkeypatch.setattr(settings, "google_token_file", tmp_path / "absent.json")

    calendar = object.__new__(GoogleCalendar)
    calendar._interactive = False

    with pytest.raises(CalendarError, match="google_auth"):
        calendar._load_credentials()


def test_get_calendar_falls_back_when_google_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fresh clone must work with no Google setup at all."""
    from app.config import settings
    from app.integrations import calendar as calendar_module

    monkeypatch.setattr(
        type(settings), "calendar_configured", property(lambda _self: False)
    )
    calendar_module.reset_calendar(None)

    backend = calendar_module.get_calendar()

    assert backend.name == "in_memory"
    calendar_module.reset_calendar(None)
