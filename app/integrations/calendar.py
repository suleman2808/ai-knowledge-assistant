"""Calendar access for the Booking Agent.

Defined as a protocol with two implementations so the agent can be built
and tested before any OAuth exists, and so a fresh clone runs end to end
with no Google Cloud setup:

- `InMemoryCalendar` — a working fake with the clinic's real opening
  hours and a few pre-existing appointments. Used automatically when
  Google credentials are absent.
- `GoogleCalendar` — the real thing. Added in step 6.

The agent depends on the protocol, never on a concrete backend, so
swapping them changes nothing above this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Protocol, runtime_checkable

from app.config import settings

logger = logging.getLogger(__name__)


class CalendarError(RuntimeError):
    """The calendar could not be read or written."""


# Clinic opening hours, mirroring data/documents/hours-and-location.md.
# Keyed by weekday number, Monday = 0. A missing key means closed.
OPENING_HOURS: dict[int, tuple[time, time]] = {
    0: (time(8, 0), time(17, 0)),
    1: (time(8, 0), time(17, 0)),
    2: (time(8, 0), time(19, 0)),
    3: (time(8, 0), time(17, 0)),
    4: (time(8, 0), time(15, 0)),
    5: (time(9, 0), time(13, 0)),  # first and third Saturday only
}

# The last appointment starts one hour before closing.
LAST_BOOKING_BUFFER = timedelta(hours=1)

# Appointment length by service, from the pricing document. The default
# covers anything unrecognised.
SERVICE_DURATIONS: dict[str, int] = {
    "new patient exam": 45,
    "check-up": 20,
    "examination": 20,
    "cleaning": 40,
    "hygiene": 40,
    "deep cleaning": 60,
    "filling": 45,
    "crown": 90,
    "root canal": 90,
    "extraction": 30,
    "wisdom tooth": 60,
    "whitening": 90,
    "implant": 120,
    "orthodontic consultation": 45,
    "emergency": 30,
}
DEFAULT_DURATION_MINUTES = 30


@dataclass
class TimeSlot:
    """A candidate appointment time."""

    start: datetime
    end: datetime

    def overlaps(self, other: "TimeSlot") -> bool:
        return self.start < other.end and other.start < self.end

    def human(self) -> str:
        """Render for a patient, e.g. 'Tuesday 23 September at 2:00 PM'."""
        return self.start.strftime("%A %-d %B at %-I:%M %p") if _supports_dash() else (
            self.start.strftime("%A %d %B at %I:%M %p").replace(" 0", " ").lstrip("0")
        )


def _supports_dash() -> bool:
    """Whether strftime accepts the `%-d` no-padding flag.

    Available on Linux and macOS, absent on Windows. Checked once rather
    than assumed, because the project is developed on Windows and
    deployed on Linux.
    """
    try:
        datetime(2026, 1, 5).strftime("%-d")
    except ValueError:
        return False
    return True


@dataclass
class Appointment:
    """A booked appointment."""

    event_id: str
    start: datetime
    end: datetime
    summary: str
    patient_name: str = ""
    phone: str = ""
    notes: str = ""


@runtime_checkable
class CalendarBackend(Protocol):
    """What the Booking Agent needs from a calendar."""

    def busy_periods(self, day: date) -> list[TimeSlot]:
        """Return existing commitments on `day`."""
        ...

    def create_appointment(
        self,
        slot: TimeSlot,
        *,
        summary: str,
        patient_name: str,
        phone: str = "",
        notes: str = "",
    ) -> Appointment:
        """Book `slot` and return the created appointment."""
        ...

    @property
    def name(self) -> str:
        """Backend identifier, recorded in analytics."""
        ...


def duration_for(service: str) -> int:
    """Look up an appointment length in minutes for a described service.

    Matches on substrings because patients write "a cleaning please" and
    "deep cleaning for my gums" rather than canonical service names. The
    longest matching key wins, so "deep cleaning" beats "cleaning".
    """
    text = (service or "").lower()
    matches = [k for k in SERVICE_DURATIONS if k in text]
    if not matches:
        return DEFAULT_DURATION_MINUTES
    return SERVICE_DURATIONS[max(matches, key=len)]


def is_open(moment: datetime) -> bool:
    """Whether the clinic is open at `moment`, allowing for closing time."""
    hours = OPENING_HOURS.get(moment.weekday())
    if not hours:
        return False
    opens, closes = hours
    latest_start = (
        datetime.combine(moment.date(), closes) - LAST_BOOKING_BUFFER
    ).time()
    return opens <= moment.time() <= latest_start


def closing_reason(moment: datetime) -> str:
    """Explain, in patient-facing terms, why a time cannot be booked."""
    hours = OPENING_HOURS.get(moment.weekday())
    if not hours:
        return f"we're closed on {moment.strftime('%A')}s"
    opens, closes = hours
    return (
        f"our {moment.strftime('%A')} hours are "
        f"{opens.strftime('%I:%M %p').lstrip('0')} to "
        f"{closes.strftime('%I:%M %p').lstrip('0')}, and the last "
        f"appointment starts an hour before we close"
    )


def open_slots(
    backend: CalendarBackend,
    day: date,
    duration_minutes: int,
    *,
    limit: int = 4,
    granularity_minutes: int = 30,
) -> list[TimeSlot]:
    """Find bookable slots on `day` that do not clash with existing ones."""
    hours = OPENING_HOURS.get(day.weekday())
    if not hours:
        return []

    opens, closes = hours
    cursor = datetime.combine(day, opens)
    last_start = datetime.combine(day, closes) - LAST_BOOKING_BUFFER
    step = timedelta(minutes=granularity_minutes)
    length = timedelta(minutes=duration_minutes)

    try:
        busy = backend.busy_periods(day)
    except CalendarError as exc:
        logger.warning("Could not read busy periods for %s: %s", day, exc)
        raise

    found: list[TimeSlot] = []
    now = datetime.now()
    while cursor <= last_start and len(found) < limit:
        candidate = TimeSlot(cursor, cursor + length)
        if cursor > now and not any(candidate.overlaps(b) for b in busy):
            found.append(candidate)
        cursor += step
    return found


@dataclass
class InMemoryCalendar:
    """A working calendar that stores appointments in process memory.

    Not a mock with hard-coded returns — it enforces real clash detection,
    so the Booking Agent's logic is genuinely exercised. Bookings are lost
    when the process exits, which is correct for a demo and honest about
    what it is.

    Seeded with a few plausible existing appointments so that availability
    checking has something to collide with; a calendar that is always free
    proves nothing.
    """

    appointments: list[Appointment] | None = None
    """Existing appointments. `None` seeds a realistic diary; an explicit
    list — including an empty one — is used exactly as given, so tests can
    ask for a genuinely free calendar."""

    _counter: int = 0

    def __post_init__(self) -> None:
        if self.appointments is None:
            self.appointments = []
            self._seed()

    @property
    def name(self) -> str:
        return "in_memory"

    def _seed(self) -> None:
        """Pre-book a realistic scattering of the next fortnight."""
        base = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        pattern = [
            (1, 9, 0, 40, "Hygiene visit"),
            (1, 14, 0, 45, "Composite filling"),
            (2, 10, 30, 90, "Crown fitting"),
            (2, 15, 0, 20, "Check-up"),
            (3, 8, 30, 120, "Implant placement"),
            (4, 11, 0, 40, "Hygiene visit"),
            (7, 9, 0, 90, "Root canal"),
            (8, 14, 0, 45, "New patient exam"),
            (9, 16, 0, 30, "Extraction"),
        ]
        for offset, hour, minute, minutes, summary in pattern:
            start = base + timedelta(days=offset, hours=hour, minutes=minute)
            if OPENING_HOURS.get(start.weekday()) is None:
                continue
            self._counter += 1
            self.appointments.append(
                Appointment(
                    event_id=f"seed-{self._counter}",
                    start=start,
                    end=start + timedelta(minutes=minutes),
                    summary=summary,
                    patient_name="(existing patient)",
                )
            )

    def busy_periods(self, day: date) -> list[TimeSlot]:
        return [
            TimeSlot(a.start, a.end)
            for a in self.appointments
            if a.start.date() == day
        ]

    def create_appointment(
        self,
        slot: TimeSlot,
        *,
        summary: str,
        patient_name: str,
        phone: str = "",
        notes: str = "",
    ) -> Appointment:
        clash = [
            a
            for a in self.appointments
            if TimeSlot(a.start, a.end).overlaps(slot)
        ]
        if clash:
            raise CalendarError(
                f"{slot.start:%Y-%m-%d %H:%M} clashes with {clash[0].summary}"
            )

        self._counter += 1
        appointment = Appointment(
            event_id=f"local-{self._counter}",
            start=slot.start,
            end=slot.end,
            summary=summary,
            patient_name=patient_name,
            phone=phone,
            notes=notes,
        )
        self.appointments.append(appointment)
        logger.info("Booked %s for %s", slot.start, patient_name)
        return appointment


_backend: CalendarBackend | None = None


def get_calendar() -> CalendarBackend:
    """Return the calendar backend appropriate to this environment.

    Falls back to the in-memory calendar when Google credentials are
    absent, so the Booking Agent always has something that works. The
    choice is logged, because silently demoing against a fake calendar
    while believing it is real would be a genuinely bad surprise.
    """
    global _backend
    if _backend is not None:
        return _backend

    if settings.calendar_configured:
        try:
            from app.integrations.google_calendar import GoogleCalendar

            _backend = GoogleCalendar()
            logger.info("Using Google Calendar backend")
            return _backend
        except Exception as exc:
            logger.warning(
                "Google Calendar unavailable (%s); falling back to in-memory", exc
            )

    _backend = InMemoryCalendar()
    logger.info("Using in-memory calendar (no Google credentials configured)")
    return _backend


def reset_calendar(backend: CalendarBackend | None = None) -> None:
    """Replace the cached backend. For tests and for `--reset` in scripts."""
    global _backend
    _backend = backend
