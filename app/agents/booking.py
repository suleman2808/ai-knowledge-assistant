"""Booking Agent — turns a request in English into a calendar appointment.

The flow is deliberately not "ask the model to book something". An LLM is
used for the one job it is good at — pulling structured fields out of
free text, including relative dates — and every decision after that is
ordinary deterministic code:

    message -> LLM extraction -> validation -> availability -> booking

That split matters. Whether 2pm next Tuesday is inside opening hours, and
whether it clashes with an existing patient, are questions with exact
answers. Asking a language model to decide them would introduce a failure
mode with no upside: the model cannot see the calendar, and a confidently
double-booked appointment is worse than no booking at all.

The agent is conversational where it has to be. If required details are
missing it asks for precisely those and sets `needs_followup`, so the
graph knows the exchange is mid-flow.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

from app.agents.base import AgentResponse, failure
from app.agents.dates import resolve as resolve_relative_date
from app.integrations.calendar import (
    Appointment,
    CalendarBackend,
    CalendarError,
    TimeSlot,
    closing_reason,
    duration_for,
    get_calendar,
    is_open,
    open_slots,
)
from app.llm import LLMError, complete_json
from app.prompts import render

logger = logging.getLogger(__name__)

AGENT_NAME = "booking"

# How far ahead a booking may be made. Beyond this the clinic's diary is
# not considered reliable.
MAX_DAYS_AHEAD = 120

PHONE_RE = re.compile(r"[\d\s().+-]{7,}")


def _format_time(moment: datetime) -> str:
    """Render a time as '2:00 PM' without a platform-specific format flag."""
    return moment.strftime("%I:%M %p").lstrip("0")


def _spoken_date(iso_date: str) -> str:
    """Render '2026-09-21' as 'Monday 21 September', or pass it through."""
    try:
        parsed = datetime.strptime(iso_date, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return str(iso_date)

    today = date.today()
    if parsed == today:
        return "today"
    if parsed == today + timedelta(days=1):
        return "tomorrow"
    return parsed.strftime("%A %d %B").replace(" 0", " ")


def _format_slot(slot: TimeSlot) -> str:
    """Render a slot as 'Tuesday 23 September at 2:00 PM'."""
    day = slot.start.strftime("%A %d %B").replace(" 0", " ")
    return f"{day} at {_format_time(slot.start)}"


def _format_history(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "(no previous messages)"
    lines = []
    for turn in history[-6:]:
        role = "Patient" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no previous messages)"


def _parse_date(value: object) -> date | None:
    """Parse the model's date field, rejecting anything implausible."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.strptime(value.strip(), "%Y-%m-%d").date()
    except ValueError:
        logger.warning("Model returned an unparseable date: %r", value)
        return None

    today = date.today()
    if parsed < today:
        # A date in the past usually means the model mis-resolved a
        # relative reference. Treating it as missing prompts the patient
        # to restate it, which is safer than silently shifting it.
        logger.info("Discarding past date %s from extraction", parsed)
        return None
    if parsed > today + timedelta(days=MAX_DAYS_AHEAD):
        return None
    return parsed


def _parse_time(value: object) -> tuple[int, int] | None:
    """Parse 'HH:MM' into hour and minute."""
    if not value or not isinstance(value, str):
        return None
    match = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", value)
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if 0 <= hour <= 23 and 0 <= minute <= 59:
        return hour, minute
    return None


def _clean_phone(value: object) -> str | None:
    """Keep a phone number only if it plausibly is one."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    digits = re.sub(r"\D", "", text)
    if len(digits) < 7 or not PHONE_RE.fullmatch(text):
        return None
    return text


def _preference_window(preference: str | None) -> tuple[int, int] | None:
    """Map a vague time preference onto an hour range."""
    return {
        "morning": (8, 12),
        "afternoon": (12, 17),
        "evening": (17, 19),
    }.get((preference or "").lower())


def _ask_for(missing: list[str], extracted: dict) -> AgentResponse:
    """Request the specific details still needed, and nothing else."""
    prompts = {
        "service": "what you'd like to come in for",
        "date": "which day suits you",
        "time": "what time of day works best",
        "patient_name": "your full name",
        "phone": "a contact number",
    }
    wanted = [prompts[field] for field in missing if field in prompts]

    if len(wanted) == 1:
        request = wanted[0]
    else:
        request = ", ".join(wanted[:-1]) + f" and {wanted[-1]}"

    known = []
    if extracted.get("service"):
        known.append(f"a {extracted['service']}")
    if extracted.get("date"):
        # Read back the date the way a person would say it. An ISO string
        # in a sentence to a patient looks like a leaked internal field.
        known.append(f"for {_spoken_date(extracted['date'])}")
    if extracted.get("time_preference"):
        known.append(f"in the {extracted['time_preference']}")

    opener = (
        f"Happy to book {' '.join(known)}. " if known else "Happy to get you booked in. "
    )

    return AgentResponse(
        answer=f"{opener}Could you tell me {request}?",
        agent=AGENT_NAME,
        success=True,
        needs_followup=True,
        metadata={"stage": "collecting_details", "missing": missing, "extracted": extracted},
    )


def _offer_alternatives(
    backend: CalendarBackend,
    day: date,
    duration: int,
    *,
    reason: str,
    extracted: dict,
) -> AgentResponse:
    """Propose the nearest bookable slots when the request cannot be met."""
    options: list[TimeSlot] = []
    cursor = day
    for _ in range(14):
        if len(options) >= 3:
            break
        try:
            options.extend(open_slots(backend, cursor, duration, limit=2))
        except CalendarError as exc:
            return failure(AGENT_NAME, "unexpected", str(exc), stage="alternatives")
        cursor += timedelta(days=1)

    if not options:
        return AgentResponse(
            answer=(
                f"{reason}. I couldn't find anything in the next fortnight either — "
                f"please call us on (503) 555-0142 and reception will find you a time."
            ),
            agent=AGENT_NAME,
            success=True,
            needs_followup=True,
            metadata={"stage": "no_availability", "extracted": extracted},
        )

    listed = "\n".join(f"- {_format_slot(slot)}" for slot in options[:3])
    return AgentResponse(
        answer=f"{reason}. Here's what I do have:\n\n{listed}\n\nWould any of those work?",
        agent=AGENT_NAME,
        success=True,
        needs_followup=True,
        metadata={
            "stage": "offered_alternatives",
            "offered": [s.start.isoformat() for s in options[:3]],
            "extracted": extracted,
        },
    )


def _confirm(appointment: Appointment, backend_name: str, extracted: dict) -> AgentResponse:
    """Confirm a booking that has actually been written to the calendar.

    The wording depends on where it was written. Against a real calendar
    the clinic's reminder policy applies. Against the in-memory
    fallback — which is what any deployed instance uses, because the
    OAuth token is deliberately not committed — the booking exists only
    in that process, and saying otherwise would be a lie the patient
    discovers when nothing arrives. Observed in exactly that way: a
    booking was made on the hosted demo and the confirmation promised
    reminders that no part of this system can send.
    """
    when = _format_slot(TimeSlot(appointment.start, appointment.end))

    if backend_name == "in_memory":
        closing = (
            "This is a demonstration, so the appointment is held in memory "
            "rather than written to the clinic's diary, and no reminder is "
            "sent. Against a connected calendar it would be a real booking."
        )
    else:
        closing = (
            "We'll send a reminder three days before, and again on the "
            "morning. If you need to change it, we ask for 48 hours' notice."
        )

    return AgentResponse(
        answer=(
            f"You're booked in, {appointment.patient_name} — "
            f"{appointment.summary.lower()} on {when}. {closing}"
        ),
        agent=AGENT_NAME,
        success=True,
        metadata={
            "stage": "booked",
            "event_id": appointment.event_id,
            "start": appointment.start.isoformat(),
            "end": appointment.end.isoformat(),
            "calendar_backend": backend_name,
            "extracted": extracted,
        },
    )


def handle_booking(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
    backend: CalendarBackend | None = None,
) -> AgentResponse:
    """Handle an appointment request.

    Args:
        message: The patient's message, verbatim.
        history: Prior turns, so details given earlier are not re-asked.
        backend: Calendar to book against. Injectable for tests.

    Returns:
        An `AgentResponse`. `needs_followup` is True whenever the agent
        has asked the patient a question rather than completed a booking.
    """
    calendar = backend or get_calendar()
    today = date.today()

    try:
        extracted = complete_json(
            render(
                "booking_extract",
                today=today.isoformat(),
                weekday=today.strftime("%A"),
                history=_format_history(history),
                message=message,
            )
        )
    except LLMError as exc:
        logger.error("Booking extraction failed: %s", exc)
        return failure(AGENT_NAME, "llm_unavailable", str(exc), stage="extraction")

    service = (extracted.get("service") or "").strip() or None

    # The model reports the phrase; code resolves it. Where the two
    # disagree, the function wins — it is arithmetic, and the model was
    # observed booking "next Tuesday" a week late in production while
    # getting the same phrase right in testing.
    phrase = (extracted.get("date_phrase") or "").strip() or None
    resolved = resolve_relative_date(phrase, today=today)
    model_date = _parse_date(extracted.get("date"))

    if resolved and model_date and resolved != model_date:
        logger.info(
            "Relative date %r resolved to %s; model said %s. Using %s.",
            phrase, resolved, model_date, resolved,
        )
    booking_date = _parse_date(resolved.isoformat()) if resolved else model_date
    clock = _parse_time(extracted.get("time"))
    preference = extracted.get("time_preference")
    name = (extracted.get("patient_name") or "").strip() or None
    phone = _clean_phone(extracted.get("phone"))
    notes = (extracted.get("notes") or "").strip()

    # Normalise what the model returned before deciding what is missing,
    # so the record in metadata reflects what was actually used.
    normalised = {
        "service": service,
        "date": booking_date.isoformat() if booking_date else None,
        "date_phrase": phrase,
        "date_resolved_in_code": bool(resolved),
        "time": f"{clock[0]:02d}:{clock[1]:02d}" if clock else None,
        "time_preference": preference,
        "patient_name": name,
        "phone": phone,
        "notes": notes or None,
    }

    missing = [
        field
        for field, value in (
            ("service", service),
            ("date", booking_date),
            ("patient_name", name),
            ("phone", phone),
        )
        if not value
    ]
    if not clock and not preference:
        missing.append("time")

    if missing:
        return _ask_for(missing, normalised)

    duration = duration_for(service or "")
    assert booking_date is not None  # guaranteed by the `missing` check

    # An explicit time is honoured if bookable; a vague preference is
    # resolved into the first free slot inside that window.
    if clock:
        start = datetime.combine(booking_date, datetime.min.time()).replace(
            hour=clock[0], minute=clock[1]
        )
        if start <= datetime.now():
            return _offer_alternatives(
                calendar, today, duration,
                reason="That time has already passed",
                extracted=normalised,
            )
        if not is_open(start):
            return _offer_alternatives(
                calendar, booking_date, duration,
                reason=f"I can't book {_format_time(start)} — {closing_reason(start)}",
                extracted=normalised,
            )

        requested = TimeSlot(start, start + timedelta(minutes=duration))
        try:
            busy = calendar.busy_periods(booking_date)
        except CalendarError as exc:
            logger.error("Calendar read failed: %s", exc)
            return failure(AGENT_NAME, "unexpected", str(exc), stage="availability")

        if any(requested.overlaps(period) for period in busy):
            return _offer_alternatives(
                calendar, booking_date, duration,
                reason=f"{_format_time(start)} is already taken",
                extracted=normalised,
            )
        slot = requested
    else:
        window = _preference_window(preference)
        try:
            candidates = open_slots(calendar, booking_date, duration, limit=12)
        except CalendarError as exc:
            logger.error("Calendar read failed: %s", exc)
            return failure(AGENT_NAME, "unexpected", str(exc), stage="availability")

        if window:
            candidates = [c for c in candidates if window[0] <= c.start.hour < window[1]]
        if not candidates:
            return _offer_alternatives(
                calendar, booking_date, duration,
                reason=f"I don't have a {preference or 'free'} slot on that day",
                extracted=normalised,
            )
        slot = candidates[0]

    try:
        appointment = calendar.create_appointment(
            slot,
            summary=(service or "Appointment").capitalize(),
            patient_name=name or "",
            phone=phone or "",
            notes=notes,
        )
    except CalendarError as exc:
        # A clash detected at write time means someone booked the slot
        # between our check and our write. Offering alternatives is the
        # correct response, not an error.
        logger.warning("Booking write failed: %s", exc)
        return _offer_alternatives(
            calendar, booking_date, duration,
            reason="That slot was taken while we were talking",
            extracted=normalised,
        )
    except Exception as exc:
        logger.exception("Unexpected booking failure")
        return failure(AGENT_NAME, "unexpected", f"{type(exc).__name__}: {exc}", stage="create")

    return _confirm(appointment, calendar.name, normalised)
