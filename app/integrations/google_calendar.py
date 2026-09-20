"""Google Calendar backend for the Booking Agent.

Implements the same `CalendarBackend` protocol as `InMemoryCalendar`, so
the Booking Agent is unchanged by which one is in use.

## Time zones

The rest of the project uses naive datetimes and treats them as the
clinic's wall-clock time, which is the right model for a business whose
opening hours are "Monday 8am to 5pm" regardless of where a patient is
sitting. Google works in absolute time and requires RFC 3339.

This module is the only place that converts between the two. Naive in,
naive out; timezone-aware strings exist only inside these functions.
Confining the conversion to one boundary is what stops timezone bugs
spreading, and those bugs are unusually nasty here — an appointment an
hour out is worse than no appointment, because the patient arrives.

## Authentication

The OAuth consent flow opens a browser, which must never happen during
an HTTP request. So the interactive flow lives in
`scripts/google_auth.py` and runs once, from a terminal. At runtime this
class only *loads* a stored token, refreshing it silently if expired. If
there is no usable token it raises, and `get_calendar()` falls back to
the in-memory calendar rather than hanging a web request on a browser
prompt that nobody will see.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.config import settings
from app.integrations.calendar import Appointment, CalendarError, TimeSlot

logger = logging.getLogger(__name__)

# Minimum scope that allows reading and writing events. Deliberately not
# `calendar`, which would also permit deleting entire calendars — an
# authorisation prompt asking for more than the app needs is both a
# security smell and something a client will query.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# Event colour in Google Calendar. 9 is a blue-violet that stands out
# from manually created entries, so clinic staff can see at a glance
# which bookings came from the assistant.
ASSISTANT_COLOUR_ID = "9"


def clinic_tz() -> ZoneInfo:
    """Return the clinic's timezone, with a clear error if unavailable."""
    try:
        return ZoneInfo(settings.clinic_timezone)
    except Exception as exc:
        raise CalendarError(
            f"Unknown timezone {settings.clinic_timezone!r}: {exc}. "
            f"On Windows this usually means the `tzdata` package is missing."
        ) from exc


def to_rfc3339(naive_local: datetime) -> str:
    """Convert a clinic wall-clock time to an absolute RFC 3339 string."""
    return naive_local.replace(tzinfo=clinic_tz()).isoformat()


def from_rfc3339(value: str) -> datetime:
    """Convert an absolute time from Google back to clinic wall-clock.

    Google returns all-day events as a bare date, which has no time at
    all; those are treated as starting at midnight local.
    """
    text = value.strip()
    if len(text) == 10:  # "2026-09-29", an all-day event
        return datetime.fromisoformat(text)

    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(clinic_tz()).replace(tzinfo=None)


class GoogleCalendar:
    """Read and write appointments in a real Google Calendar."""

    def __init__(self, *, interactive: bool = False) -> None:
        """Build the client.

        Args:
            interactive: Allow the browser-based consent flow. Only the
                setup script passes True. At runtime this stays False so
                a missing token fails fast into the in-memory fallback
                instead of blocking on a prompt nobody can answer.

        Raises:
            CalendarError: No usable credentials, or the API is
                unreachable.
        """
        self._interactive = interactive
        self._service: Any = None
        self._calendar_id = settings.google_calendar_id
        self._service = self._build_service()

    @property
    def name(self) -> str:
        return "google"

    # -- Authentication ---------------------------------------------------

    def _load_credentials(self) -> Any:
        """Load, refresh or obtain OAuth credentials."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise CalendarError(f"Google client libraries missing: {exc}") from exc

        token_path = settings.abs_path(settings.google_token_file)
        creds_path = settings.abs_path(settings.google_credentials_file)

        credentials = None
        if token_path.is_file():
            try:
                credentials = Credentials.from_authorized_user_file(
                    str(token_path), SCOPES
                )
            except Exception as exc:
                logger.warning("Stored token is unreadable (%s); ignoring it", exc)

        if credentials and credentials.valid:
            return credentials

        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
                token_path.write_text(credentials.to_json(), encoding="utf-8")
                logger.info("Refreshed Google Calendar token")
                return credentials
            except Exception as exc:
                # A revoked or expired refresh token cannot be recovered
                # without the user consenting again.
                logger.warning("Token refresh failed (%s); re-authorisation needed", exc)

        if not self._interactive:
            raise CalendarError(
                f"No usable Google credentials at {token_path.name}. "
                f"Run `python -m scripts.google_auth` once to authorise."
            )

        if not creds_path.is_file():
            raise CalendarError(
                f"No OAuth client file at {creds_path}. Download it from the "
                f"Google Cloud console and save it there."
            )

        flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
        credentials = flow.run_local_server(port=0, prompt="consent")
        token_path.write_text(credentials.to_json(), encoding="utf-8")
        logger.info("Stored new Google Calendar token at %s", token_path)
        return credentials

    def _build_service(self) -> Any:
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise CalendarError(f"google-api-python-client missing: {exc}") from exc

        credentials = self._load_credentials()
        try:
            # cache_discovery=False avoids a noisy warning and a file cache
            # that is useless in a server process.
            return build(
                "calendar", "v3", credentials=credentials, cache_discovery=False
            )
        except Exception as exc:
            raise CalendarError(f"Could not build the Calendar client: {exc}") from exc

    # -- Protocol ---------------------------------------------------------

    def busy_periods(self, day: date) -> list[TimeSlot]:
        """Return existing events on `day`, in clinic wall-clock time.

        Raises:
            CalendarError: The API call failed.
        """
        start = datetime.combine(day, datetime.min.time())
        end = start + timedelta(days=1)

        try:
            response = (
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=to_rfc3339(start),
                    timeMax=to_rfc3339(end),
                    # Expands recurring events into individual instances;
                    # without it a weekly staff meeting appears once and
                    # every other week looks free.
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=100,
                )
                .execute()
            )
        except Exception as exc:
            raise CalendarError(f"Could not read the calendar: {exc}") from exc

        periods: list[TimeSlot] = []
        for event in response.get("items", []):
            # An event the organiser has declined, or marked free, does
            # not block the diary.
            if event.get("transparency") == "transparent":
                continue
            if event.get("status") == "cancelled":
                continue

            start_raw = (event.get("start") or {}).get("dateTime") or (
                event.get("start") or {}
            ).get("date")
            end_raw = (event.get("end") or {}).get("dateTime") or (
                event.get("end") or {}
            ).get("date")
            if not start_raw or not end_raw:
                continue

            try:
                periods.append(TimeSlot(from_rfc3339(start_raw), from_rfc3339(end_raw)))
            except ValueError as exc:
                logger.warning("Skipping event with unparseable time: %s", exc)

        return periods

    def create_appointment(
        self,
        slot: TimeSlot,
        *,
        summary: str,
        patient_name: str,
        phone: str = "",
        notes: str = "",
    ) -> Appointment:
        """Create an event, re-checking for clashes immediately first.

        Google has no conditional insert, so a clash check cannot be
        atomic with the write. Re-checking here narrows the window to
        milliseconds rather than the seconds a conversation takes, and
        the Booking Agent already treats a `CalendarError` on write as
        "offer alternatives" rather than as an error.

        Raises:
            CalendarError: The slot is taken, or the write failed.
        """
        for existing in self.busy_periods(slot.start.date()):
            if slot.overlaps(existing):
                raise CalendarError(
                    f"{slot.start:%Y-%m-%d %H:%M} was taken between checking and booking"
                )

        description_lines = [f"Booked by the AI assistant for {patient_name}."]
        if phone:
            description_lines.append(f"Contact: {phone}")
        if notes:
            description_lines.append(f"Notes: {notes}")

        body = {
            "summary": f"{summary} — {patient_name}" if patient_name else summary,
            "description": "\n".join(description_lines),
            "start": {
                "dateTime": to_rfc3339(slot.start),
                "timeZone": settings.clinic_timezone,
            },
            "end": {
                "dateTime": to_rfc3339(slot.end),
                "timeZone": settings.clinic_timezone,
            },
            "colorId": ASSISTANT_COLOUR_ID,
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 3 * 24 * 60},
                    {"method": "popup", "minutes": 120},
                ],
            },
            # Recorded on the event itself so staff can tell where a
            # booking came from without consulting the analytics store.
            "extendedProperties": {
                "private": {
                    "source": "ai-assistant",
                    "patient_name": patient_name,
                    "patient_phone": phone,
                }
            },
        }

        try:
            created = (
                self._service.events()
                .insert(calendarId=self._calendar_id, body=body)
                .execute()
            )
        except Exception as exc:
            raise CalendarError(f"Could not create the appointment: {exc}") from exc

        logger.info(
            "Created Google Calendar event %s for %s at %s",
            created.get("id"), patient_name, slot.start,
        )

        return Appointment(
            event_id=str(created.get("id", "")),
            start=slot.start,
            end=slot.end,
            summary=summary,
            patient_name=patient_name,
            phone=phone,
            notes=notes,
        )

    def check(self) -> dict[str, Any]:
        """Confirm the connection works, using only the granted scope.

        Deliberately an `events.list` call rather than `calendars.get`.
        Reading calendar *metadata* requires the broader `calendar`
        scope, and widening the scope purely so a health check can print
        a calendar's name would undo the reason for requesting the
        narrow one. The check verifies what the app actually does:
        reading events.
        """
        today = date.today()
        try:
            response = (
                self._service.events()
                .list(
                    calendarId=self._calendar_id,
                    timeMin=to_rfc3339(datetime.combine(today, datetime.min.time())),
                    timeMax=to_rfc3339(
                        datetime.combine(today, datetime.min.time()) + timedelta(days=7)
                    ),
                    singleEvents=True,
                    maxResults=10,
                )
                .execute()
            )
        except Exception as exc:
            raise CalendarError(f"Could not read events: {exc}") from exc

        return {
            "id": self._calendar_id,
            # `summary` is the calendar's own name, returned by the list
            # endpoint without needing metadata access.
            "summary": response.get("summary", self._calendar_id),
            # Likewise the calendar's configured time zone.
            "timezone": response.get("timeZone", ""),
            "events_next_7_days": len(response.get("items", [])),
        }
