"""Authorise Google Calendar access. Run once, from a terminal.

    python -m scripts.google_auth            # authorise and verify
    python -m scripts.google_auth --check    # verify an existing token
    python -m scripts.google_auth --revoke   # delete the stored token

This opens a browser for Google's consent screen. It is a separate
script rather than part of the app because a browser prompt must never
happen inside an HTTP request — the server would hang on a dialogue
nobody can see.

Skipping this entirely is fine. Without credentials the Booking Agent
uses an in-memory calendar and the whole project still runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from app.config import settings
from app.integrations.calendar import CalendarError

SETUP_STEPS = """
To get credentials.json:

 1. Go to https://console.cloud.google.com/  and sign in.
 2. Create a project (top-left project picker -> New Project).
    Name it anything, e.g. "dental-assistant".
 3. Enable the API:
    APIs & Services -> Library -> search "Google Calendar API" -> Enable.
 4. Configure the consent screen:
    APIs & Services -> OAuth consent screen
      - User type: External
      - App name, your email for both support and developer contact
      - Scopes: skip, this script requests them
      - Test users: ADD YOUR OWN GOOGLE ADDRESS. Without this,
        authorisation fails with "access_denied".
 5. Create the client:
    APIs & Services -> Credentials -> Create Credentials
      -> OAuth client ID -> Application type: Desktop app
 6. Download the JSON and save it in the project root as
    credentials.json  (it is gitignored).

Then run this script again.
"""


def check() -> int:
    """Verify that a stored token works, and show the next free slots."""
    from app.integrations.calendar import duration_for, open_slots
    from app.integrations.google_calendar import GoogleCalendar

    try:
        calendar = GoogleCalendar(interactive=False)
        info = calendar.check()
    except CalendarError as exc:
        print(f"Not connected: {exc}")
        return 1

    print("Connected to Google Calendar")
    print(f"  calendar : {info['summary']}  ({info['id']})")
    print(f"  Google timezone : {info['timezone']}")
    print(f"  clinic timezone : {settings.clinic_timezone}")

    if info["timezone"] and info["timezone"] != settings.clinic_timezone:
        print(
            f"\n  Note: the calendar's own timezone differs from the clinic's. "
            f"Events are written with an explicit timezone so this is safe, "
            f"but times will display in {info['timezone']} in Google's UI."
        )

    print("\nNext available 40-minute slots:")
    found = 0
    day = date.today()
    for _ in range(14):
        if found >= 5:
            break
        try:
            for slot in open_slots(calendar, day, duration_for("cleaning"), limit=2):
                print(f"  {slot.start:%A %d %B  %H:%M}")
                found += 1
        except CalendarError as exc:
            print(f"  Could not read {day}: {exc}")
            return 1
        day += timedelta(days=1)

    if not found:
        print("  (none in the next fortnight — is the calendar full?)")
    return 0


def authorise() -> int:
    """Run the consent flow, then verify."""
    creds_path = settings.abs_path(settings.google_credentials_file)
    if not creds_path.is_file():
        print(f"No OAuth client file at {creds_path}")
        print(SETUP_STEPS)
        return 1

    print("Opening your browser for Google's consent screen...")
    print("If it does not open, the URL is printed below.\n")

    from app.integrations.google_calendar import GoogleCalendar

    try:
        GoogleCalendar(interactive=True)
    except CalendarError as exc:
        print(f"\nAuthorisation failed: {exc}")
        if "access_denied" in str(exc).lower():
            print(
                "\nThis usually means your Google address is not listed as a "
                "test user on the OAuth consent screen. Add it under "
                "APIs & Services -> OAuth consent screen -> Test users."
            )
        return 1

    token_path = settings.abs_path(settings.google_token_file)
    print(f"\nAuthorised. Token saved to {token_path.name} (gitignored).\n")
    return check()


def revoke() -> int:
    """Delete the stored token, forcing re-authorisation next time."""
    token_path = settings.abs_path(settings.google_token_file)
    if not token_path.is_file():
        print("No stored token to remove.")
        return 0
    token_path.unlink()
    print(f"Deleted {token_path.name}. The Booking Agent will use the "
          f"in-memory calendar until you authorise again.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify an existing token.")
    parser.add_argument("--revoke", action="store_true", help="Delete the stored token.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    if args.revoke:
        return revoke()
    if args.check:
        return check()
    return authorise()


if __name__ == "__main__":
    sys.exit(main())
