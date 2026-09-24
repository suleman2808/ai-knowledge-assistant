"""Resolve relative dates in code rather than trusting the model.

"Next Tuesday" is an exact question with an exact answer, and the
Booking Agent already decides every other exact question — opening
hours, clashes, appointment length — in code. Date resolution was the
exception, and it was wrong in production: on Thursday 24 September the
model booked "next Tuesday" as 6 October, a week late. It had resolved
the same phrase correctly in earlier tests, which is the problem: a
model that is usually right about dates is worse than a function that
is always right, because the failures are rare enough to ship.

So the model now reports the *phrase* it saw, verbatim, and this module
turns it into a date. The model keeps the job it is good at — spotting
that "week on Tuesday" is a date reference at all — and loses the job it
is unreliable at.

The conventions here are choices, not facts, because English is genuinely
ambiguous. They are the ones a receptionist would use:

- **"Tuesday"** alone — the next Tuesday to occur. Said on a Tuesday, it
  means today.
- **"this Tuesday"** — the Tuesday of the current week if it has not
  passed, otherwise the next one.
- **"next Tuesday"** — the Tuesday of *next* week, where a week starts
  on Monday. Said on Thursday 24 September, that is 29 September.
- **"week on Tuesday" / "a week on Tuesday"** — seven days after the
  next Tuesday.

When a phrase does not match any of these, the model's own date is used.
This narrows the model's responsibility rather than removing it.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

WEEKDAYS = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_WEEKDAY_PATTERN = "|".join(sorted(WEEKDAYS, key=len, reverse=True))

RE_WEEK_ON = re.compile(rf"\b(?:a\s+)?week\s+on\s+({_WEEKDAY_PATTERN})\b")
RE_NEXT = re.compile(rf"\bnext\s+({_WEEKDAY_PATTERN})\b")
RE_THIS = re.compile(rf"\bthis\s+(?:coming\s+)?({_WEEKDAY_PATTERN})\b")
RE_BARE = re.compile(rf"\b(?:on\s+)?({_WEEKDAY_PATTERN})\b")
RE_IN_DAYS = re.compile(r"\bin\s+(\d{1,2})\s+days?\b")
RE_IN_WEEKS = re.compile(r"\bin\s+(\d{1,2})\s+weeks?\b")


def _next_occurrence(target: int, today: date, *, allow_today: bool = True) -> date:
    """The next date falling on `target` weekday."""
    ahead = (target - today.weekday()) % 7
    if ahead == 0 and not allow_today:
        ahead = 7
    return today + timedelta(days=ahead)


def _start_of_week(day: date) -> date:
    """The Monday of `day`'s week."""
    return day - timedelta(days=day.weekday())


def resolve(phrase: str | None, *, today: date | None = None) -> date | None:
    """Turn a relative date phrase into a date, or None if it is not one.

    Args:
        phrase: What the patient wrote, e.g. "next tuesday afternoon".
        today: Reference date. Injectable so tests do not depend on when
            they run.

    Returns:
        The resolved date, or None when the phrase carries no resolvable
        relative reference — in which case the caller should fall back to
        whatever the model extracted.
    """
    if not phrase:
        return None

    text = phrase.lower().strip()
    today = today or date.today()

    # Order matters throughout: each rule is checked before the less
    # specific rule whose words it contains. "day after tomorrow"
    # contains "tomorrow", and a test caught that matching the wrong
    # way round.
    if re.search(r"\bday\s+after\s+tomorrow\b", text):
        return today + timedelta(days=2)
    if re.search(r"\btomorrow\b", text):
        return today + timedelta(days=1)
    if re.search(r"\btoday\b|\bthis\s+afternoon\b|\bthis\s+morning\b|\btonight\b", text):
        return today

    match = RE_IN_DAYS.search(text)
    if match:
        return today + timedelta(days=int(match.group(1)))

    match = RE_IN_WEEKS.search(text)
    if match:
        return today + timedelta(weeks=int(match.group(1)))

    # Most specific weekday phrasing first: "week on Tuesday" also
    # contains "Tuesday", and "next Tuesday" would match the bare rule.
    match = RE_WEEK_ON.search(text)
    if match:
        return _next_occurrence(WEEKDAYS[match.group(1)], today, allow_today=False) \
            + timedelta(days=7)

    match = RE_NEXT.search(text)
    if match:
        # The named day in the week beginning next Monday.
        return _start_of_week(today) + timedelta(days=7 + WEEKDAYS[match.group(1)])

    match = RE_THIS.search(text)
    if match:
        return _next_occurrence(WEEKDAYS[match.group(1)], today, allow_today=True)

    if re.search(r"\bnext\s+week\b", text):
        return _start_of_week(today) + timedelta(days=7)

    match = RE_BARE.search(text)
    if match:
        return _next_occurrence(WEEKDAYS[match.group(1)], today, allow_today=True)

    return None
