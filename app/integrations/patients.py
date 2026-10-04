"""Recognise returning customers by phone number.

A login is the wrong instrument for this. People book a blood test once a
year; they will not remember a password, and asking for one at the start
of a conversation loses more customers than it identifies. A phone number
is the identifier someone will always give you and never forget, which is
why clinics and labs key on it in practice.

So there is no account system. If a number has been seen before, the
assistant knows the name attached to it and what was booked last time,
and can stop asking questions it already has answers to.

The obvious consequence: anyone who types someone else's number sees that
person's name and booking history. That is the same exposure a receptionist
gives over the phone, and acceptable for a demonstration, but it is the
reason this is not an authentication mechanism and must never be used as
one. Anything that actually needs protecting goes behind the dashboard
password instead.
"""

from __future__ import annotations

import logging
from typing import Any

from app.integrations.analytics import connect
from app.integrations.phone import normalise, phone_in  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

def find(phone: str | None) -> dict[str, Any] | None:
    """Look up a customer by phone number.

    Returns None when the number is unrecognised, unusable, or the
    database cannot be read — a failed lookup must never break a
    conversation, it just means the assistant does not recognise someone.
    """
    key = normalise(phone)
    if not key:
        return None

    try:
        with connect() as connection:
            row = connection.execute(
                "SELECT * FROM patients WHERE phone = ?", (key,)
            ).fetchone()
            if row is None:
                return None

            bookings = connection.execute(
                """
                SELECT service, starts_at, created_at
                FROM bookings
                WHERE phone_key = ?
                ORDER BY starts_at DESC
                LIMIT 3
                """,
                (key,),
            ).fetchall()
    except Exception as exc:
        logger.warning("Customer lookup failed for a number: %s", exc)
        return None

    return {
        "phone": key,
        "name": row["name"],
        "visit_count": row["visit_count"],
        "first_seen": row["first_seen"],
        "last_seen": row["last_seen"],
        "bookings": [dict(b) for b in bookings],
    }


def describe(patient: dict[str, Any] | None) -> str:
    """Render a customer's history for a prompt.

    Deliberately terse. This goes into every agent's context, and a
    paragraph about someone's history would crowd out the retrieved
    documents that the answer actually depends on.
    """
    if not patient:
        return "(not recognised - treat as a new customer)"

    parts = [f"Name: {patient['name'] or 'unknown'}"]
    parts.append(f"Previous bookings: {patient['visit_count']}")
    for booking in patient["bookings"][:2]:
        when = (booking["starts_at"] or "")[:10]
        parts.append(f"- {booking['service'] or 'appointment'} on {when}")
    return "\n".join(parts)
