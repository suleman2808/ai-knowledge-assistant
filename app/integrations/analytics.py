"""Analytics: every turn recorded, and a summary a clinic owner can use.

Stored in SQLite, a single file created on first use. No server, no
account, no setup — and it is a real SQL database, so the summary is
ordinary queries rather than Python loops over a JSON file.

## What gets stored, and what does not

Patient messages contain names, phone numbers and email addresses.
Analytics needs to know *what* patients ask about, not *who* asked, so
phone numbers and emails are redacted before anything touches disk. The
redaction happens at the write boundary, in one function, so no caller
can forget it.

Names are harder to detect reliably and are not redacted. That is a
stated limitation, not an oversight: a production deployment would add
named-entity redaction or not store message text at all.

## Failure policy

Logging must never break a conversation. If the database is locked,
missing or corrupt, the failure is logged and the patient still gets
their answer. An analytics outage is an inconvenience; a chat that
errors because a log write failed is a product defect.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from app.config import settings

logger = logging.getLogger(__name__)

# Phone-number candidates: a run of digits and the separators people use.
# The lookarounds exclude runs attached to an identifier by a hyphen, so
# a complaint reference like CMP-20260920-4F2A is left intact — redacting
# it would break the traceability the reference exists to provide.
PHONE_RE = re.compile(r"(?<![\w-])\+?\d[\d\s().-]{5,}\d(?![\w-])")
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Fewer digits than this is not a phone number anywhere.
MIN_PHONE_DIGITS = 7


def _redact_phone(match: re.Match[str]) -> str:
    candidate = match.group(0)
    if ISO_DATE_RE.match(candidate.strip()):
        return candidate
    if sum(ch.isdigit() for ch in candidate) < MIN_PHONE_DIGITS:
        return candidate
    return "[phone]"

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at        TEXT    NOT NULL,
    session_id        TEXT    NOT NULL DEFAULT '',
    message           TEXT    NOT NULL,
    intent            TEXT    NOT NULL,
    secondary_intent  TEXT,
    confidence        REAL    NOT NULL DEFAULT 0,
    routed_by         TEXT    NOT NULL DEFAULT '',
    success           INTEGER NOT NULL DEFAULT 1,
    grounded          INTEGER,
    retrieval_status  TEXT,
    best_score        REAL,
    needs_followup    INTEGER NOT NULL DEFAULT 0,
    escalated         INTEGER NOT NULL DEFAULT 0,
    booked            INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    error_kind        TEXT
);

CREATE INDEX IF NOT EXISTS idx_turns_created ON turns (created_at);
CREATE INDEX IF NOT EXISTS idx_turns_intent  ON turns (intent);

CREATE TABLE IF NOT EXISTS complaints (
    reference           TEXT PRIMARY KEY,
    received_at         TEXT NOT NULL,
    message             TEXT NOT NULL,
    summary             TEXT NOT NULL,
    category            TEXT NOT NULL,
    severity            TEXT NOT NULL,
    escalated           INTEGER NOT NULL,
    escalation_reasons  TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_complaints_received ON complaints (received_at);
"""


def redact(text: str) -> str:
    """Remove phone numbers and email addresses from free text."""
    text = EMAIL_RE.sub("[email]", text or "")
    return PHONE_RE.sub(_redact_phone, text)


def db_path() -> Path:
    return settings.abs_path(settings.analytics_db)


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open a connection, ensuring the schema exists.

    A fresh connection per operation rather than a shared one, because
    sqlite3 connections are not safe to share across the threads FastAPI
    uses. SQLite opens in microseconds, so this costs nothing measurable.
    """
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target, timeout=5.0)
    connection.row_factory = sqlite3.Row
    try:
        # WAL lets the analytics endpoint read while a turn is being
        # written, instead of one blocking the other.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def log_turn(turn: dict[str, Any], *, latency_ms: int = 0, path: Path | None = None) -> bool:
    """Record one conversation turn. Never raises.

    Args:
        turn: The record produced by the graph's `finalise` node.
        latency_ms: End-to-end time for the turn.
        path: Database file. Injectable for tests.

    Returns:
        True if written, False if logging failed (the failure is logged).
    """
    meta = turn.get("agent_metadata") or {}

    grounded = meta.get("grounded")
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": str(turn.get("session_id") or ""),
        "message": redact(str(turn.get("message") or "")),
        "intent": str(turn.get("intent") or "unknown"),
        "secondary_intent": turn.get("secondary_intent"),
        "confidence": float(turn.get("confidence") or 0.0),
        "routed_by": str(turn.get("routed_by") or ""),
        "success": int(bool(turn.get("success", True))),
        "grounded": None if grounded is None else int(bool(grounded)),
        "retrieval_status": meta.get("retrieval_status"),
        "best_score": meta.get("best_score"),
        "needs_followup": int(bool(turn.get("needs_followup"))),
        "escalated": int(bool(meta.get("escalated"))),
        "booked": int(meta.get("stage") == "booked"),
        "latency_ms": int(latency_ms),
        "error_kind": meta.get("error_kind"),
    }

    try:
        with connect(path) as connection:
            columns = ", ".join(row)
            placeholders = ", ".join(f":{k}" for k in row)
            connection.execute(f"INSERT INTO turns ({columns}) VALUES ({placeholders})", row)
        return True
    except Exception as exc:
        logger.error("Analytics write failed (turn still served): %s", exc)
        return False


# --------------------------------------------------------------------------
# Complaint store, SQLite-backed
# --------------------------------------------------------------------------


class SQLiteComplaintStore:
    """Persistent replacement for the in-memory `ComplaintStore`.

    Same two-method interface, so the Complaint Agent is unchanged. Unlike
    turn logging, a complaint write failure *is* surfaced to the caller:
    losing a complaint silently is the one failure this project treats as
    unacceptable, so the agent needs to know it happened.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def record(self, complaint: Any) -> None:
        with connect(self.path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO complaints
                    (reference, received_at, message, summary, category,
                     severity, escalated, escalation_reasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    complaint.reference,
                    complaint.received_at.isoformat(timespec="seconds"),
                    redact(complaint.message),
                    redact(complaint.summary),
                    complaint.category,
                    complaint.severity,
                    int(complaint.escalated),
                    json.dumps(complaint.escalation_reasons),
                ),
            )
        logger.info(
            "Complaint %s stored: %s/%s escalated=%s",
            complaint.reference, complaint.category, complaint.severity,
            complaint.escalated,
        )

    def all(self) -> list[dict[str, Any]]:
        with connect(self.path) as connection:
            rows = connection.execute(
                "SELECT * FROM complaints ORDER BY received_at DESC"
            ).fetchall()
        return [
            {**dict(r), "escalated": bool(r["escalated"]),
             "escalation_reasons": json.loads(r["escalation_reasons"] or "[]")}
            for r in rows
        ]


# --------------------------------------------------------------------------
# Reading: the summary
# --------------------------------------------------------------------------


def summary(*, days: int = 30, path: Path | None = None) -> dict[str, Any]:
    """Answer "what are patients contacting us about, and how well?"

    Structured around questions a clinic owner would ask, not around the
    database schema:

    - What do people want? (intent mix)
    - Can the assistant answer them? (grounded rate, and which questions
      it could not answer — the knowledge gaps)
    - Is it creating work or saving it? (bookings made, escalations)
    - Is it fast and reliable? (latency, error rate)
    """
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    with connect(path) as c:
        total = c.execute(
            "SELECT COUNT(*) FROM turns WHERE created_at >= ?", (since,)
        ).fetchone()[0]

        intents = {
            r["intent"]: r["n"]
            for r in c.execute(
                "SELECT intent, COUNT(*) AS n FROM turns WHERE created_at >= ? "
                "GROUP BY intent ORDER BY n DESC",
                (since,),
            )
        }

        inquiry = c.execute(
            """
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN grounded = 1 THEN 1 ELSE 0 END)   AS answered,
                SUM(CASE WHEN grounded = 0 THEN 1 ELSE 0 END)   AS declined
            FROM turns WHERE intent = 'inquiry' AND created_at >= ?
            """,
            (since,),
        ).fetchone()

        # The most valuable output here. Every question the assistant had
        # to decline is a gap in the clinic's documents — something a
        # patient wanted to know that nobody has written down.
        gaps = [
            {"question": r["message"], "times": r["n"], "best_score": r["best"]}
            for r in c.execute(
                """
                SELECT message, COUNT(*) AS n, MAX(best_score) AS best
                FROM turns
                WHERE intent = 'inquiry' AND grounded = 0 AND success = 1
                  AND created_at >= ?
                GROUP BY LOWER(message)
                ORDER BY n DESC, best DESC
                LIMIT 10
                """,
                (since,),
            )
        ]

        outcomes = c.execute(
            """
            SELECT
                SUM(booked)                                   AS bookings,
                SUM(escalated)                                AS escalations,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END)  AS failures,
                SUM(CASE WHEN routed_by = 'keyword' THEN 1 ELSE 0 END) AS fast_path,
                SUM(CASE WHEN secondary_intent IS NOT NULL THEN 1 ELSE 0 END) AS mixed
            FROM turns WHERE created_at >= ?
            """,
            (since,),
        ).fetchone()

        latencies = [
            r[0]
            for r in c.execute(
                "SELECT latency_ms FROM turns WHERE created_at >= ? AND latency_ms > 0 "
                "ORDER BY latency_ms",
                (since,),
            )
        ]

        complaints = {
            r["severity"]: r["n"]
            for r in c.execute(
                "SELECT severity, COUNT(*) AS n FROM complaints WHERE received_at >= ? "
                "GROUP BY severity",
                (since,),
            )
        }
        complaint_categories = {
            r["category"]: r["n"]
            for r in c.execute(
                "SELECT category, COUNT(*) AS n FROM complaints WHERE received_at >= ? "
                "GROUP BY category ORDER BY n DESC",
                (since,),
            )
        }
        sessions = c.execute(
            "SELECT COUNT(DISTINCT session_id) FROM turns "
            "WHERE created_at >= ? AND session_id != ''",
            (since,),
        ).fetchone()[0]

    def pct(part: int | None, whole: int | None) -> float | None:
        return round(100 * (part or 0) / whole, 1) if whole else None

    def percentile(values: list[int], q: float) -> int | None:
        if not values:
            return None
        index = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
        return values[index]

    inquiry_total = inquiry["total"] or 0

    return {
        "window_days": days,
        "turns": total,
        "conversations": sessions,
        "intents": intents,
        "inquiries": {
            "total": inquiry_total,
            "answered": inquiry["answered"] or 0,
            "declined": inquiry["declined"] or 0,
            "answer_rate_pct": pct(inquiry["answered"], inquiry_total),
        },
        "knowledge_gaps": gaps,
        "bookings_made": outcomes["bookings"] or 0,
        "escalations": outcomes["escalations"] or 0,
        "complaints_by_severity": complaints,
        "complaints_by_category": complaint_categories,
        "mixed_intent_turns": outcomes["mixed"] or 0,
        "fast_path_pct": pct(outcomes["fast_path"], total),
        "failure_rate_pct": pct(outcomes["failures"], total),
        "latency_ms": {
            "p50": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
        },
    }
