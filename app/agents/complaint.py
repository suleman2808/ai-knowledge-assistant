"""Complaint Agent — records complaints, rates severity, escalates.

Two LLM calls with deterministic logic between them:

    message -> assessment (JSON) -> escalation rules -> log -> reply

The assessment is separated from the reply for a reason worth stating.
Asking one call to both judge severity and write a soothing response
biases the judgement: a model composing an apology tends to mirror the
patient's tone, so a furious message about a magazine rates higher than a
calm report of a clinical injury. Assessing first, in JSON, with no
audience, produces a more defensible rating — and it is the rating that
decides whether a human is alerted.

Escalation is then decided by **code**, not by the model. The model's
`requires_escalation` is taken as one input among several; any mention of
harm, legal action, or a critical rating escalates regardless of what the
model concluded. A missed escalation is the expensive failure here, and a
false positive costs one person one email.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.agents.base import AgentResponse, failure
from app.llm import LLMError, complete, complete_json
from app.prompts import render

logger = logging.getLogger(__name__)

AGENT_NAME = "complaint"

VALID_SEVERITIES = ("low", "medium", "high", "critical")
VALID_CATEGORIES = (
    "clinical",
    "billing",
    "waiting_time",
    "staff_conduct",
    "facilities",
    "communication",
    "other",
)

# Used when the model's reply cannot be generated. Still gives the patient
# a reference and a route to a human, which is the part that matters.
FALLBACK_REPLY = (
    "Thank you for telling us — I'm sorry this happened. I've logged it as "
    "{reference} and passed it to our practice manager, Fiona Adeyemi, who "
    "will look into it. You can reach her on (503) 555-0142 or at "
    "fiona@riverbenddental.example."
)

NOT_A_COMPLAINT = (
    "Thank you — I've passed that on to the team, they'll be glad to hear it. "
    "Is there anything else I can help you with?"
)


@dataclass
class ComplaintRecord:
    """A logged complaint."""

    reference: str
    received_at: datetime
    message: str
    summary: str
    category: str
    severity: str
    escalated: bool
    escalation_reasons: list[str] = field(default_factory=list)
    assessment: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "received_at": self.received_at.isoformat(),
            "message": self.message,
            "summary": self.summary,
            "category": self.category,
            "severity": self.severity,
            "escalated": self.escalated,
            "escalation_reasons": self.escalation_reasons,
        }


class ComplaintStore:
    """In-memory complaint log.

    Used in tests, and as the emergency fallback when the persistent
    store cannot be written. The interface — `record` and `all` — is
    shared with `SQLiteComplaintStore`, so the agent does not care which
    it has.
    """

    def __init__(self) -> None:
        self._records: list[ComplaintRecord] = []

    def record(self, complaint: ComplaintRecord) -> None:
        self._records.append(complaint)
        logger.info(
            "Complaint %s logged: %s/%s escalated=%s",
            complaint.reference, complaint.category, complaint.severity,
            complaint.escalated,
        )

    def all(self) -> list[ComplaintRecord]:
        return list(self._records)


_store: Any = None

# Complaints that could not be persisted. Held in memory so they are at
# least visible to this process, and logged at error level so an
# operator is alerted. Better than losing them.
_fallback = ComplaintStore()


def get_store() -> Any:
    """Return the process-wide complaint store (SQLite-backed)."""
    global _store
    if _store is None:
        from app.integrations.analytics import SQLiteComplaintStore

        _store = SQLiteComplaintStore()
    return _store


def _persist(store: Any, record: "ComplaintRecord") -> bool:
    """Write a complaint, falling back to memory if the store fails.

    Returns True if the primary store accepted it. A False return is
    recorded on the response so the failure is visible in analytics and
    logs, but the patient's reply is unaffected — they still get a
    reference number, and the complaint still exists.
    """
    try:
        store.record(record)
        return True
    except Exception as exc:
        logger.error(
            "COMPLAINT NOT PERSISTED — held in memory only. ref=%s error=%s",
            record.reference, exc,
        )
        _fallback.record(record)
        return False


def _reference() -> str:
    """Generate a patient-facing reference, e.g. 'CMP-20260920-4F2A'."""
    return f"CMP-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:4].upper()}"


def _coerce(assessment: dict[str, Any]) -> dict[str, Any]:
    """Validate the model's JSON, replacing anything unusable.

    Constrained decoding guarantees valid JSON, not sensible values. A
    severity of "extremely bad" is syntactically fine and semantically
    useless, so unrecognised values fall back to the cautious option
    rather than being trusted.
    """
    severity = str(assessment.get("severity", "")).lower().strip()
    if severity not in VALID_SEVERITIES:
        logger.warning("Unrecognised severity %r; defaulting to high", severity)
        severity = "high"

    category = str(assessment.get("category", "")).lower().strip()
    if category not in VALID_CATEGORIES:
        category = "other"

    def flag(key: str, default: bool = False) -> bool:
        value = assessment.get(key, default)
        return bool(value) if isinstance(value, bool) else default

    summary = str(assessment.get("summary") or "").strip()

    return {
        "summary": summary,
        "category": category,
        "severity": severity,
        "requires_escalation": flag("requires_escalation"),
        "patient_appears_distressed": flag("patient_appears_distressed"),
        "mentions_legal_action": flag("mentions_legal_action"),
        "mentions_harm": flag("mentions_harm"),
        # Defaults to True: treating a genuine complaint as praise is a
        # far worse error than logging a compliment as a complaint.
        "is_actually_a_complaint": flag("is_actually_a_complaint", True),
    }


def _escalation_decision(assessment: dict[str, Any]) -> tuple[bool, list[str]]:
    """Decide escalation in code, recording why.

    The model's own recommendation is one input. The rules below override
    it upward but never downward, so a model that under-rates a serious
    complaint cannot suppress the alert.
    """
    reasons: list[str] = []

    if assessment["severity"] == "critical":
        reasons.append("severity is critical")
    if assessment["severity"] == "high":
        # Every high-severity complaint reaches a human. The prompt's own
        # definition of "high" — failed treatment, a significant billing
        # error, repeated unanswered contact — describes cases a clinic
        # would always want its practice manager to see. Leaving this to
        # the model's judgement let a double-billing complaint with three
        # ignored phone calls through unescalated during testing.
        reasons.append(f"high-severity {assessment['category']} complaint")
    if assessment["mentions_harm"]:
        reasons.append("patient reports harm")
    if assessment["mentions_legal_action"]:
        reasons.append("legal or regulatory action mentioned")
    if assessment["requires_escalation"]:
        reasons.append("flagged by assessment")

    return bool(reasons), reasons


def handle_complaint(
    message: str,
    *,
    history: list[dict[str, str]] | None = None,
    store: ComplaintStore | None = None,
) -> AgentResponse:
    """Assess, log and respond to a complaint.

    Args:
        message: The patient's message, verbatim.
        history: Prior turns. Currently unused by the assessment, which
            deliberately judges the complaint as stated.
        store: Where to log. Injectable for tests.

    Returns:
        An `AgentResponse` whose metadata carries the reference, severity
        and escalation decision for the analytics layer.
    """
    log = store or get_store()

    try:
        raw = complete_json(render("complaint_assess", message=message))
    except LLMError as exc:
        logger.error("Complaint assessment failed: %s", exc)
        # A complaint that cannot be assessed is still a complaint, and
        # losing it is unacceptable. It is logged at high severity and
        # escalated, so a human sees it even though the model did not.
        reference = _reference()
        stored = _persist(
            log,
            ComplaintRecord(
                reference=reference,
                received_at=datetime.now(),
                message=message,
                summary="(automatic assessment unavailable)",
                category="other",
                severity="high",
                escalated=True,
                escalation_reasons=["assessment failed; escalated by default"],
            ),
        )
        return AgentResponse(
            answer=FALLBACK_REPLY.format(reference=reference),
            agent=AGENT_NAME,
            success=False,
            metadata={
                "reference": reference,
                "severity": "high",
                "escalated": True,
                "error_detail": str(exc),
                "assessment_failed": True,
                "persisted": stored,
            },
        )

    assessment = _coerce(raw)

    if not assessment["is_actually_a_complaint"]:
        logger.info("Message routed to complaint agent was not a complaint")
        return AgentResponse(
            answer=NOT_A_COMPLAINT,
            agent=AGENT_NAME,
            success=True,
            metadata={"logged": False, "reason": "not_a_complaint", **assessment},
        )

    escalated, reasons = _escalation_decision(assessment)
    reference = _reference()

    record = ComplaintRecord(
        reference=reference,
        received_at=datetime.now(),
        message=message,
        summary=assessment["summary"] or message[:160],
        category=assessment["category"],
        severity=assessment["severity"],
        escalated=escalated,
        escalation_reasons=reasons,
        assessment=assessment,
    )
    # Logged before the reply is generated. If reply generation fails the
    # complaint must still exist; the record is the part with legal and
    # regulatory weight, the reply is courtesy.
    stored = _persist(log, record)

    try:
        reply = complete(
            render(
                "complaint_reply",
                message=message,
                category=assessment["category"],
                severity=assessment["severity"],
                reference=reference,
                escalated="yes" if escalated else "no",
            )
        )
    except LLMError as exc:
        logger.error("Complaint reply generation failed: %s", exc)
        reply = FALLBACK_REPLY.format(reference=reference)

    return AgentResponse(
        answer=reply,
        agent=AGENT_NAME,
        success=True,
        metadata={
            "logged": True,
            "reference": reference,
            "category": assessment["category"],
            "severity": assessment["severity"],
            "escalated": escalated,
            "escalation_reasons": reasons,
            "mentions_harm": assessment["mentions_harm"],
            "mentions_legal_action": assessment["mentions_legal_action"],
            "persisted": stored,
        },
    )
