"""Request and response models for the HTTP layer.

Pydantic validates every request before a handler sees it, so malformed
input is rejected at the edge with a clear 422 rather than surfacing as a
confusing error inside an agent. These models are also what FastAPI turns
into the interactive documentation at `/docs`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# A patient message longer than this is almost certainly a paste or an
# abuse attempt. Long inputs cost tokens and can crowd retrieved context
# out of the prompt, so the limit is enforced rather than merely advised.
MAX_MESSAGE_CHARS = 2000


class ChatRequest(BaseModel):
    """One message from a patient."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_CHARS,
        description="The patient's message.",
        examples=["How much is a root canal on a molar?"],
    )
    session_id: str = Field(
        default="",
        max_length=64,
        description=(
            "Groups messages into a conversation so follow-up questions work. "
            "Omit it and the server issues one."
        ),
    )

    @field_validator("message")
    @classmethod
    def _not_only_whitespace(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message cannot be only whitespace")
        return cleaned

    @field_validator("session_id")
    @classmethod
    def _safe_session_id(cls, value: str) -> str:
        # Session ids reach logs and dictionary keys. Restricting the
        # alphabet keeps anything injectable out of both.
        cleaned = value.strip()
        if cleaned and not all(c.isalnum() or c in "-_" for c in cleaned):
            raise ValueError("session_id may contain only letters, digits, - and _")
        return cleaned


class Source(BaseModel):
    """A cited section of a clinic document."""

    breadcrumb: str = Field(description="Heading path, e.g. 'Insurance > Payment Plans'.")
    source: str = Field(description="Filename the text came from.")
    score: float = Field(description="Similarity, 0-1.")


class ChatResponse(BaseModel):
    """The assistant's reply, plus how it was produced."""

    answer: str
    session_id: str
    intent: Literal["booking", "inquiry", "complaint", "other", "unknown"]
    secondary_intent: str | None = None
    confidence: float = 0.0
    sources: list[Source] = Field(default_factory=list)
    needs_followup: bool = False
    grounded: bool | None = Field(
        default=None,
        description="Whether the answer came from retrieved documents. Null when "
                    "the question was not an inquiry.",
    )
    latency_ms: int = 0


class HealthResponse(BaseModel):
    """Whether each dependency is usable. Never fails, so it can be polled."""

    status: Literal["ok", "degraded"]
    checks: dict[str, Any]


class ErrorResponse(BaseModel):
    """What the client gets when something goes wrong.

    `detail` is safe to display: technical causes stay in the logs.
    """

    detail: str
