"""Inquiry Agent — answers questions strictly from retrieved documents.

The agent is a pipeline with two independent safeguards against answering
something the clinic's documents do not actually say:

1. **The retrieval threshold** (in `app.rag.retriever`) drops matches that
   are not close enough to be worth considering.
2. **The grounding check** in the prompt. The model is told to reply with
   the single token `INSUFFICIENT_CONTEXT` when the retrieved material is
   topically related but does not answer the question. The agent converts
   that into a polite refusal with a route to a human.

The second safeguard exists because the first is provably insufficient.
Measured on this corpus, "what time does the cinema open" scores 0.422 and
retrieves the clinic's opening-hours table, out-scoring legitimate
questions about parking (0.380) and billing (0.384). No threshold
separates those cases, so the model is asked to make the final judgement
with the evidence in front of it.
"""

from __future__ import annotations

import logging

from app.agents.base import AgentResponse, failure
from app.config import settings
from app.llm import LLMError, complete, complete_verbose
from app.prompts import render
from app.rag.retriever import RetrievalResult, RetrievalStatus, retrieve

logger = logging.getLogger(__name__)

AGENT_NAME = "inquiry"

# The exact string the model emits when the context does not answer the
# question. Checked case-insensitively and tolerant of surrounding
# punctuation, because models occasionally wrap it in quotes or a full
# stop despite instructions.
INSUFFICIENT = "INSUFFICIENT_CONTEXT"

# Shown when we have nothing solid to answer from. Deliberately specific:
# it names the limitation, offers a real alternative, and does not
# pretend the question was unreasonable.
NO_ANSWER = (
    "I don't have that in the clinic's information, so I'd rather not guess. "
    "Please call us on (503) 555-0142 and the team can answer properly — or "
    "ask me something else about our services, hours, policies or treatments."
)

NOT_INGESTED = (
    "My knowledge base hasn't been set up yet, so I can't answer questions "
    "about the clinic. Please call (503) 555-0142 for help."
)

SEARCH_DOWN = (
    "I can't search the clinic's information right now. Please call "
    "(503) 555-0142 and someone will help you straight away."
)


def _is_refusal(text: str) -> bool:
    """Detect the model's insufficient-context signal.

    Tolerant by design. A model that returns `"INSUFFICIENT_CONTEXT."` has
    followed the instruction in substance, and treating that as a real
    answer would put the literal token in front of a patient.
    """
    normalised = text.strip().strip("\"'`*. \n").upper()
    return normalised == INSUFFICIENT or normalised.startswith(INSUFFICIENT)


def _format_history(history: list[dict[str, str]] | None) -> str:
    """Render recent turns for the prompt.

    Only the last few turns are included. Long histories dilute the
    retrieved context and cost tokens without improving answers, and this
    agent's job is answering the current question, not maintaining a
    relationship across fifty messages.
    """
    if not history:
        return "(this is the first message)"

    lines = []
    for turn in history[-4:]:
        role = "Patient" if turn.get("role") == "user" else "Assistant"
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(this is the first message)"


def _standalone_question(
    question: str, history: list[dict[str, str]] | None
) -> tuple[str, bool]:
    """Rewrite a follow-up into a question that can be searched on its own.

    Retrieval sees only the text it is given. "Is that for one surface?"
    retrieves nothing useful, because the word that gives it meaning —
    "filling" — is in the previous turn. Found by the analytics report:
    both follow-up questions in the seeded data were logged as knowledge
    gaps despite the documents answering them.

    Only runs when there is history, so a first message costs no extra
    call. Uses the small model: this is reformulation, not reasoning.

    Returns:
        `(query_for_retrieval, was_rewritten)`. Falls back to the original
        on any failure — a rewrite is an improvement, never a dependency.
    """
    if not history:
        return question, False

    try:
        rewritten = complete(
            render(
                "inquiry_rewrite",
                history=_format_history(history),
                message=question,
            ),
            model=settings.router_model,
            reasoning_effort=settings.router_reasoning_effort,
            max_tokens=300,
            temperature=0.0,
        )
    except LLMError as exc:
        logger.warning("Query rewrite failed, searching the original: %s", exc)
        return question, False

    rewritten = rewritten.strip().strip("\"'").splitlines()[0].strip() if rewritten else ""
    # Guard against the model answering instead of rewriting, or padding
    # the query with invented detail. A standalone question is rarely more
    # than a few times longer than the follow-up it came from.
    if not rewritten or len(rewritten) > max(200, len(question) * 6):
        return question, False
    return rewritten, rewritten.lower() != question.lower()


def _refusal_for(result: RetrievalResult) -> AgentResponse:
    """Map a non-OK retrieval status onto a user-facing response.

    Each status gets different wording because each needs a different
    action from whoever reads the logs: an empty index is an operator
    error, an unavailable store is an outage, and a low-confidence match
    is the system working correctly.
    """
    if result.status is RetrievalStatus.EMPTY_INDEX:
        answer, success = NOT_INGESTED, False
    elif result.status is RetrievalStatus.UNAVAILABLE:
        answer, success = SEARCH_DOWN, False
    else:
        # LOW_CONFIDENCE or NO_MATCH: the system behaved correctly and
        # simply does not know, so this is not a failure.
        answer, success = NO_ANSWER, True

    return AgentResponse(
        answer=answer,
        agent=AGENT_NAME,
        success=success,
        metadata={
            "grounded": False,
            "retrieval_status": result.status.value,
            "best_score": round(result.best_score, 3),
            "retrieval_detail": result.detail,
        },
    )


def answer_inquiry(
    question: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> AgentResponse:
    """Answer a question using only the clinic's documents.

    Args:
        question: The patient's question, verbatim.
        history: Prior conversation turns as `{"role", "content"}` dicts.

    Returns:
        An `AgentResponse`. `metadata["grounded"]` records whether the
        answer came from retrieved documents; when it is False the answer
        is a refusal, never an ungrounded guess.
    """
    search_query, rewritten = _standalone_question(question, history)
    result = retrieve(search_query)

    if not result.grounded:
        logger.info(
            "Inquiry not grounded (%s, best=%.3f): %s",
            result.status.value, result.best_score, search_query[:80],
        )
        response = _refusal_for(result)
        if rewritten:
            response.metadata["search_query"] = search_query
        return response

    prompt = render(
        "inquiry_user",
        context=result.as_context(),
        history=_format_history(history),
        question=question,
    )

    try:
        response = complete_verbose(prompt, system=render("inquiry_system"))
    except LLMError as exc:
        logger.error("Inquiry LLM call failed: %s", exc)
        return failure(
            AGENT_NAME,
            "llm_unavailable",
            str(exc),
            retrieval_status=result.status.value,
            best_score=round(result.best_score, 3),
        )

    if _is_refusal(response.text):
        # Retrieval found something above the threshold, but the model
        # judged it not to answer this question. This is the safeguard
        # working, so it is logged at info rather than as an error.
        logger.info(
            "Model declined as ungrounded despite score %.3f: %s",
            result.best_score, question[:80],
        )
        return AgentResponse(
            answer=NO_ANSWER,
            agent=AGENT_NAME,
            success=True,
            metadata={
                "grounded": False,
                "refused_by": "model",
                "search_query": search_query if rewritten else None,
                "retrieval_status": result.status.value,
                "best_score": round(result.best_score, 3),
                "retrieved": [c.breadcrumb for c in result.chunks],
                "latency_ms": response.latency_ms,
            },
        )

    return AgentResponse(
        answer=response.text,
        agent=AGENT_NAME,
        success=True,
        sources=result.sources,
        metadata={
            "grounded": True,
            "search_query": search_query if rewritten else None,
            "retrieval_status": result.status.value,
            "best_score": round(result.best_score, 3),
            "retrieved": [c.breadcrumb for c in result.chunks],
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "model": response.model,
        },
    )
