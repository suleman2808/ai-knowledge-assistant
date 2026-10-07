"""Inquiry Agent — answers questions strictly from retrieved documents.

The agent is a pipeline with two independent safeguards against answering
something the laboratory's documents do not actually say:

1. **The retrieval threshold** (in `app.rag.retriever`) drops matches that
   are not close enough to be worth considering.
2. **The grounding check** in the prompt. The model is told to reply with
   the single token `INSUFFICIENT_CONTEXT` when the retrieved material is
   topically related but does not answer the question. The agent converts
   that into a polite refusal with a route to a human.

The second safeguard exists because the first is provably insufficient.
Measured on this corpus, "what time does the cinema open" scores 0.422 and
retrieves the laboratory's opening-hours table, out-scoring legitimate
questions about parking (0.380) and billing (0.384). No threshold
separates those cases, so the model is asked to make the final judgement
with the evidence in front of it.
"""

from __future__ import annotations

import logging
import re

from app.agents.base import AgentResponse, failure, guarded_tokens
from app.config import settings
from app.llm import LLMError, active_sink, complete, complete_verbose
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
# Said when retrieval finds nothing solid. It covers two situations -
# a fair question about this laboratory that the documents happen not
# to answer, and a question about something else that the router sent
# here anyway - so it cannot presume which. An earlier draft said "that's
# a fair question... something that affects your care", which reads
# oddly in reply to someone asking about a cinema.
NO_ANSWER = (
    "I can't find that in the laboratory's information, so I'd rather not"
    " guess at it.\n\n"
    "The team on (503) 555-0142 can help with anything I can't. In the"
    " meantime I'm happy to answer questions about our tests and prices,"
    " how to prepare for an appointment, or when results are ready - and I"
    " can book you in whenever suits you."
)

NOT_INGESTED = (
    "My knowledge base hasn't been set up yet, so I can't answer questions "
    "about the laboratory. Please call (503) 555-0142 for help."
)

SEARCH_DOWN = (
    "I can't search the laboratory's information right now. Please call "
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
    """Answer a question using only the laboratory's documents.

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
        # A compound question can also embed to a point between its two
        # halves and clear neither threshold.
        split = _retrieve_each_part(search_query)
        if split is not None:
            result = split

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
        # Streamed, so the reader sees the answer being written. The guard
        # holds the opening characters back until the sentinel is ruled
        # out: watching INSUFFICIENT_CONTEXT appear and then be replaced
        # would be worse than a moment's wait.
        with guarded_tokens(INSUFFICIENT):
            response = complete_verbose(
                prompt, system=render("inquiry_system"), stream=True
            )
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
        # Before accepting the refusal: a compound question is the one
        # case where the model is right that the context cannot answer
        # *the question* while the documents can answer both halves of
        # it. Retrieving per part and asking once more is cheap, and it
        # happens only on a path that was about to decline anyway.
        retried = _retry_as_parts(search_query, question, history, result)
        if retried is not None:
            return retried

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


# A compound question is two questions sharing a sentence. These are the
# joins that separate them; "and" inside a noun phrase ("ferritin and
# iron studies") is why the parts are length-checked before use.
_PART_SPLIT = re.compile(
    r"\s*(?:\?+|;|\band\s+also\b|,\s*\band\b|\band\b|\balso\b)\s*",
    re.IGNORECASE,
)
MIN_PART_WORDS = 3
MAX_PARTS = 3


def _split_question(question: str) -> list[str]:
    """Break a compound question into the questions it is made of."""
    parts = [p.strip(" ?.,") for p in _PART_SPLIT.split(question) if p and p.strip()]
    parts = [p for p in parts if len(p.split()) >= MIN_PART_WORDS]
    return parts[:MAX_PARTS] if len(parts) > 1 else []


def _retrieve_each_part(question: str) -> RetrievalResult | None:
    """Retrieve for each half of a compound question and merge the results.

    Returns None when the question is not compound, or when the parts do
    no better than the whole - in which case the caller keeps the
    original result and declines, which is the right outcome.
    """
    parts = _split_question(question)
    if not parts:
        return None

    chunks: list[Any] = []
    seen: set[str] = set()
    best = 0.0
    grounded_parts = 0

    for part in parts:
        outcome = retrieve(part)
        best = max(best, outcome.best_score)
        if not outcome.grounded:
            continue
        grounded_parts += 1
        for chunk in outcome.chunks:
            key = f"{chunk.source}#{chunk.breadcrumb}"
            if key not in seen:
                seen.add(key)
                chunks.append(chunk)

    # One half answerable is enough to be worth replying to: the model is
    # told to answer what the documents cover and say what they do not.
    if not grounded_parts or not chunks:
        return None

    logger.info(
        "Compound question answered from %d of %d parts: %s",
        grounded_parts, len(parts), question[:80],
    )
    return RetrievalResult(
        query=question,
        status=RetrievalStatus.OK,
        chunks=chunks,
        best_score=best,
    )


def _retry_as_parts(
    search_query: str,
    question: str,
    history: list[dict[str, str]] | None,
    first: RetrievalResult,
) -> AgentResponse | None:
    """Answer a compound question one half at a time.

    The failure this fixes: asked "what are your opening hours and where
    are your branches", retrieval finds the hours and the model then
    refuses, correctly, because the context does not answer the whole
    question. The patient gets nothing, though half the answer was
    sitting right there.

    Each half is treated as its own question - retrieved for, and
    answered on its own context - and the halves that can be answered are
    joined. A half that cannot be answered is simply left out: the reply
    says what the documents cover and nothing else, which is the same
    promise the single-question path makes.

    Returns None when the question is not compound or no half could be
    answered, and the caller then declines as it would have.
    """
    parts = _split_question(search_query)
    if not parts:
        return None

    from app.graph.build import wants_a_person

    parts = [part for part in parts if not wants_a_person(part)]
    if not parts:
        return None

    answers: list[str] = []
    sources: list[dict[str, str | float]] = []
    seen: set[str] = set()
    best = 0.0

    for part in parts:
        outcome = retrieve(part)
        best = max(best, outcome.best_score)
        if not outcome.grounded:
            continue

        # Streamed like any other answer. The guard matters more here
        # than anywhere: a part the documents cannot answer refuses, and
        # that refusal must not appear on screen between two halves that
        # did answer.
        if answers:
            sink = active_sink()
            if sink:
                sink("\n\n")

        try:
            with guarded_tokens(INSUFFICIENT):
                reply = complete_verbose(
                    render(
                        "inquiry_user",
                        context=outcome.as_context(),
                        history=_format_history(history),
                        question=part,
                    ),
                    system=render("inquiry_system"),
                    stream=True,
                )
        except LLMError as exc:
            logger.error("Compound part failed: %s", exc)
            continue

        if _is_refusal(reply.text):
            continue

        answers.append(reply.text.strip())
        for source in outcome.sources:
            key = f"{source['source']}#{source['breadcrumb']}"
            if key not in seen:
                seen.add(key)
                sources.append(source)

    if not answers:
        return None

    logger.info(
        "Compound question answered in %d of %d parts: %s",
        len(answers), len(parts), question[:80],
    )
    return AgentResponse(
        answer="\n\n".join(answers),
        agent=AGENT_NAME,
        success=True,
        sources=sources,
        metadata={
            "grounded": True,
            "compound_parts": len(parts),
            "compound_answered": len(answers),
            "search_query": search_query,
            "retrieval_status": RetrievalStatus.OK.value,
            "best_score": round(best, 3),
        },
    )
