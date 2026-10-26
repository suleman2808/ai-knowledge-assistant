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
from app.rag.catalogue import rows_named_in
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


# ---------------------------------------------------------------------------
# A patient asking what their own result means
# ---------------------------------------------------------------------------

# The laboratory's policy is that staff do not interpret individual
# results, and the model was asked to say so in the prompt. It did, and
# then added "9.2 g/dL is not below the critical threshold (7 g/dL)" -
# every time, through two rewordings of the instruction. That sentence is
# an interpretation of the patient's number in all but name, and a patient
# hears it as "you're fine". A rule this important is not left to a
# prompt: it is decided here, and the model is not asked.
RESULT_INTERPRETATION_ANSWER = (
    "I can't tell you whether that result is a problem for you. Our staff "
    "are not able to interpret individual results or advise on what they "
    "mean, so please take your report to the doctor who requested the test, "
    "or to any general practitioner, for an assessment. If you'd like to "
    "talk it through with our team first, you can call us on (503) 555-0142."
)

_RESULT_CUES = (
    "haemoglobin", "hemoglobin", "glucose", "cholesterol", "hba1c", "tsh",
    "creatinine", "platelet", "wbc", "rbc", "ferritin", "vitamin", "bilirubin",
    "triglyceride", "ldl", "hdl", "sugar", "level", "levels", "result",
    "results", "count", "reading", "value", "report", "g/dl", "mg/dl",
    "mmol", "iu/", "ng/ml", "%",
)
_INTERPRETIVE_WORDS = (
    "bad", "normal", "abnormal", "high", "low", "concerning", "worried",
    "worry", "serious", "dangerous", "okay", "ok", "fine", "healthy", "good",
    "problem", "worrying", "too high", "too low", "mean", "means",
)


def asks_to_interpret_a_result(message: str) -> bool:
    """Whether the patient has given a value and asked what it means.

    Needs all three of a number, something that marks it as a result (an
    analyte, a unit, "my result"), and a judgement being asked for. Any one
    alone is ordinary: "do I need to fast for 12 hours", "how much is a
    glucose test", "is 8am ok".
    """
    text = " ".join(message.lower().split())
    if not any(ch.isdigit() for ch in text):
        return False
    words = set(text.replace("?", " ").replace(",", " ").replace(".", " ").split())
    judged = any(w in words for w in _INTERPRETIVE_WORDS if " " not in w) or any(
        phrase in text for phrase in _INTERPRETIVE_WORDS if " " in phrase
    )
    cued = any(cue in text for cue in _RESULT_CUES)
    return judged and cued


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
    if asks_to_interpret_a_result(question):
        # Retrieved only so the staff dashboard can show which section the
        # policy came from; what is said does not depend on it.
        policy = retrieve("staff do not interpret individual results")
        return AgentResponse(
            answer=RESULT_INTERPRETATION_ANSWER,
            agent=AGENT_NAME,
            success=True,
            sources=policy.sources if policy.grounded else [],
            metadata={
                "grounded": True,
                "handled": "result_interpretation",
                "retrieval_status": policy.status.value,
                "retrieved": [c.breadcrumb for c in policy.chunks],
            },
        )

    search_query, rewritten = _standalone_question(question, history)
    result = retrieve(search_query)

    # A test named in the question is looked up in the catalogue by name,
    # and its row is put in front of the model whatever retrieval found.
    # "How much is a lipid profile?" never retrieved the table that holds
    # the answer, so the model, correctly, refused to answer it.
    result = _with_catalogue_rows(result, search_query)

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

    # A question with two parts is answered one part at a time. Asked "what
    # are your opening hours and where are your branches", one prompt over
    # one blended context answered the hours and dropped the branches
    # without a word; asked again after a wording change, it answered the
    # branches and dropped the hours. A model told to answer both does not
    # reliably do so, and the failure is silent. Each part gets its own
    # question and its own context, and a part that cannot be answered is
    # named as such rather than left out.
    if len(_split_question(search_query)) >= 2:
        in_parts = _answer_in_parts(search_query, question, history, result)
        if in_parts is not None:
            return in_parts

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
BUSINESS_NAME = "Riverbend Diagnostics"


def _anchor(part: str) -> str:
    """Name the business in one half of a compound question.

    "Where are your branches" scored 0.28 and was rejected as unrelated,
    while the same question with the company name in it scores 0.62 and
    finds the right section. A patient never says the company's name -
    "your" means it - but every document is about it, so a fragment cut
    out of a longer question loses the one word that located it.

    Only applied to the parts of a compound question, never to a single
    one: that would lift the score of every off-topic question as well,
    and the whole point of the threshold is to keep those out.
    """
    return part if BUSINESS_NAME.lower() in part.lower() else f"{part} {BUSINESS_NAME}"
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
        outcome = retrieve(_anchor(part))
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


def _with_catalogue_rows(result: RetrievalResult, query: str) -> RetrievalResult:
    """Put the catalogue rows for any test the question names ahead of `result`.

    Counts as grounded even when retrieval found nothing: an exact match on
    a test's name in the catalogue is better evidence than a similarity
    score, and it is the same document the answer would come from.
    """
    rows = rows_named_in(query)
    if not rows:
        return result

    have = {c.chunk_id for c in result.chunks}
    row_chunks = [r.as_chunk() for r in rows if r.as_chunk().chunk_id not in have]
    return RetrievalResult(
        query=result.query,
        status=RetrievalStatus.OK,
        chunks=[*row_chunks, *result.chunks],
        best_score=max(result.best_score, 1.0),
    )


def _with_each_part(result: RetrievalResult, search_query: str) -> RetrievalResult:
    """Add the chunks each half of a compound question retrieves to `result`.

    A single question is returned untouched, at no cost: nothing is
    retrieved a second time unless the question genuinely has parts.
    """
    split = _retrieve_each_part(search_query)
    if split is None:
        return result

    have = {f"{c.source}#{c.breadcrumb}" for c in result.chunks}
    extra = [c for c in split.chunks if f"{c.source}#{c.breadcrumb}" not in have]
    if not extra:
        return result

    return RetrievalResult(
        query=result.query,
        status=RetrievalStatus.OK,
        chunks=[*result.chunks, *extra],
        best_score=max(result.best_score, split.best_score),
    )


_PRONOUNS = frozenset({"it", "that", "this", "them", "those", "these", "one"})


PART_MAX_CHUNKS = 6
MAX_QUOTED_PART = 80


def _retrieve_part(part: str) -> RetrievalResult:
    """Retrieve for one part of a compound question, both ways.

    Naming the business helps some fragments and hurts others, so neither
    is chosen. "Where are your branches" scores 0.28 bare and is rejected,
    0.62 anchored and finds the right section. "What are your hours" is
    the reverse: anchored, the company introduction outranks the section
    that actually lists the hours, and the model - shown an introduction -
    correctly says it cannot tell the patient the hours. The bare query
    finds the hours section. Both are run and the results combined, the
    plain one first.
    """
    plain = retrieve(part)
    anchored = retrieve(_anchor(part))

    chunks, have = [], set()
    for chunk in [*(plain.chunks if plain.grounded else []),
                  *(anchored.chunks if anchored.grounded else [])]:
        if chunk.chunk_id not in have:
            have.add(chunk.chunk_id)
            chunks.append(chunk)

    combined = RetrievalResult(
        query=part,
        status=RetrievalStatus.OK if chunks else anchored.status,
        chunks=chunks[:PART_MAX_CHUNKS],
        best_score=max(plain.best_score, anchored.best_score),
    )
    return _with_catalogue_rows(combined, part)


def _part_question(part: str, previous: str) -> str:
    """The question put to the model for one part of a compound question.

    Asked bare, because the model checks whether the material answers "the
    question". An earlier version appended the patient's whole message to
    each part for context, and then the model, seeing "...and do you do
    allergy testing?" in it, replied INSUFFICIENT_CONTEXT to "what are your
    hours" - the half it could have answered - because the other half was
    not covered.

    The one thing a part can lack is its subject: "how much does it cost"
    means nothing alone. Only when it contains a pronoun is the previous
    part given, as the thing the pronoun refers to.
    """
    words = {w.strip("?,.!").lower() for w in part.split()}
    if previous and words & _PRONOUNS:
        return f"{part} (here 'it' or 'that' means the subject of: {previous})"
    return part


def _without_repeats(answers: list[str]) -> list[str]:
    """Drop an answer that says what an earlier one already said.

    Each part is answered on a context that includes everything the whole
    question retrieved, so a part whose question overlaps another's - "do
    I need to fast for a lipid profile, and how much does it cost" - can be
    answered completely twice, once per part. Said twice, it reads as a
    glitch.
    """
    from difflib import SequenceMatcher

    kept: list[str] = []
    for answer in answers:
        normalised = " ".join(answer.lower().split())
        if any(SequenceMatcher(None, normalised, " ".join(k.lower().split())).ratio() > 0.8
               for k in kept):
            continue
        kept.append(answer)
    return kept


def _answer_in_parts(
    search_query: str,
    question: str,
    history: list[dict[str, str]] | None,
    whole: RetrievalResult,
) -> AgentResponse | None:
    """Answer a question with several parts one part at a time.

    Each part is asked as its own question, on its own context: the chunks
    retrieved for that part, plus whatever the whole question retrieved, so
    that "how much does it cost" can still see the test the other half
    named. The answers are joined in the order asked.

    A part that cannot be answered is named in the reply. Leaving it out
    would be the silent drop this function exists to prevent: the patient
    would read the answer to the other half as the whole answer.

    Returns None when the question turns out not to have two parts worth
    answering, or when none could be answered, and the caller then answers
    the question as one.
    """
    from app.graph.build import wants_a_person

    parts = [p for p in _split_question(search_query) if not wants_a_person(p)]
    if len(parts) < 2:
        return None

    answers: list[str] = []
    unanswered: list[str] = []
    sources: list[dict[str, str | float]] = []
    seen: set[str] = set()
    best = 0.0

    previous = ""
    for index, part in enumerate(parts):
        previous = parts[index - 1] if index else ""
        own = _retrieve_part(part)
        best = max(best, own.best_score)

        chunks, have = [], set()
        for chunk in [*(own.chunks if own.grounded else []), *whole.chunks]:
            if chunk.chunk_id not in have:
                have.add(chunk.chunk_id)
                chunks.append(chunk)
        if not chunks:
            unanswered.append(part)
            continue

        context = "\n\n".join(c.cite(i) for i, c in enumerate(chunks, start=1))
        try:
            reply = complete_verbose(
                render(
                    "inquiry_user",
                    context=context,
                    history=_format_history(history),
                    question=_part_question(part, previous),
                ),
                system=render("inquiry_system"),
            )
        except LLMError as exc:
            logger.error("Compound part failed: %s", exc)
            unanswered.append(part)
            continue

        if _is_refusal(reply.text):
            unanswered.append(part)
            continue

        answers.append(reply.text.strip())
        for chunk in chunks:
            key = f"{chunk.source}#{chunk.breadcrumb}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {"breadcrumb": chunk.breadcrumb, "source": chunk.source,
                     "score": round(chunk.score, 3)}
                )

    if not answers:
        return None

    answers = _without_repeats(answers)

    for part in unanswered:
        # Quoted back so the patient can see which half was meant, but
        # capped: a part is whatever the patient typed, and echoing an
        # arbitrarily long or hostile string verbatim helps nobody.
        shown = part if len(part) <= MAX_QUOTED_PART else part[:MAX_QUOTED_PART].rstrip() + "..."
        answers.append(
            f"I can't find an answer to \"{shown}\" in the laboratory's information."
        )

    logger.info(
        "Compound question: %d of %d parts answered: %s",
        len(parts) - len(unanswered), len(parts), question[:80],
    )
    final = "\n\n".join(answers)

    # Released once, finished. Streaming each part as it was written meant
    # the "I can't find..." note and the removal of a repeated answer both
    # happened after the reader had already seen the text, so a sentence
    # would type out twice and then collapse. The page paces the reveal
    # itself, so nothing is lost by sending the whole answer at once.
    sink = active_sink()
    if sink:
        sink(final)

    return AgentResponse(
        answer="\n\n".join(answers),
        agent=AGENT_NAME,
        success=True,
        sources=sources,
        metadata={
            "grounded": True,
            "compound_parts": len(parts),
            "compound_unanswered": unanswered,
            "search_query": search_query,
            "retrieval_status": RetrievalStatus.OK.value,
            "best_score": round(best, 3),
        },
    )
