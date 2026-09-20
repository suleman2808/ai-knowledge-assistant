"""Query the knowledge base and return grounded context.

This layer sits between the vector store and the Inquiry Agent. Its job is
not simply "return the nearest chunks" — it is to answer the question *do
we actually know this?* honestly.

Three principles shape it:

1. **Never raise for an expected outcome.** An empty knowledge base, a
   query with no good match, or a vector store that will not open are all
   ordinary conditions in production. Each returns a `RetrievalResult`
   carrying a status the agent can act on, rather than an exception the
   API layer has to translate into an apology.
2. **A weak match is not a match.** Vector search always returns its `k`
   nearest neighbours, however far away they are. Ask a dental clinic's
   knowledge base about car insurance and it will cheerfully return the
   dental insurance page. Without a score threshold, the agent would
   ground a confident wrong answer in that chunk. The threshold is the
   single most important line of defence against hallucination here.
3. **Context is formatted for citation.** Chunks are numbered and labelled
   with their heading path, so the model can reference `[2]` and we can
   map that back to a real section of a real document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.rag.embeddings import EmbeddingError, embed_query
from app.rag.store import VectorStoreError, get_collection

logger = logging.getLogger(__name__)

# At most this many chunks from any single document section. One long
# section can otherwise occupy the entire context window and crowd out a
# different document that answers the question better.
MAX_PER_SECTION = 2


class RetrievalStatus(str, Enum):
    """Why a retrieval returned what it did.

    The agent branches on this, so the distinctions are behavioural rather
    than cosmetic:

    - `OK` — usable context was found; answer from it.
    - `LOW_CONFIDENCE` — matches exist but all score below the threshold.
      Say we are not certain and offer to connect the user to a human.
    - `NO_MATCH` — the knowledge base is reachable but returned nothing.
    - `EMPTY_INDEX` — nothing has been ingested. An operator error, not a
      user one, and worth reporting differently.
    - `UNAVAILABLE` — the store or embedding model failed. A genuine
      outage.
    """

    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    NO_MATCH = "no_match"
    EMPTY_INDEX = "empty_index"
    UNAVAILABLE = "unavailable"


@dataclass
class RetrievedChunk:
    """A single search hit."""

    chunk_id: str
    text: str
    score: float
    """Cosine similarity in roughly 0-1. Higher is closer."""

    breadcrumb: str
    source: str
    doc_title: str = ""

    def cite(self, index: int) -> str:
        """Render as a numbered block for the prompt."""
        return f"[{index}] {self.breadcrumb}\n{self.text}"


@dataclass
class RetrievalResult:
    """Everything the agent needs to decide how to answer."""

    query: str
    status: RetrievalStatus
    chunks: list[RetrievedChunk] = field(default_factory=list)
    best_score: float = 0.0
    detail: str = ""
    """Operator-facing explanation. Never shown to the end user verbatim."""

    @property
    def grounded(self) -> bool:
        """True when there is context solid enough to answer from."""
        return self.status is RetrievalStatus.OK and bool(self.chunks)

    @property
    def sources(self) -> list[dict[str, str | float]]:
        """Citation list for the API response and the UI."""
        return [
            {
                "breadcrumb": c.breadcrumb,
                "source": c.source,
                "score": round(c.score, 3),
            }
            for c in self.chunks
        ]

    def as_context(self) -> str:
        """Format the chunks as numbered context for a prompt."""
        if not self.chunks:
            return ""
        return "\n\n".join(c.cite(i) for i, c in enumerate(self.chunks, start=1))


def _diversify(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Cap how many chunks any one section may contribute.

    Long sections are split into several adjacent chunks with overlapping
    text. Without a cap those near-duplicates can fill every slot, so the
    model sees one section three times instead of three perspectives.
    """
    seen: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        count = seen.get(chunk.breadcrumb, 0)
        if count >= MAX_PER_SECTION:
            continue
        seen[chunk.breadcrumb] = count + 1
        kept.append(chunk)
    return kept


def retrieve(
    query: str,
    *,
    top_k: int | None = None,
    min_score: float | None = None,
    collection: object | None = None,
) -> RetrievalResult:
    """Search the knowledge base for context relevant to `query`.

    Args:
        query: The user's question, verbatim.
        top_k: Number of chunks to return. Defaults to `RETRIEVAL_TOP_K`.
        min_score: Similarity below which a match is treated as no match.
            Defaults to `RETRIEVAL_MIN_SCORE`.
        collection: An open collection to search. Injectable so tests can
            run against a fake without a database.

    Returns:
        A `RetrievalResult`. This function does not raise for expected
        failures; inspect `status` instead.
    """
    query = (query or "").strip()
    if not query:
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.NO_MATCH,
            detail="Empty query.",
        )

    k = top_k or settings.retrieval_top_k
    threshold = settings.retrieval_min_score if min_score is None else min_score

    # Over-fetch so that diversity capping and threshold filtering have
    # candidates to work with rather than shrinking the result below k.
    fetch_k = max(k * 3, k + 4)

    try:
        target = collection if collection is not None else get_collection(create=False)
        vector = embed_query(query)
        raw = target.query(  # type: ignore[attr-defined]
            query_embeddings=[vector],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
    except VectorStoreError as exc:
        logger.warning("Knowledge base unavailable: %s", exc)
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.EMPTY_INDEX,
            detail=str(exc),
        )
    except EmbeddingError as exc:
        logger.error("Embedding failed: %s", exc)
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.UNAVAILABLE,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("Unexpected retrieval failure")
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.UNAVAILABLE,
            detail=f"{type(exc).__name__}: {exc}",
        )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]

    if not documents:
        # Distinguish "nothing was ever ingested" from "nothing matched".
        # The first is an operator problem and needs a different message.
        try:
            indexed = int(target.count())  # type: ignore[attr-defined]
        except Exception:
            indexed = -1
        if indexed == 0:
            return RetrievalResult(
                query=query,
                status=RetrievalStatus.EMPTY_INDEX,
                detail="The collection exists but holds no chunks. Run `python -m scripts.ingest`.",
            )
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.NO_MATCH,
            detail="The knowledge base returned no rows for this query.",
        )

    candidates: list[RetrievedChunk] = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata = metadata or {}
        # Chroma reports cosine *distance*; similarity is 1 - distance.
        # Clamped because floating-point error can nudge it fractionally
        # outside the valid range, and a score of 1.0000001 in a log is a
        # distraction nobody needs.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        candidates.append(
            RetrievedChunk(
                chunk_id=str(chunk_id),
                text=str(text),
                score=score,
                breadcrumb=str(metadata.get("breadcrumb") or metadata.get("source", "")),
                source=str(metadata.get("source", "")),
                doc_title=str(metadata.get("doc_title", "")),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    best_score = candidates[0].score

    above = [c for c in candidates if c.score >= threshold]
    if not above:
        # Matches exist, but none close enough to trust. The nearest ones
        # are still attached so an operator can see what nearly matched,
        # while `status` keeps the agent from answering from them.
        return RetrievalResult(
            query=query,
            status=RetrievalStatus.LOW_CONFIDENCE,
            chunks=[],
            best_score=best_score,
            detail=(
                f"Best match scored {best_score:.3f}, below the "
                f"{threshold:.2f} threshold. Nearest section: "
                f"{candidates[0].breadcrumb!r}."
            ),
        )

    selected = _diversify(above)[:k]
    return RetrievalResult(
        query=query,
        status=RetrievalStatus.OK,
        chunks=selected,
        best_score=best_score,
        detail=f"{len(selected)} chunk(s) at or above {threshold:.2f}.",
    )
