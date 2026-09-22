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

import numpy as np

from app.config import settings
from app.rag.embeddings import EmbeddingError, embed_query
from app.rag.lexical import BM25Index
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
    matched_terms: list[str] = field(default_factory=list)
    """Query terms matched by keyword search, when that contributed."""

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


# --------------------------------------------------------------------------
# Hybrid search
# --------------------------------------------------------------------------

# A keyword hit is trusted on its own — without a passing vector score —
# only when all three hold: it ranks near the top, it matched a rare
# term, and that match covers most of what the query was asking.
LEXICAL_TRUST_RANK = 2
# "Rare" means the term appears in at most this many chunks. "cigna"
# appears in one; "open" and "time" appear in dozens and prove nothing.
RARE_TERM_MAX_DF = 3
# The first version used only rank and rarity, and doubled false
# positives from 3/8 to 6/8 on out-of-scope questions: "renew my car
# insurance" matched "car park", "reset my email password" matched
# "benefits reset on 1 January". Measured on 14 labelled cases, IDF
# coverage separates them where vector score does not — legitimate hits
# ran 0.63-1.00, coincidental ones 0.21-0.52. This is the midpoint of that
# gap. Fitted on a small set, so treat it as calibrated, not proven.
MIN_LEXICAL_COVERAGE = 0.575
# Reciprocal Rank Fusion constant. 60 is the value from the original RRF
# paper and the near-universal default; it damps the advantage of rank 1
# over rank 2 so that neither list dominates the other.
RRF_K = 60

# Cached per collection, keyed on name and size, so ingestion (which
# changes the count) invalidates it automatically.
_lexical_cache: dict[tuple[str, int], tuple[BM25Index, dict[str, dict]]] = {}


def _lexical_index(target: object) -> tuple[BM25Index, dict[str, dict]]:
    """Build or fetch the BM25 index for a collection.

    Built from the collection's own documents, so the keyword index and
    the vector index can never disagree about what exists. Each row keeps
    its embedding, so a chunk found only by keyword can still report its
    true vector similarity.
    """
    count = int(target.count())  # type: ignore[attr-defined]
    key = (str(getattr(target, "name", id(target))), count)
    if key in _lexical_cache:
        return _lexical_cache[key]

    raw = target.get(include=["documents", "metadatas", "embeddings"])  # type: ignore[attr-defined]
    ids = raw.get("ids") or []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or [{}] * len(ids)
    embeddings = raw.get("embeddings")
    if embeddings is None:
        embeddings = [None] * len(ids)

    rows: dict[str, dict] = {}
    corpus: list[tuple[str, str]] = []
    for chunk_id, text, meta, embedding in zip(ids, documents, metadatas, embeddings):
        meta = meta or {}
        rows[str(chunk_id)] = {"text": text, "meta": meta, "embedding": embedding}
        # Headings are indexed with the body: "Cancellation Policy" is the
        # most informative text a cancellation chunk has.
        corpus.append((str(chunk_id), f"{meta.get('heading_path', '')} {text}"))

    index = BM25Index.build(corpus)
    _lexical_cache.clear()  # only ever one live collection
    _lexical_cache[key] = (index, rows)
    return index, rows


def _fuse_with_lexical(
    query: str,
    target: object,
    query_vector: list[float],
    *,
    candidates: list[RetrievedChunk],
    qualified: list[RetrievedChunk],
    fetch_k: int,
) -> list[RetrievedChunk]:
    """Merge BM25 results into the vector results.

    Two decisions, kept separate because they answer different questions:

    1. **Which chunks qualify** as relevant enough to show the model:
       anything that passed the vector threshold, plus keyword hits that
       matched a rare term near the top of the BM25 ranking.
    2. **In what order**: Reciprocal Rank Fusion over both rankings. RRF
       uses ranks rather than raw scores, which matters because cosine
       similarity and BM25 scores live on unrelated scales and cannot be
       meaningfully added.

    Falls back to the vector result untouched if the keyword index cannot
    be built — hybrid search is an improvement, not a dependency.
    """
    try:
        index, rows = _lexical_index(target)
    except Exception as exc:
        logger.warning("Keyword index unavailable, using vector results only: %s", exc)
        return qualified

    hits = index.search(query, k=fetch_k)
    if not hits:
        return qualified

    by_id = {c.chunk_id: c for c in candidates}
    chosen = {c.chunk_id: c for c in qualified}

    for rank, hit in enumerate(hits, start=1):
        trusted = (
            rank <= LEXICAL_TRUST_RANK
            and hit.rarest_df <= RARE_TERM_MAX_DF
            and hit.coverage >= MIN_LEXICAL_COVERAGE
        )
        if not trusted or hit.doc_id in chosen:
            continue

        chunk = by_id.get(hit.doc_id)
        if chunk is None:
            row = rows.get(hit.doc_id)
            if row is None:
                continue
            embedding = row["embedding"]
            # True vector similarity for display and logging, since this
            # chunk never appeared in the vector results.
            score = (
                float(np.dot(np.asarray(embedding), np.asarray(query_vector)))
                if embedding is not None
                else 0.0
            )
            meta = row["meta"]
            chunk = RetrievedChunk(
                chunk_id=hit.doc_id,
                text=str(row["text"]),
                score=max(0.0, min(1.0, score)),
                breadcrumb=str(meta.get("breadcrumb") or meta.get("source", "")),
                source=str(meta.get("source", "")),
                doc_title=str(meta.get("doc_title", "")),
            )
        chunk.matched_terms = hit.matched
        chosen[hit.doc_id] = chunk

    vector_rank = {c.chunk_id: r for r, c in enumerate(candidates, start=1)}
    lexical_rank = {h.doc_id: r for r, h in enumerate(hits, start=1)}

    def rrf(chunk: RetrievedChunk) -> float:
        total = 0.0
        if chunk.chunk_id in vector_rank:
            total += 1 / (RRF_K + vector_rank[chunk.chunk_id])
        if chunk.chunk_id in lexical_rank:
            total += 1 / (RRF_K + lexical_rank[chunk.chunk_id])
        return total

    return sorted(chosen.values(), key=rrf, reverse=True)


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
    mode: str | None = None,
) -> RetrievalResult:
    """Search the knowledge base for context relevant to `query`.

    Args:
        query: The user's question, verbatim.
        top_k: Number of chunks to return. Defaults to `RETRIEVAL_TOP_K`.
        min_score: Similarity below which a match is treated as no match.
            Defaults to `RETRIEVAL_MIN_SCORE`.
        collection: An open collection to search. Injectable so tests can
            run against a fake without a database.
        mode: "hybrid" (vector plus BM25) or "vector". Defaults to
            `RETRIEVAL_MODE`. "vector" is kept so the two can be compared
            on the same questions.

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

    if (mode or settings.retrieval_mode) == "hybrid":
        above = _fuse_with_lexical(
            query, target, vector, candidates=candidates, qualified=above,
            fetch_k=fetch_k,
        )

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
