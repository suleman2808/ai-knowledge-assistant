"""Tests for the retrieval layer.

These run against a fake collection rather than a real Chroma database, so
they are fast, deterministic and do not require ingestion to have been run.
What is being tested is the *policy* — thresholding, diversity, failure
handling — not Chroma's nearest-neighbour search, which is not our code.
"""

from __future__ import annotations

import pytest

from app.rag.retriever import (
    MAX_PER_SECTION,
    RetrievalStatus,
    retrieve,
)
from app.rag.store import VectorStoreError


class FakeCollection:
    """Minimal stand-in for a Chroma collection.

    `rows` are (id, text, breadcrumb, distance) tuples, returned in the
    order given. Chroma reports cosine *distance*, so a distance of 0.2
    corresponds to a similarity of 0.8.
    """

    def __init__(self, rows: list[tuple[str, str, str, float]]) -> None:
        self.rows = rows
        self.last_n_results: int | None = None
        self.name = f"fake-{id(self)}"

    def count(self) -> int:
        return len(self.rows)

    def get(self, *, include):  # noqa: ANN001, ARG002
        """Full scan, used to build the keyword index."""
        return {
            "ids": [r[0] for r in self.rows],
            "documents": [r[1] for r in self.rows],
            "metadatas": [{"breadcrumb": r[2], "heading_path": r[2],
                           "source": "doc.md"} for r in self.rows],
            "embeddings": None,
        }

    def query(self, *, query_embeddings, n_results, include):  # noqa: ANN001
        self.last_n_results = n_results
        window = self.rows[:n_results]
        return {
            "ids": [[r[0] for r in window]],
            "documents": [[r[1] for r in window]],
            "metadatas": [
                [
                    {
                        "breadcrumb": r[2],
                        "source": f"{r[2].split(' > ')[0].lower().replace(' ', '-')}.md",
                        "doc_title": r[2].split(" > ")[0],
                    }
                    for r in window
                ]
            ],
            "distances": [[r[3] for r in window]],
        }


class BrokenCollection:
    """A collection that fails the way an unreachable database would."""

    def count(self) -> int:
        return 0

    def query(self, **_kwargs):  # noqa: ANN003
        raise RuntimeError("connection reset by peer")


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid loading the real model; the vector value is irrelevant here."""
    from app.rag import retriever

    monkeypatch.setattr("app.rag.retriever.embed_query", lambda _text: [0.0] * 384)
    retriever._lexical_cache.clear()


def test_returns_chunks_above_the_threshold() -> None:
    collection = FakeCollection(
        [
            ("a::0", "Root canal, molar: $1,250", "Services > Restorative", 0.25),
            ("b::0", "We open at eight", "Hours > Opening Hours", 0.55),
        ]
    )

    result = retrieve("root canal cost", collection=collection, min_score=0.30)

    assert result.status is RetrievalStatus.OK
    assert result.grounded
    assert [c.chunk_id for c in result.chunks] == ["a::0", "b::0"]
    assert result.chunks[0].score == pytest.approx(0.75)


def test_weak_matches_are_not_treated_as_answers() -> None:
    """The core anti-hallucination guarantee.

    Vector search always returns its nearest neighbours however distant, so
    a query the corpus cannot answer still produces rows. Those must not
    reach the agent as grounding.
    """
    collection = FakeCollection(
        [("a::0", "We open at eight", "Hours > Opening Hours", 0.88)]
    )

    result = retrieve("what is the capital of France", collection=collection, min_score=0.30)

    assert result.status is RetrievalStatus.LOW_CONFIDENCE
    assert result.grounded is False
    assert result.chunks == [], "weak chunks must not be offered as context"
    assert result.best_score == pytest.approx(0.12)


def test_threshold_boundary_is_inclusive() -> None:
    """A score exactly at the threshold counts as a match."""
    collection = FakeCollection([("a::0", "text", "Doc > Section", 0.70)])

    result = retrieve("q", collection=collection, min_score=0.30)

    assert result.status is RetrievalStatus.OK


def test_one_section_cannot_monopolise_the_context() -> None:
    """Adjacent chunks from a long section are capped."""
    rows = [
        (f"a::{i}", f"part {i}", "Aftercare > After an Extraction", 0.10 + i * 0.01)
        for i in range(6)
    ]
    rows.append(("b::0", "different", "Emergency > First Aid", 0.20))
    collection = FakeCollection(rows)

    result = retrieve("extraction", collection=collection, top_k=5, min_score=0.30)

    breadcrumbs = [c.breadcrumb for c in result.chunks]
    assert breadcrumbs.count("Aftercare > After an Extraction") == MAX_PER_SECTION
    assert "Emergency > First Aid" in breadcrumbs


def test_results_are_ordered_by_score() -> None:
    collection = FakeCollection(
        [
            ("a::0", "weakest", "Doc > A", 0.40),
            ("b::0", "strongest", "Doc > B", 0.10),
            ("c::0", "middle", "Doc > C", 0.25),
        ]
    )

    result = retrieve("q", collection=collection, min_score=0.30)

    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert result.chunks[0].chunk_id == "b::0"


def test_top_k_is_respected() -> None:
    rows = [(f"d{i}::0", f"text {i}", f"Doc > Section {i}", 0.10) for i in range(10)]
    collection = FakeCollection(rows)

    result = retrieve("q", collection=collection, top_k=3, min_score=0.30)

    assert len(result.chunks) == 3


def test_over_fetches_so_filtering_has_candidates() -> None:
    """Fetching exactly k would leave nothing after capping and filtering."""
    collection = FakeCollection([(f"d{i}::0", "t", f"Doc > S{i}", 0.10) for i in range(30)])

    retrieve("q", collection=collection, top_k=4)

    assert collection.last_n_results is not None
    assert collection.last_n_results > 4


def test_empty_index_is_distinguished_from_no_match() -> None:
    """An operator error and a user miss need different messages."""
    result = retrieve("anything", collection=FakeCollection([]))

    assert result.status is RetrievalStatus.EMPTY_INDEX
    assert "ingest" in result.detail.lower()


def test_store_failure_degrades_rather_than_raising() -> None:
    """A database outage must not surface as a stack trace."""
    result = retrieve("anything", collection=BrokenCollection())

    assert result.status is RetrievalStatus.UNAVAILABLE
    assert result.grounded is False
    assert "connection reset" in result.detail


def test_missing_collection_reports_empty_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Querying before ingestion should explain itself, not crash."""

    def _raise() -> None:
        raise VectorStoreError("Collection 'clinic_docs' does not exist. Run ingest.")

    monkeypatch.setattr("app.rag.retriever.get_collection", lambda create: _raise())

    result = retrieve("anything")

    assert result.status is RetrievalStatus.EMPTY_INDEX


def test_empty_query_is_rejected_without_searching() -> None:
    collection = FakeCollection([("a::0", "text", "Doc > S", 0.10)])

    result = retrieve("   ", collection=collection)

    assert result.status is RetrievalStatus.NO_MATCH
    assert collection.last_n_results is None, "should not have queried at all"


def test_context_is_numbered_for_citation() -> None:
    collection = FakeCollection(
        [
            ("a::0", "Cancellation is $50.", "Policy > Cancellation", 0.20),
            ("b::0", "We open at eight.", "Hours > Opening Hours", 0.30),
        ]
    )

    result = retrieve("q", collection=collection, min_score=0.30)
    context = result.as_context()

    assert context.startswith("[1] Policy > Cancellation")
    assert "[2] Hours > Opening Hours" in context
    assert "Cancellation is $50." in context


def test_sources_are_serialisable_for_the_api() -> None:
    collection = FakeCollection([("a::0", "text", "Doc > Section", 0.20)])

    result = retrieve("q", collection=collection, min_score=0.30)
    sources = result.sources

    assert sources == [
        {"breadcrumb": "Doc > Section", "source": "doc.md", "score": 0.8}
    ]
