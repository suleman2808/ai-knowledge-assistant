"""Tests for BM25, hybrid fusion, and follow-up query rewriting.

These encode the three retrieval bugs found by the first analytics
report, plus the regression introduced by the first attempt at fixing
one of them.
"""

from __future__ import annotations

import pytest

from app.agents.inquiry import _standalone_question, answer_inquiry
from app.llm import LLMError
from app.rag import retriever
from app.rag.lexical import BM25Index, tokenize
from app.rag.retriever import RetrievalResult, RetrievalStatus, retrieve


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_tokenizer_drops_stopwords_and_plural_s() -> None:
    assert tokenize("Do you take the fillings?") == ["take", "filling"]


def test_tokenizer_does_not_mangle_double_s() -> None:
    """"glass" is not the plural of "glas"."""
    assert tokenize("glass") == ["glass"]


def test_rare_terms_outrank_common_ones() -> None:
    index = BM25Index.build(
        [
            ("insurance", "We are in network with Cigna Dental PPO and Aetna."),
            ("hours", "We are open Monday to Friday from eight."),
            ("parking", "We are above the pharmacy with parking behind."),
        ]
    )

    hits = index.search("do you take cigna")

    assert hits[0].doc_id == "insurance"
    assert hits[0].rarest_df == 1
    assert hits[0].matched == ["cigna"]


# A small corpus that mirrors the real one's word frequencies where they
# matter. IDF is relative, so in a four-document corpus where "take" and
# "cigna" each appear once they weigh the same and nothing is learned. In
# the real corpus "take" is in 10 of 74 chunks and "cigna" in 1; here it
# is common for the same reason — aftercare advice says "take" a lot.
SMALL_CORPUS = [
    ("insurance", "We accept Cigna Dental PPO."),
    ("parking", "A surface car park is behind the building."),
    ("benefits", "Your insurance annual maximum resets in January."),
    ("aftercare", "Take pain relief before the anaesthetic wears off."),
    ("implants", "Take all prescribed medication to completion."),
    ("xrays", "X-rays take about ten minutes."),
]


def test_coverage_reflects_how_much_of_the_query_matched() -> None:
    """The measure that separates "cigna" from "car" in "car insurance"."""
    index = BM25Index.build(SMALL_CORPUS)

    cigna = next(h for h in index.search("do you take cigna") if h.doc_id == "insurance")
    car = next(h for h in index.search("renew my car insurance") if h.doc_id == "parking")

    assert cigna.coverage > 0.575
    assert car.coverage < 0.575


def test_words_the_corpus_has_never_seen_lower_coverage() -> None:
    """A known limitation, pinned down so it stays deliberate.

    A query word absent from every document gets maximal IDF. That is
    what correctly sinks "renew my car insurance" — "renew" appears
    nowhere, because the clinic does not do that. The cost is that
    unfamiliar filler ("yo", "ur") lowers coverage too, so a slangy
    brand-name question can miss the keyword path.

    It fails safe: the query falls back to vector search and, at worst,
    a refusal. It never produces a false answer.
    """
    index = BM25Index.build(SMALL_CORPUS)

    plain = next(h for h in index.search("do you take cigna") if h.doc_id == "insurance")
    slangy = next(h for h in index.search("yo does ur place take cigna")
                  if h.doc_id == "insurance")

    assert slangy.coverage < plain.coverage


def test_no_overlap_returns_nothing() -> None:
    index = BM25Index.build([("a", "fillings and crowns")])

    assert index.search("capital of france") == []


# ---------------------------------------------------------------------------
# Hybrid fusion
# ---------------------------------------------------------------------------


class Collection:
    """Fake collection where vector distances are set per row."""

    def __init__(self, rows: list[tuple[str, str, str, float]]) -> None:
        self.rows = rows
        self.name = f"fake-{id(self)}"

    def count(self) -> int:
        return len(self.rows)

    def query(self, *, query_embeddings, n_results, include):  # noqa: ANN001, ARG002
        ordered = sorted(self.rows, key=lambda r: r[3])[:n_results]
        return {
            "ids": [[r[0] for r in ordered]],
            "documents": [[r[1] for r in ordered]],
            "metadatas": [[{"breadcrumb": r[2], "source": "d.md"} for r in ordered]],
            "distances": [[r[3] for r in ordered]],
        }

    def get(self, *, include):  # noqa: ANN001, ARG002
        return {
            "ids": [r[0] for r in self.rows],
            "documents": [r[1] for r in self.rows],
            "metadatas": [{"breadcrumb": r[2], "heading_path": r[2], "source": "d.md"}
                          for r in self.rows],
            "embeddings": None,
        }


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.rag.retriever.embed_query", lambda _t: [0.0] * 384)
    retriever._lexical_cache.clear()


# Distance 0.77 is similarity 0.23: the observed vector score for "do you
# take cigna" against the page listing Cigna, below the 0.30 threshold.
CORPUS = [
    ("ins::0", "We are in-network with Delta Dental, Cigna Dental PPO and MetLife.",
     "Insurance > Accepted Insurance Providers", 0.80),
    ("pay::0", "We accept cash, cards, HSA and FSA cards.",
     "Insurance > Payment Methods", 0.77),
    ("park::0", "A surface car park behind the building is free for three hours.",
     "Hours > Parking and Transit", 0.85),
    ("hours::0", "Monday to Friday 8am to 5pm. Saturday 9am to 1pm.",
     "Hours > Opening Hours", 0.90),
    # "take" is common, as in the real corpus, so it carries little weight.
    ("after::0", "Take pain relief before the anaesthetic wears off.",
     "Aftercare > After a Filling", 0.95),
    ("after::1", "Take all prescribed medication to completion.",
     "Aftercare > After Implant Surgery", 0.96),
    ("xray::0", "Bitewing X-rays take about ten minutes.",
     "Services > Preventive Care", 0.97),
]


def test_vector_mode_misses_a_bare_brand_name() -> None:
    """The original bug, reproduced."""
    result = retrieve("do you take cigna", collection=Collection(CORPUS), mode="vector")

    assert result.status is RetrievalStatus.LOW_CONFIDENCE


def test_hybrid_mode_finds_a_bare_brand_name() -> None:
    result = retrieve("do you take cigna", collection=Collection(CORPUS), mode="hybrid")

    assert result.grounded
    assert result.chunks[0].breadcrumb.endswith("Accepted Insurance Providers")
    assert result.chunks[0].matched_terms == ["cigna"]


def test_coincidental_rare_word_is_not_trusted() -> None:
    """The regression the first hybrid attempt introduced.

    "car" is rare and matches "car park", but it is a fraction of what
    "renew my car insurance" asks. Rank and rarity alone let it through
    and doubled out-of-scope false positives from 3/8 to 6/8.
    """
    result = retrieve(
        "can you help me renew my car insurance", collection=Collection(CORPUS),
        mode="hybrid",
    )

    assert not any(c.chunk_id == "park::0" for c in result.chunks)


def test_hybrid_never_admits_more_than_vector_on_no_overlap() -> None:
    for mode in ("vector", "hybrid"):
        result = retrieve("what is the capital of france", collection=Collection(CORPUS),
                          mode=mode)
        assert not result.grounded


def test_hybrid_falls_back_to_vector_if_the_keyword_index_fails() -> None:
    """Hybrid is an improvement, not a dependency."""

    class NoFullScan(Collection):
        def get(self, **_kwargs):  # noqa: ANN003
            raise RuntimeError("scan not supported")

    rows = [("a::0", "text", "Doc > A", 0.2)]
    result = retrieve("anything", collection=NoFullScan(rows), mode="hybrid")

    assert result.status is RetrievalStatus.OK
    assert [c.chunk_id for c in result.chunks] == ["a::0"]


def test_keyword_index_rebuilds_when_the_collection_changes() -> None:
    """Re-ingestion changes the count, which must invalidate the cache."""
    collection = Collection(CORPUS[:2])
    retrieve("cigna", collection=collection, mode="hybrid")

    collection.rows = CORPUS
    retrieve("cigna", collection=collection, mode="hybrid")

    (index, _rows), = retriever._lexical_cache.values()
    assert index.size == len(CORPUS)


# ---------------------------------------------------------------------------
# Follow-up rewriting
# ---------------------------------------------------------------------------

HISTORY = [
    {"role": "user", "content": "how much is a filling"},
    {"role": "assistant", "content": "A composite filling on one surface is $165."},
]


def test_no_rewrite_without_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first message must not cost an extra model call."""
    monkeypatch.setattr(
        "app.agents.inquiry.complete",
        lambda *_a, **_k: pytest.fail("should not rewrite without history"),
    )

    assert _standalone_question("how much is a crown", None) == ("how much is a crown", False)


def test_follow_up_is_rewritten_for_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.agents.inquiry.complete",
        lambda *_a, **_k: "Is the price of a filling for one surface?",
    )

    query, rewritten = _standalone_question("is that for one surface?", HISTORY)

    assert rewritten is True
    assert "filling" in query


def test_rewrite_failure_falls_back_to_the_original(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_a, **_k):  # noqa: ANN002, ANN003
        raise LLMError("rate limited")

    monkeypatch.setattr("app.agents.inquiry.complete", _fail)

    assert _standalone_question("is that for one surface?", HISTORY) == (
        "is that for one surface?", False,
    )


def test_a_rewrite_that_answers_instead_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that answers rather than rewrites pads the query with
    invented facts. Length is the cheap, reliable tell."""
    monkeypatch.setattr("app.agents.inquiry.complete", lambda *_a, **_k: "x" * 500)

    query, rewritten = _standalone_question("and that?", HISTORY)

    assert (query, rewritten) == ("and that?", False)


def test_rewritten_query_is_what_gets_searched(monkeypatch: pytest.MonkeyPatch) -> None:
    searched: list[str] = []

    def _retrieve(query: str) -> RetrievalResult:
        searched.append(query)
        return RetrievalResult(query=query, status=RetrievalStatus.LOW_CONFIDENCE)

    monkeypatch.setattr(
        "app.agents.inquiry.complete",
        lambda *_a, **_k: "Is the price of a filling for one surface?",
    )
    monkeypatch.setattr("app.agents.inquiry.retrieve", _retrieve)

    response = answer_inquiry("is that for one surface?", history=HISTORY)

    assert searched == ["Is the price of a filling for one surface?"]
    assert response.metadata["search_query"] == "Is the price of a filling for one surface?"
