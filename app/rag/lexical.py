"""BM25 keyword search over the chunks.

Vector search matches by meaning, which is its strength and its blind
spot. It handles "my tooth got knocked out" finding a section titled
"First Aid" with no shared words. It fails on the opposite case: a bare
proper noun. "do you take cigna" scored 0.23 against the insurance page
that literally lists "Cigna Dental PPO", because an embedding model gives
a brand name very little semantic weight.

BM25 is the complement. It scores exact term overlap, weighting rare
terms heavily — "cigna" appears in one chunk out of 74, so matching it is
strong evidence. Combining the two ("hybrid search") is the standard fix,
and the retriever does that fusion; this module only provides the
lexical ranking.

Implemented directly rather than via a library because the algorithm is
short and worth being able to read. The index is rebuilt from the vector
store's own documents, so the two can never disagree about what exists.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

# Standard BM25 parameters. k1 controls how quickly repeated terms stop
# adding score; b controls how much long chunks are penalised.
K1 = 1.5
B = 0.75

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words carrying no retrieval signal. Deliberately short: aggressive
# stopword lists remove words like "not" and "before", which matter in
# aftercare instructions.
STOPWORDS = frozenset(
    """
    a an the and or but if of to in on at by for with from as is are was
    were be been being do does did i me my we our you your it its this that
    these those can could would should will shall may might have has had
    what which who whom when where why how there here so than too very just
    about into over also any some please
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stopwords, crude-stem.

    The stemming only strips a plural "s", so "fillings" matches "filling"
    and "implants" matches "implant". Anything cleverer (Porter, Snowball)
    starts conflating words like "surgery" and "surge" that should stay
    apart in a clinical corpus.
    """
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


@dataclass
class LexicalHit:
    doc_id: str
    score: float
    matched: list[str]
    """Query terms that appeared in this document."""

    rarest_df: int
    """Document frequency of the rarest matched term. 1 means that term
    appears in this chunk and nowhere else in the corpus."""

    coverage: float = 0.0
    """Share of the query's information this document matched, 0-1.

    IDF-weighted, so matching "cigna" in "do you take cigna" counts for
    far more than matching "take". This is what separates a rare word that
    *is* the question from a rare word that merely appears in it — "car"
    in "renew my car insurance" matches a chunk about the car park."""


@dataclass
class BM25Index:
    """An in-memory BM25 index over a fixed set of documents."""

    doc_ids: list[str] = field(default_factory=list)
    term_freqs: list[Counter] = field(default_factory=list)
    doc_lengths: list[int] = field(default_factory=list)
    doc_freq: Counter = field(default_factory=Counter)
    avg_length: float = 0.0

    @classmethod
    def build(cls, documents: list[tuple[str, str]]) -> "BM25Index":
        """Index `(doc_id, text)` pairs."""
        index = cls()
        for doc_id, text in documents:
            tokens = tokenize(text)
            counts = Counter(tokens)
            index.doc_ids.append(doc_id)
            index.term_freqs.append(counts)
            index.doc_lengths.append(len(tokens))
            index.doc_freq.update(counts.keys())
        index.avg_length = (
            sum(index.doc_lengths) / len(index.doc_lengths) if index.doc_lengths else 0.0
        )
        return index

    @property
    def size(self) -> int:
        return len(self.doc_ids)

    def idf(self, term: str) -> float:
        """Inverse document frequency, BM25+ style so it is never negative."""
        df = self.doc_freq.get(term, 0)
        return math.log(1 + (self.size - df + 0.5) / (df + 0.5))

    def search(self, query: str, *, k: int = 10) -> list[LexicalHit]:
        """Return up to `k` documents ranked by BM25 score."""
        terms = list(dict.fromkeys(tokenize(query)))  # de-duplicated, ordered
        if not terms or not self.size:
            return []

        query_information = sum(self.idf(t) for t in terms) or 1.0

        hits: list[LexicalHit] = []
        for i, counts in enumerate(self.term_freqs):
            score = 0.0
            matched: list[str] = []
            length_norm = 1 - B + B * self.doc_lengths[i] / (self.avg_length or 1)
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                matched.append(term)
                score += self.idf(term) * tf * (K1 + 1) / (tf + K1 * length_norm)
            if matched:
                hits.append(
                    LexicalHit(
                        doc_id=self.doc_ids[i],
                        score=score,
                        matched=matched,
                        rarest_df=min(self.doc_freq[t] for t in matched),
                        coverage=sum(self.idf(t) for t in matched) / query_information,
                    )
                )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
