"""Look a named test up in the test catalogue, by name.

Why this exists. Asked "how much is a lipid profile?", the retriever
returned the generic Panels, home-collection and pricing sections and
never the Biochemistry table that holds the row - the question's "how
much is a" pulled it towards every section about money, and a long table
chunk ranks below shorter ones that merely mention the test. The Inquiry
Agent then, correctly, refused: it had not been shown the answer. The most
asked question about the laboratory's most common test was being declined.

Retuning a retriever to fix one phrasing risks every other, and there are
labelled questions guarding the current balance. The catalogue is a
structured table, though, and a question that names a test in it does not
need a similarity search to find the row. So the row is found by name, and
handed to the model verbatim alongside whatever retrieval returned. The
model still writes the answer, and the answer is still grounded in the
document - only the lookup is deterministic.

Nothing here is copied into code. The names come from the document, so a
test added to the catalogue is found at once and a test removed from it is
no longer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

from app.config import settings
from app.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

CATALOGUE_FILE = "test-catalogue-and-pricing.md"
CATALOGUE_TITLE = "Test Catalogue and Pricing"
MAX_ROWS = 3

# Words that appear in many test names without identifying one. A name is
# matched on all its words, so these only matter when they are the whole
# question.
_IGNORED = frozenset({"a", "an", "the", "of", "and", "or", "for", "to", "in"})


@dataclass(frozen=True)
class CatalogueRow:
    """One line of one table in the catalogue."""

    section: str
    name: str
    words: frozenset[str]
    aliases: frozenset[str]
    header: str
    line: str

    def as_chunk(self) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=f"catalogue-row:{self.section}:{self.name}",
            text=f"{self.header}\n{self.line}",
            # An exact match on a test's name, not a similarity: it is
            # shown to staff as the strongest evidence there is.
            score=1.0,
            breadcrumb=f"{CATALOGUE_TITLE} > {self.section}",
            source=CATALOGUE_FILE,
            doc_title=CATALOGUE_TITLE,
            matched_terms=[self.name],
        )


def _tokens(text: str) -> list[str]:
    """Lower-case words, with a plural "s" removed so tests matches test."""
    cleaned = "".join(c.lower() if c.isalnum() else " " for c in text)
    out = []
    for word in cleaned.split():
        if word in _IGNORED:
            continue
        out.append(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return out


def _split_name(cell: str) -> tuple[str, frozenset[str]]:
    """"Full blood count (FBC/CBC)" -> ("Full blood count", {"fbc", "cbc"})."""
    if "(" not in cell:
        return cell.strip(), frozenset()
    name, _, rest = cell.partition("(")
    inside = rest.rstrip(") ").strip()
    aliases = {
        part.strip().lower()
        for part in inside.split("/")
        # An abbreviation is one short token. "(3-month average)" is a
        # description, and would match any question containing "average".
        if part.strip() and " " not in part.strip() and len(part.strip()) <= 8
        and not any(ch.isdigit() for ch in part.strip()[:1])
    }
    return name.strip(), frozenset(aliases)


@lru_cache(maxsize=1)
def catalogue_rows() -> tuple[CatalogueRow, ...]:
    """Every test in the catalogue, read from the document."""
    path = settings.documents_dir / CATALOGUE_FILE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.warning("Test catalogue unreadable; named-test lookup is off")
        return ()

    rows: list[CatalogueRow] = []
    section = ""
    header = ""
    for raw in lines:
        line = raw.strip()
        if line.startswith("## "):
            section, header = line[3:].strip(), ""
            continue
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set(line) <= set("|- :"):
            continue  # the separator under a header
        if not header:
            header = line  # the first table row is the header
            continue
        if not cells or not cells[0]:
            continue
        name, aliases = _split_name(cells[0])
        words = frozenset(_tokens(name))
        if words:
            rows.append(CatalogueRow(section, name, words, aliases, header, line))
    return tuple(rows)


def rows_named_in(question: str) -> list[CatalogueRow]:
    """The catalogue rows for tests the question names, longest name first.

    A row matches when every word of its name is in the question, or when
    the question uses one of its abbreviations (FBC, LFT, ESR). Requiring
    every word keeps "blood glucose" from matching both the fasting and the
    random test, which would be answering a question nobody asked.
    """
    asked = set(_tokens(question))
    found = [
        row for row in catalogue_rows()
        if row.words <= asked or (row.aliases & asked)
    ]
    found.sort(key=lambda r: len(r.words), reverse=True)

    # A longer name swallows a shorter one it contains: "glucose tolerance
    # test" should not also pull in "glucose".
    kept: list[CatalogueRow] = []
    for row in found:
        if not any(row.words < other.words for other in kept):
            kept.append(row)
    return kept[:MAX_ROWS]
