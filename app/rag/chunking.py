"""Split markdown documents into retrievable chunks.

Chunking is the decision that most determines whether retrieval feels sharp
or sloppy, so it lives in its own module with no dependencies on the vector
store or the embedding model. That makes it directly testable: give it
text, inspect the chunks.

The strategy, in order of preference:

1. **Split on document structure, not character count.** A markdown heading
   marks a boundary the author already decided was meaningful. A chunk that
   corresponds to "Insurance and Payment > Payment Plans" is a coherent
   answer to a question; a chunk that starts mid-sentence in the fee table
   and stops before the policy it belongs to is not.
2. **Keep tables intact.** A markdown table split across two chunks leaves
   rows with no header, which is worse than useless — the numbers survive
   but their meaning does not.
3. **Fall back to size-based splitting only when a section is genuinely too
   long**, and then split on paragraph boundaries with overlap, never
   mid-sentence.
4. **Carry the heading path as metadata.** It is prepended to the embedded
   text, which measurably sharpens retrieval, and it lets the Inquiry Agent
   cite "Insurance and Payment > Payment Plans" instead of "chunk 47".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Upper bound on chunk size, in characters. Roughly 300 tokens, which sits
# comfortably inside the embedding model's 256-token window for the text
# that matters while leaving room for the heading prefix.
MAX_CHUNK_CHARS = 1200

# How much of the previous window to repeat at the start of the next one
# when a long section must be split. Overlap means a fact that straddles a
# boundary still appears whole in at least one chunk.
OVERLAP_CHARS = 150

# Sections shorter than this are merged into a neighbour. A 40-character
# chunk carries almost no signal and pollutes results.
MIN_CHUNK_CHARS = 120

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """One retrievable unit of a document."""

    chunk_id: str
    text: str
    """The body text, as it will be shown to the user and the LLM."""

    heading_path: list[str] = field(default_factory=list)
    """Heading breadcrumb, outermost first, e.g. ["Insurance", "Payment Plans"]."""

    source: str = ""
    """Filename the chunk came from."""

    doc_title: str = ""
    """The document's H1, used for citations."""

    chunk_index: int = 0

    @property
    def breadcrumb(self) -> str:
        """Human-readable location, used in citations."""
        return " > ".join(self.heading_path) if self.heading_path else self.doc_title

    @property
    def embed_text(self) -> str:
        """The string actually sent to the embedding model.

        The heading path is prepended so that a chunk about late
        cancellation fees embeds near the words "cancellation" and "fee"
        even when the body itself only says "$50 applies". Without this,
        sections that rely on their heading for context retrieve poorly.
        """
        if self.heading_path:
            return f"{' > '.join(self.heading_path)}\n\n{self.text}"
        return self.text

    def to_metadata(self) -> dict[str, str | int]:
        """Flatten to Chroma-compatible metadata.

        Chroma only accepts scalar metadata values, so the heading path is
        stored as a pre-joined string rather than a list.
        """
        return {
            "source": self.source,
            "doc_title": self.doc_title,
            "heading_path": " > ".join(self.heading_path),
            "breadcrumb": self.breadcrumb,
            "chunk_index": self.chunk_index,
            "n_chars": len(self.text),
        }


@dataclass
class _Section:
    """An intermediate heading-delimited region of a document."""

    heading_path: list[str]
    body: str


def _parse_sections(markdown: str) -> tuple[str, list[_Section]]:
    """Split markdown into sections keyed by their heading path.

    Returns the document title (its H1, or an empty string) together with
    the sections in document order. Body text appearing before any heading
    is attached to the title.
    """
    lines = markdown.splitlines()
    doc_title = ""
    sections: list[_Section] = []

    # stack[i] holds the heading text at level i+1, so the path for a level
    # 3 heading is stack[:3].
    stack: list[str] = []
    current_path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(_Section(heading_path=list(current_path), body=body))
        buffer.clear()

    for line in lines:
        match = HEADING_RE.match(line)
        if not match:
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        heading = match.group(2).strip()

        if level == 1 and not doc_title:
            doc_title = heading

        # Truncate the stack to the parent level, then push this heading.
        del stack[level - 1 :]
        stack.append(heading)
        current_path = list(stack)

    flush()
    return doc_title, sections


def _split_blocks(body: str) -> list[str]:
    """Break a section body into atomic blocks that must not be split.

    Paragraphs split on blank lines. A run of consecutive table rows is
    treated as a single block regardless of length, so that a table is
    never severed from its header row.
    """
    blocks: list[str] = []
    table: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append("\n".join(paragraph).strip())
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append("\n".join(table).strip())
            table.clear()

    for line in body.splitlines():
        is_table_row = line.lstrip().startswith("|")

        if is_table_row:
            flush_paragraph()
            table.append(line)
            continue

        flush_table()
        if line.strip():
            paragraph.append(line)
        else:
            flush_paragraph()

    flush_paragraph()
    flush_table()
    return [b for b in blocks if b]


def _split_table(table: str) -> list[str]:
    """Split an oversized markdown table, repeating the header in each part.

    Only reached when a single table exceeds the chunk limit on its own.
    Every resulting piece is a valid, self-describing table.
    """
    rows = table.splitlines()
    if len(rows) < 3:
        return [table]

    header, separator, body_rows = rows[0], rows[1], rows[2:]
    prefix = f"{header}\n{separator}"
    parts: list[str] = []
    current: list[str] = []

    for row in body_rows:
        candidate = len(prefix) + sum(len(r) + 1 for r in current) + len(row) + 1
        if current and candidate > MAX_CHUNK_CHARS:
            parts.append("\n".join([prefix, *current]))
            current = [row]
        else:
            current.append(row)

    if current:
        parts.append("\n".join([prefix, *current]))
    return parts


def _pack(blocks: list[str]) -> list[str]:
    """Greedily pack blocks into windows no larger than the limit.

    Consecutive windows overlap by `OVERLAP_CHARS` characters of trailing
    text, so a fact spanning a boundary still appears intact somewhere.
    """
    expanded: list[str] = []
    for block in blocks:
        if len(block) <= MAX_CHUNK_CHARS:
            expanded.append(block)
        elif block.lstrip().startswith("|"):
            expanded.extend(_split_table(block))
        else:
            # An unusually long paragraph: split on sentence boundaries.
            sentences = re.split(r"(?<=[.!?])\s+", block)
            current = ""
            for sentence in sentences:
                if current and len(current) + len(sentence) + 1 > MAX_CHUNK_CHARS:
                    expanded.append(current.strip())
                    current = sentence
                else:
                    current = f"{current} {sentence}".strip()
            if current:
                expanded.append(current.strip())

    windows: list[str] = []
    current = ""
    for block in expanded:
        if not current:
            current = block
        elif len(current) + len(block) + 2 <= MAX_CHUNK_CHARS:
            current = f"{current}\n\n{block}"
        else:
            windows.append(current)
            tail = current[-OVERLAP_CHARS:].lstrip()
            # Only carry overlap when it does not push the new window over
            # the limit on its own.
            current = f"{tail}\n\n{block}" if len(tail) + len(block) + 2 <= MAX_CHUNK_CHARS else block

    if current:
        windows.append(current)
    return windows


def chunk_markdown(markdown: str, *, source: str) -> list[Chunk]:
    """Chunk one markdown document.

    Args:
        markdown: The document's full text.
        source: Filename, recorded on every chunk for citation.

    Returns:
        Chunks in document order, each with a stable id of the form
        `<source stem>::<index>`.
    """
    doc_title, sections = _parse_sections(markdown)
    stem = Path(source).stem

    # Merge sections too small to stand alone into the previous one, so long
    # as the result stays within the limit. Short subsections are common in
    # policy documents and are far more useful attached to their context.
    #
    # Merging is restricted to *siblings* — sections sharing a parent. A
    # short section absorbed into an unrelated branch would be served under
    # the wrong breadcrumb, so a question about complaints could be answered
    # with text cited as "Payment > Refunds". Losing the merge is a small
    # cost; citing the wrong section undermines the whole premise.
    merged: list[_Section] = []
    for section in sections:
        if (
            merged
            and len(section.body) < MIN_CHUNK_CHARS
            and merged[-1].heading_path[:-1] == section.heading_path[:-1]
            and len(merged[-1].body) + len(section.body) + 2 <= MAX_CHUNK_CHARS
        ):
            previous = merged[-1]
            heading = section.heading_path[-1] if section.heading_path else ""
            addition = f"{heading}\n{section.body}" if heading else section.body
            previous.body = f"{previous.body}\n\n{addition}"
        else:
            merged.append(_Section(list(section.heading_path), section.body))

    chunks: list[Chunk] = []
    for section in merged:
        for window in _pack(_split_blocks(section.body)):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}::{index}",
                    text=window,
                    heading_path=section.heading_path,
                    source=source,
                    doc_title=doc_title,
                    chunk_index=index,
                )
            )
    return chunks


def chunk_file(path: Path) -> list[Chunk]:
    """Read and chunk a markdown file from disk."""
    return chunk_markdown(path.read_text(encoding="utf-8"), source=path.name)
