"""Tests for the chunking strategy.

Chunking has no external dependencies — no model, no network, no database —
so it can be tested exhaustively and fast. These tests encode the
guarantees the retrieval layer relies on.
"""

from __future__ import annotations

from app.rag.chunking import (
    MAX_CHUNK_CHARS,
    chunk_markdown,
)


def test_splits_on_headings_not_character_count() -> None:
    """Each heading should own its own chunk."""
    markdown = """# Clinic Handbook

## Opening Hours

We open at eight in the morning and close at five in the afternoon.
Saturday hours differ and are listed separately below for clarity.

## Parking

Parking behind the building is free for three hours with validation
collected from the reception desk before you leave the premises.
"""
    chunks = chunk_markdown(markdown, source="handbook.md")

    breadcrumbs = [c.breadcrumb for c in chunks]
    assert "Clinic Handbook > Opening Hours" in breadcrumbs
    assert "Clinic Handbook > Parking" in breadcrumbs

    hours = next(c for c in chunks if c.breadcrumb.endswith("Opening Hours"))
    assert "Parking" not in hours.text


def test_heading_path_tracks_nesting() -> None:
    """A deeper heading keeps its ancestors, and siblings replace peers."""
    markdown = """# Policies

## Payment

### Payment Plans

Treatment over eight hundred dollars may be split into interest-free
instalments arranged directly with the practice over several months.

### Refunds

Where treatment has failed within the guarantee period we will remake it
or refund the fee that was paid for that treatment.

## Complaints

Complaints are acknowledged within two working days of being received by
any member of the practice team.
"""
    chunks = chunk_markdown(markdown, source="policies.md")
    breadcrumbs = [c.breadcrumb for c in chunks]

    assert "Policies > Payment > Payment Plans" in breadcrumbs
    assert "Policies > Payment > Refunds" in breadcrumbs
    # Complaints is a sibling of Payment, so Payment must not remain in the path.
    assert "Policies > Complaints" in breadcrumbs


def test_table_is_never_split_from_its_header() -> None:
    """Every chunk containing table rows must contain the header row."""
    rows = "\n".join(
        f"| Service number {i} | ${i * 25} | {i * 5} min |" for i in range(1, 90)
    )
    markdown = f"""# Fees

## Price List

| Service | Price | Duration |
| --- | --- | --- |
{rows}
"""
    chunks = chunk_markdown(markdown, source="fees.md")
    table_chunks = [c for c in chunks if "|" in c.text]

    assert len(table_chunks) > 1, "table should have been split across chunks"
    for chunk in table_chunks:
        assert "| Service | Price | Duration |" in chunk.text


def test_chunks_respect_the_size_limit() -> None:
    """Long prose is split rather than emitted as one oversized chunk."""
    paragraph = (
        "The practice provides a full range of restorative treatment to "
        "patients of every age and background across the wider area. "
    )
    markdown = "# Overview\n\n## Services\n\n" + (paragraph * 40)

    chunks = chunk_markdown(markdown, source="overview.md")

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= MAX_CHUNK_CHARS


def test_embed_text_includes_the_heading_path() -> None:
    """Headings are embedded with the body so short sections retrieve well."""
    markdown = """# Appointment Policy

## Cancellation Fees

A fee of fifty dollars applies when less than twenty-four hours of notice
is given before the appointment time.
"""
    chunk = chunk_markdown(markdown, source="policy.md")[0]

    assert chunk.embed_text.startswith("Appointment Policy > Cancellation Fees")
    assert "fifty dollars" in chunk.embed_text
    # The stored text stays clean; only the embedded form carries the prefix.
    assert not chunk.text.startswith("Appointment Policy >")


def test_chunk_ids_are_stable_across_runs() -> None:
    """Ingestion must be idempotent, which requires deterministic ids."""
    markdown = """# Guide

## First

Content for the first section of the guide, long enough to survive the
minimum-size merge that combines very short sections together.

## Second

Content for the second section of the guide, also written at a length
that will not be merged into its preceding sibling section.
"""
    first = chunk_markdown(markdown, source="guide.md")
    second = chunk_markdown(markdown, source="guide.md")

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id == "guide::0"


def test_metadata_is_scalar_only() -> None:
    """Chroma rejects list and dict metadata values."""
    markdown = "# Doc\n\n## Section\n\nSome body text that is long enough to keep.\n"
    chunk = chunk_markdown(markdown, source="doc.md")[0]

    for key, value in chunk.to_metadata().items():
        assert isinstance(value, (str, int, float, bool)), f"{key} is {type(value)}"


def test_tiny_sections_are_merged_into_their_neighbour() -> None:
    """A heading with one short line should not become its own chunk."""
    markdown = """# Contact

## Telephone

Call us.

## Email

Write to us.

## Address

Two four one eight South East Hawthorne Boulevard, Suite Two Hundred,
Portland, Oregon, nine seven two one four, above the pharmacy on the
second floor of the Hawthorne Medical Building.
"""
    chunks = chunk_markdown(markdown, source="contact.md")

    assert len(chunks) < 3, "short sections should have been merged"
    combined = " ".join(c.text for c in chunks)
    assert "Call us." in combined
    assert "Write to us." in combined


def test_empty_document_produces_no_chunks() -> None:
    """A heading with no body is not a retrievable unit."""
    assert chunk_markdown("# Title Only\n\n## Empty\n", source="empty.md") == []
