"""Build the knowledge base: load documents, chunk, embed, persist.

The pipeline is deliberately a straight line, and each stage is a function
that can be run and inspected on its own:

    load_documents()  ->  chunk_markdown()  ->  embed_documents()  ->  Chroma

Ingestion is idempotent. Running it twice produces the same collection,
because chunk ids are derived from the source filename and position rather
than being generated randomly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.rag.chunking import Chunk, chunk_file
from app.rag.embeddings import embed_documents, embedding_dimension
from app.rag.store import get_collection, reset_collection

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Ingestion could not complete."""


@dataclass
class IngestionReport:
    """Outcome of an ingestion run, for printing and for tests."""

    documents: int
    chunks: int
    characters: int
    dimension: int
    collection: str
    rebuilt: bool
    per_document: dict[str, int]

    def summary(self) -> str:
        action = "Rebuilt" if self.rebuilt else "Updated"
        return (
            f"{action} collection {self.collection!r}: "
            f"{self.chunks} chunks from {self.documents} document(s), "
            f"{self.characters:,} characters, {self.dimension}-dim vectors."
        )


def find_documents(directory: Path | None = None) -> list[Path]:
    """Return the markdown files that make up the knowledge base.

    Raises:
        IngestionError: The directory is missing or contains no documents.
    """
    docs_dir = directory or settings.abs_path(settings.documents_dir)
    if not docs_dir.is_dir():
        raise IngestionError(
            f"Documents directory not found: {docs_dir}. "
            f"Create it and add markdown files, or set DOCUMENTS_DIR in .env."
        )

    paths = sorted(docs_dir.glob("*.md"))
    if not paths:
        raise IngestionError(f"No .md files found in {docs_dir}.")
    return paths


def build_chunks(paths: list[Path]) -> list[Chunk]:
    """Chunk every document, skipping any that fail rather than aborting.

    One malformed file should not prevent the rest of the knowledge base
    from being built; the failure is logged and reported instead.
    """
    chunks: list[Chunk] = []
    for path in paths:
        try:
            produced = chunk_file(path)
        except Exception as exc:
            logger.error("Skipping %s: %s", path.name, exc)
            continue
        if not produced:
            logger.warning("%s produced no chunks (is it empty?)", path.name)
            continue
        chunks.extend(produced)
    return chunks


def ingest(*, rebuild: bool = True, show_progress: bool = True) -> IngestionReport:
    """Run the full ingestion pipeline.

    Args:
        rebuild: Delete and recreate the collection first. Defaults to True
            because a partial update after a chunking change leaves stale
            chunks behind, which is a genuinely hard bug to spot — answers
            come back grounded in text that no longer exists in the source.
        show_progress: Display the embedding progress bar.

    Returns:
        An `IngestionReport` describing what was written.

    Raises:
        IngestionError: No documents were found, or nothing could be
            chunked.
    """
    paths = find_documents()
    chunks = build_chunks(paths)
    if not chunks:
        raise IngestionError("No chunks were produced from any document.")

    logger.info("Embedding %d chunks with %s", len(chunks), settings.embedding_model)
    vectors = embed_documents(
        [c.embed_text for c in chunks], show_progress=show_progress
    )

    collection = reset_collection() if rebuild else get_collection(create=True)

    # Chroma has a per-call batch ceiling; staying well under it also keeps
    # memory flat for larger corpora.
    batch_size = 256
    for start in range(0, len(chunks), batch_size):
        window = chunks[start : start + batch_size]
        collection.upsert(
            ids=[c.chunk_id for c in window],
            documents=[c.text for c in window],
            embeddings=vectors[start : start + batch_size],
            metadatas=[c.to_metadata() for c in window],
        )

    per_document: dict[str, int] = {}
    for chunk in chunks:
        per_document[chunk.source] = per_document.get(chunk.source, 0) + 1

    return IngestionReport(
        documents=len(paths),
        chunks=len(chunks),
        characters=sum(len(c.text) for c in chunks),
        dimension=embedding_dimension(),
        collection=settings.chroma_collection,
        rebuilt=rebuild,
        per_document=per_document,
    )
