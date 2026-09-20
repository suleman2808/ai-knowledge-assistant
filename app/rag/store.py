"""ChromaDB vector store access.

One module owns the connection to Chroma so that the collection is opened
with consistent settings everywhere, and so that swapping the vector store
later touches one file rather than both ingestion and retrieval.

The collection is configured for **cosine** distance. Chroma defaults to
squared L2, which would still rank correctly for normalised vectors but
returns distances in a different range, making any score threshold we set
meaningless. Being explicit here is what lets the retrieval layer use a
fixed, interpretable relevance cut-off.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_client: Any = None


class VectorStoreError(RuntimeError):
    """The vector store could not be opened or queried."""


def get_client() -> Any:
    """Return the persistent Chroma client, creating it on first use."""
    global _client
    if _client is not None:
        return _client

    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise VectorStoreError(f"chromadb is not installed: {exc}") from exc

    path = settings.abs_path(settings.chroma_dir)
    path.mkdir(parents=True, exist_ok=True)

    _client = chromadb.PersistentClient(
        path=str(path),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )
    return _client


def get_collection(*, create: bool = True) -> Any:
    """Open the document collection.

    Args:
        create: When True the collection is created if absent. Retrieval
            passes False so that querying before ingestion produces a clear
            "knowledge base is empty" error rather than silently returning
            no results from a collection that never existed.
    """
    client = get_client()
    name = settings.chroma_collection

    if create:
        return client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    try:
        return client.get_collection(name=name)
    except Exception as exc:
        raise VectorStoreError(
            f"Collection {name!r} does not exist. Run `python -m scripts.ingest` "
            f"to build the knowledge base."
        ) from exc


def reset_collection() -> Any:
    """Delete and recreate the collection, returning the empty collection.

    Used by `--rebuild` during ingestion. Recreating is preferred over
    upserting because it guarantees no orphaned chunks survive from a
    previous chunking strategy, which is the most common way a knowledge
    base quietly ends up serving stale text.
    """
    client = get_client()
    name = settings.chroma_collection
    try:
        client.delete_collection(name=name)
        logger.info("Deleted existing collection %s", name)
    except Exception:
        # Not existing is the normal case on a first run.
        pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def collection_stats() -> dict[str, Any]:
    """Summarise the knowledge base, for `/health` and diagnostics."""
    try:
        collection = get_collection(create=False)
    except VectorStoreError:
        return {"exists": False, "count": 0, "sources": []}

    try:
        count = collection.count()
        sources: set[str] = set()
        if count:
            sample = collection.get(include=["metadatas"], limit=count)
            sources = {
                str(m.get("source", "")) for m in (sample.get("metadatas") or []) if m
            }
        return {
            "exists": True,
            "count": count,
            "sources": sorted(s for s in sources if s),
        }
    except Exception as exc:
        logger.warning("Could not read collection stats: %s", exc)
        return {"exists": True, "count": 0, "sources": [], "error": str(exc)}
