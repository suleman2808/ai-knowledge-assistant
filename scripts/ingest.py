"""Build the knowledge base from the documents on disk.

    python -m scripts.ingest              # rebuild from scratch (default)
    python -m scripts.ingest --update     # add to the existing collection

Run this once after cloning, and again whenever the documents or the
chunking strategy change.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from app.rag.embeddings import EmbeddingError
from app.rag.ingest import IngestionError, ingest
from app.rag.store import VectorStoreError, collection_stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--update",
        action="store_true",
        help="Upsert into the existing collection instead of rebuilding it.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide the progress bar.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    # The model download logs every HTTP request at INFO, which buries the
    # output that matters.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    print("Building knowledge base")
    print("  (first run downloads the embedding model, roughly 90 MB)\n")

    started = time.perf_counter()
    try:
        report = ingest(rebuild=not args.update, show_progress=not args.quiet)
    except IngestionError as exc:
        print(f"\nIngestion failed: {exc}")
        return 1
    except EmbeddingError as exc:
        print(f"\nEmbedding failed: {exc}")
        return 1
    except VectorStoreError as exc:
        print(f"\nVector store failed: {exc}")
        return 1

    elapsed = time.perf_counter() - started

    print(f"\n{report.summary()}")
    print(f"Completed in {elapsed:.1f}s\n")
    print("Chunks per document:")
    for source, count in sorted(report.per_document.items()):
        print(f"  {count:>3}  {source}")

    stats = collection_stats()
    print(f"\nCollection now holds {stats['count']} chunks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
