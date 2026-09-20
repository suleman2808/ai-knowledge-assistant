"""Inspect chunking output without embedding anything.

Chunking quality is easy to get wrong and easy to check, so it is worth
looking at before spending time on embeddings:

    python -m scripts.inspect_chunks              # summary for every file
    python -m scripts.inspect_chunks --show 3     # print 3 sample chunks
    python -m scripts.inspect_chunks --file insurance-and-payment.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.config import settings
from app.rag.chunking import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, chunk_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Inspect a single filename.")
    parser.add_argument(
        "--show", type=int, default=0, help="Print this many full chunks per file."
    )
    args = parser.parse_args()

    docs_dir = settings.abs_path(settings.documents_dir)
    if not docs_dir.is_dir():
        print(f"No documents directory at {docs_dir}")
        return 1

    paths = sorted(docs_dir.glob("*.md"))
    if args.file:
        paths = [p for p in paths if p.name == args.file]
        if not paths:
            print(f"No such document: {args.file}")
            return 1

    total_chunks = 0
    total_chars = 0
    oversized: list[str] = []
    undersized: list[str] = []

    for path in paths:
        chunks = chunk_file(path)
        sizes = [len(c.text) for c in chunks]
        total_chunks += len(chunks)
        total_chars += sum(sizes)

        print(f"\n{path.name}")
        print(
            f"  {len(chunks):>3} chunks | "
            f"min {min(sizes):>4} | median {sorted(sizes)[len(sizes) // 2]:>4} | "
            f"max {max(sizes):>4} chars"
        )

        for chunk in chunks:
            if len(chunk.text) > MAX_CHUNK_CHARS:
                oversized.append(f"{chunk.chunk_id} ({len(chunk.text)} chars)")
            if len(chunk.text) < MIN_CHUNK_CHARS:
                undersized.append(f"{chunk.chunk_id} ({len(chunk.text)} chars)")

        for chunk in chunks[: args.show]:
            print(f"\n  --- {chunk.chunk_id} | {chunk.breadcrumb} ---")
            for line in chunk.text.splitlines():
                print(f"  | {line}")

    print(f"\n{'=' * 62}")
    print(f"{len(paths)} document(s), {total_chunks} chunks, {total_chars:,} characters")
    print(f"Average chunk: {total_chars // max(total_chunks, 1)} characters")

    if oversized:
        print(f"\nOver {MAX_CHUNK_CHARS} chars ({len(oversized)}):")
        for item in oversized:
            print(f"  {item}")
    if undersized:
        print(f"\nUnder {MIN_CHUNK_CHARS} chars ({len(undersized)}):")
        for item in undersized:
            print(f"  {item}")
    if not oversized and not undersized:
        print("All chunks within the configured size bounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
