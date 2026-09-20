"""Query the knowledge base from the command line.

Useful for judging retrieval quality and for calibrating the relevance
threshold before any agent is involved.

    python -m scripts.search "how much is a root canal"
    python -m scripts.search "do you take Delta Dental" --full
    python -m scripts.search --calibrate
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings
from app.rag.retriever import RetrievalStatus, retrieve

# Questions the clinic's documents genuinely answer, paired with questions
# they do not. A good threshold separates these two groups cleanly.
IN_SCOPE = [
    "how much does a root canal cost",
    "do you accept Delta Dental insurance",
    "what are your opening hours on Saturday",
    "my tooth was knocked out, what should I do",
    "what happens if I cancel my appointment late",
    "can I be treated while pregnant",
    "who do I speak to about a billing problem",
    "how long do fillings last",
    "is there parking at the clinic",
    "what should I avoid after an extraction",
    "do you see nervous patients",
    "can I pay in instalments",
]

OUT_OF_SCOPE = [
    "what is the capital of France",
    "can you help me renew my car insurance",
    "do you sell mobile phones",
    "what time does the cinema open",
    "how do I reset my email password",
    "book me a table for dinner tonight",
]


def show(query: str, *, full: bool) -> None:
    result = retrieve(query)

    print(f"\nQ: {query}")
    print(f"   status={result.status.value}  best={result.best_score:.3f}")
    if result.detail:
        print(f"   {result.detail}")

    if not result.chunks:
        return

    for i, chunk in enumerate(result.chunks, start=1):
        print(f"\n   [{i}] {chunk.breadcrumb}  ({chunk.score:.3f})")
        body = chunk.text if full else chunk.text[:220].rstrip()
        for line in body.splitlines():
            print(f"       {line}")
        if not full and len(chunk.text) > 220:
            print("       ...")


def calibrate() -> int:
    """Print the score distribution for in-scope and out-of-scope queries.

    The threshold should sit in the gap between the two groups. Choosing it
    by inspection beats guessing, and the numbers are worth keeping: they
    are the evidence behind the configured value.
    """
    print("Calibrating the relevance threshold")
    print(f"Current RETRIEVAL_MIN_SCORE = {settings.retrieval_min_score}\n")

    def scores(queries: list[str], label: str) -> list[float]:
        collected: list[float] = []
        print(f"{label}")
        for query in queries:
            # Threshold of 0 so every query reports its true best score.
            result = retrieve(query, min_score=0.0)
            collected.append(result.best_score)
            top = result.chunks[0].breadcrumb if result.chunks else "-"
            print(f"  {result.best_score:.3f}  {query[:44]:<46} {top}")
        return collected

    in_scores = scores(IN_SCOPE, "In scope (the documents answer these):")
    print()
    out_scores = scores(OUT_OF_SCOPE, "Out of scope (they do not):")

    lowest_in = min(in_scores)
    highest_out = max(out_scores)

    print(f"\n{'=' * 66}")
    print(f"Lowest in-scope score : {lowest_in:.3f}")
    print(f"Highest out-of-scope  : {highest_out:.3f}")

    if lowest_in > highest_out:
        midpoint = (lowest_in + highest_out) / 2
        print(f"Clean separation. A threshold near {midpoint:.2f} splits them.")
        print(f"Configured value {settings.retrieval_min_score} is "
              f"{'inside' if highest_out < settings.retrieval_min_score < lowest_in else 'OUTSIDE'} that gap.")
    else:
        print("The groups overlap: no single threshold separates them cleanly.")
        print("This is the usual argument for adding a reranker or hybrid search.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="The question to ask.")
    parser.add_argument("--full", action="store_true", help="Print whole chunks.")
    parser.add_argument(
        "--calibrate", action="store_true", help="Show the score distribution."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    if args.calibrate:
        return calibrate()

    if not args.query:
        parser.print_help()
        return 1

    query = " ".join(args.query)
    result = retrieve(query)
    show(query, full=args.full)

    return 0 if result.status is not RetrievalStatus.UNAVAILABLE else 1


if __name__ == "__main__":
    sys.exit(main())
