"""Measure retrieval quality against hand-labelled questions.

    python -m scripts.eval_retrieval                 # current mode
    python -m scripts.eval_retrieval --compare       # vector vs hybrid

Each question is paired with the section that should answer it. The
metrics are the standard ones:

- **hit@1** — the right section came back first.
- **hit@k** — the right section is somewhere in the context the agent
  sees. This is the one that decides whether an answer is possible.
- **MRR** — mean reciprocal rank; rewards ranking it higher.

Includes every retrieval miss observed while building the project, so a
regression on any of them shows up here.
"""

from __future__ import annotations

import argparse
import logging
import sys

from app.config import settings
from app.rag.retriever import retrieve

# (question, a substring of the breadcrumb that should be retrieved)
LABELLED: list[tuple[str, str]] = [
    # Observed misses, kept as regression cases
    ("do you take cigna", "Accepted Insurance Providers"),                    # step 7: brand name
    ("what if I cancel late", "Cancellation Policy"),                         # step 2: "late" collision
    ("what should I avoid after having a tooth out", "After an Extraction"),  # step 4
    ("whats the cancellation fee if I cancel tomorrow", "Cancellation Policy"),
    # Brand and proper-noun lookups
    ("do you accept metlife", "Accepted Insurance Providers"),
    ("is carecredit available", "Payment Plans"),
    ("are you in network with aetna", "Accepted Insurance Providers"),
    ("do you take oregon health plan", "Accepted Insurance Providers"),
    ("tell me about dr reyes", "Samuel Reyes"),
    ("who is fiona", "Fiona Adeyemi"),
    ("which bus goes to the clinic", "Parking and Transit"),
    # Ordinary questions
    ("how much is a root canal on a molar", "Restorative Treatment"),
    ("what time do you close on friday", "Opening Hours"),
    ("my tooth was knocked out", "First Aid"),
    ("can I be treated while pregnant", "pregnant"),
    ("do you see nervous patients", "nervous patients"),
    ("is there parking", "Parking and Transit"),
    ("how long do fillings last", "How long do fillings last"),
    ("can I pay in instalments", "Payment Plans"),
    ("what happens if I miss an appointment without telling you", "Cancellation Policy"),
    ("how do I make a complaint", "Complaints"),
    ("do you do teeth whitening", "Cosmetic Treatment"),
    ("what age do you see children from", "Children"),
    ("my gums bleed when I brush", "gums bleed"),
    ("dry socket", "After an Extraction"),
    ("how much is an implant", "Dentures and Implants"),
    ("are you open on christmas eve", "Holiday Closures"),
    ("can I have a chaperone", "Chaperones"),
]


def evaluate(mode: str) -> dict:
    hits_at_1 = 0
    hits_at_k = 0
    reciprocal = 0.0
    misses: list[str] = []

    for question, expected in LABELLED:
        result = retrieve(question, mode=mode)
        crumbs = [c.breadcrumb for c in result.chunks]
        rank = next(
            (i for i, c in enumerate(crumbs, start=1) if expected.lower() in c.lower()),
            None,
        )
        if rank == 1:
            hits_at_1 += 1
        if rank:
            hits_at_k += 1
            reciprocal += 1 / rank
        else:
            top = crumbs[0].split(" > ")[-1] if crumbs else f"({result.status.value})"
            misses.append(f"{question[:48]:<50} got: {top}")

    n = len(LABELLED)
    return {
        "mode": mode,
        "n": n,
        "hit@1": hits_at_1 / n,
        "hit@k": hits_at_k / n,
        "mrr": reciprocal / n,
        "misses": misses,
    }


def report(result: dict) -> None:
    print(f"\n{result['mode']}  (k={settings.retrieval_top_k}, n={result['n']})")
    print(f"  hit@1  {result['hit@1']:.0%}")
    print(f"  hit@k  {result['hit@k']:.0%}")
    print(f"  MRR    {result['mrr']:.3f}")
    if result["misses"]:
        print("  misses:")
        for miss in result["misses"]:
            print(f"    {miss}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", action="store_true", help="Vector-only vs hybrid.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    modes = ["vector", "hybrid"] if args.compare else [settings.retrieval_mode]
    results = [evaluate(mode) for mode in modes]
    for result in results:
        report(result)

    if len(results) == 2:
        a, b = results
        print(f"\nhybrid vs vector:  hit@1 {b['hit@1'] - a['hit@1']:+.0%}   "
              f"hit@k {b['hit@k'] - a['hit@k']:+.0%}   MRR {b['mrr'] - a['mrr']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
