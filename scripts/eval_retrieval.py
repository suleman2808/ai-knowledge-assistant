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
    # Brand and proper-noun lookups: the weakness vector search has, and
    # the reason BM25 was added.
    ("do you take cigna", "Accepted Insurance Providers"),
    ("are you in network with aetna", "Accepted Insurance Providers"),
    ("do you accept providence", "Accepted Insurance Providers"),
    ("do you take oregon health plan", "Accepted Insurance Providers"),
    ("tell me about dr reyes", "Samuel Reyes"),
    ("who is fiona", "Fiona Adeyemi"),
    ("where is the gresham branch", "Gresham"),
    ("which bus goes to the hawthorne branch", "Parking and Transit"),
    # Prices and turnaround — the commonest real questions
    ("how much is a full blood count", "Haematology"),
    ("what does a lipid profile cost", "Biochemistry"),
    ("how much is the basic health check", "Panels"),
    ("how much does hba1c cost", "Biochemistry"),
    ("when will my urine culture be ready", "Microbiology"),
    ("how long does a blood culture take", "Microbiology"),
    # Preparation — where getting it wrong means a repeat sample
    ("do I need to fast for a lipid profile", "Fasting"),
    ("can I drink water while fasting", "Fasting"),
    ("is black coffee allowed before a blood test", "Fasting"),
    ("should I stop my medication before the test", "Medication"),
    ("how do I collect a urine sample", "Before Specific Tests"),
    ("what happens during a glucose tolerance test", "Before Specific Tests"),
    ("why am I bruising after the blood draw", "After a Blood Draw"),
    # Reports and results
    ("can you tell me what my result means", "Cannot Interpret"),
    ("can I get my report over the phone", "How You Receive It"),
    ("my friend is collecting my report", "Collecting a Printed Report"),
    ("what is a critical result", "Critical Results"),
    ("my result is outside the reference range", "Reference Ranges"),
    # Logistics and policy
    # The FAQ answers this in full, which is a better hit than the
    # price table it was first labelled against.
    ("do you come to my house", "Do you come to my house"),
    ("do I need an appointment", "Do I Need an Appointment"),
    ("are you open on sunday", "Hawthorne"),
    ("how do I make a complaint", "Complaints"),
    ("will my employer see my results", "Privacy and Records"),
    ("is hiv testing confidential", "HIV"),
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
