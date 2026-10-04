"""Run one agent directly, without the graph or the API.

    python -m scripts.try_agent inquiry "how much is a root canal"
    python -m scripts.try_agent inquiry --suite
    python -m scripts.try_agent booking "can I come in next Tuesday at 2pm"
    python -m scripts.try_agent complaint "I was charged twice and nobody called back"

The `--suite` flag runs a fixed set of probes for that agent, including
cases the agent is expected to refuse.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

INQUIRY_SUITE = [
    # Should answer, grounded.
    "how much is a full blood count",
    "do you accept cigna",
    "do I need to fast for a lipid profile",
    "when will my urine culture be ready",
    "do you come to my house",
    # Should refuse: related material exists, but does not answer this.
    "what time does the cinema open",
    "how much does an MRI cost",
    "what is your laboratory's annual revenue",
    "who is the current president",
    # The refusal that matters most in this domain.
    "my haemoglobin is 9.2, is that bad?",
]

BOOKING_SUITE = [
    "can I book a full blood count next Tuesday at 8am, my name is Sarah Chen, 503-555-0180",
    "I need an appointment",
    "book me in for a glucose tolerance test tomorrow morning",
]

COMPLAINT_SUITE = [
    "I waited 45 minutes past my appointment time and nobody apologised",
    "I was charged twice for the same panel and three phone calls have gone unanswered",
    "The phlebotomist tried four times and my arm is badly bruised. This is unacceptable.",
    "Your receptionist was lovely, just wanted to say thanks",
]


def show(response, query: str) -> None:  # noqa: ANN001
    print(f"\n{'=' * 70}")
    print(f"Q: {query}")
    print(f"{'-' * 70}")
    print(response.answer)
    print(f"{'-' * 70}")
    flags = [f"agent={response.agent}", f"success={response.success}"]
    if response.needs_followup:
        flags.append("needs_followup=True")
    print("  " + "  ".join(flags))
    if response.sources:
        print("  sources:")
        for source in response.sources:
            print(f"    {source['score']}  {source['breadcrumb']}")
    if response.metadata:
        print(f"  metadata: {json.dumps(response.metadata, default=str)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agent", choices=["inquiry", "booking", "complaint"])
    parser.add_argument("query", nargs="*")
    parser.add_argument("--suite", action="store_true", help="Run the standard probes.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    if args.agent == "inquiry":
        from app.agents.inquiry import answer_inquiry as run

        suite = INQUIRY_SUITE
    elif args.agent == "booking":
        from app.agents.booking import handle_booking as run

        suite = BOOKING_SUITE
    else:
        from app.agents.complaint import handle_complaint as run

        suite = COMPLAINT_SUITE

    queries = suite if args.suite else [" ".join(args.query)]
    if not queries or not queries[0]:
        parser.print_help()
        return 1

    for query in queries:
        show(run(query), query)
    return 0


if __name__ == "__main__":
    sys.exit(main())
