"""Print the analytics summary.

    python -m scripts.analytics_report            # last 30 days
    python -m scripts.analytics_report --days 7
    python -m scripts.analytics_report --json     # raw, as the API returns it
    python -m scripts.analytics_report --reset    # delete all recorded data
"""

from __future__ import annotations

import argparse
import json
import sys

from app.integrations.analytics import db_path, summary


def bar(value: int, total: int, width: int = 28) -> str:
    filled = round(width * value / total) if total else 0
    return "#" * filled + "." * (width - filled)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Delete the analytics database.")
    args = parser.parse_args()

    if args.reset:
        path = db_path()
        removed = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = path.with_name(path.name + suffix)
            if candidate.exists():
                candidate.unlink()
                removed += 1
        print(f"Removed {removed} file(s) for {path.name}." if removed else "Nothing to remove.")
        return 0

    data = summary(days=args.days)

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    total = data["turns"]
    print(f"Riverbend Dental Care — assistant activity, last {data['window_days']} days")
    print("=" * 66)

    if not total:
        print("No conversations recorded yet.")
        print("Run `python -m scripts.chat` or `python -m scripts.seed_demo`.")
        return 0

    print(f"{total} messages across {data['conversations']} conversations\n")

    print("What patients wanted")
    for intent, count in data["intents"].items():
        print(f"  {intent:<10} {bar(count, total)} {count:>4}  ({100 * count / total:.0f}%)")

    inq = data["inquiries"]
    print("\nQuestions")
    print(f"  answered from clinic documents : {inq['answered']} of {inq['total']}"
          f"  ({inq['answer_rate_pct'] or 0}%)")
    print(f"  declined (no answer on file)   : {inq['declined']}")

    if data["knowledge_gaps"]:
        print("\nKnowledge gaps — questions patients asked that the documents don't cover")
        for gap in data["knowledge_gaps"]:
            times = f"x{gap['times']}" if gap["times"] > 1 else "  "
            print(f"  {times:>3}  {gap['question'][:70]}")

    print("\nOutcomes")
    print(f"  appointments booked   : {data['bookings_made']}")
    print(f"  escalated to a human  : {data['escalations']}")
    if data["complaints_by_severity"]:
        order = ["critical", "high", "medium", "low"]
        parts = [f"{s} {data['complaints_by_severity'][s]}" for s in order
                 if s in data["complaints_by_severity"]]
        print(f"  complaints by severity: {', '.join(parts)}")
    if data["complaints_by_category"]:
        parts = [f"{k} {v}" for k, v in data["complaints_by_category"].items()]
        print(f"  complaints by category: {', '.join(parts)}")
    print(f"  mixed-intent messages : {data['mixed_intent_turns']}")

    lat = data["latency_ms"]
    print("\nService")
    print(f"  median response  : {lat['p50']} ms")
    print(f"  95th percentile  : {lat['p95']} ms")
    print(f"  answered free (no model call): {data['fast_path_pct']}%")
    print(f"  failed turns     : {data['failure_rate_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
