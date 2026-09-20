"""Talk to the assistant through the graph.

    python -m scripts.chat                    # interactive session
    python -m scripts.chat "how much is a crown"
    python -m scripts.chat --routing          # routing accuracy probe
    python -m scripts.chat --scenario booking # a scripted conversation

Interactive mode keeps history, so multi-turn flows such as completing a
booking can be exercised properly.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from app.graph.build import run

# Messages paired with the intent a human would assign. Used to measure
# routing accuracy, including the cases designed to be hard.
ROUTING_PROBES: list[tuple[str, str]] = [
    # Unambiguous
    ("how much is a crown", "inquiry"),
    ("what time do you open on Saturday", "inquiry"),
    ("can I book a cleaning for next Tuesday", "booking"),
    ("I need to move my appointment to Thursday", "booking"),
    ("I waited an hour and nobody said sorry", "complaint"),
    ("hi", "other"),
    ("thanks, that's all", "other"),
    # Policy question vs. diary change — the classic confusion
    ("what is your cancellation policy", "inquiry"),
    ("I need to cancel Tuesday's appointment", "booking"),
    # Treatment mentioned, but not a booking
    ("how long does a root canal take", "inquiry"),
    ("do I need a crown after a root canal", "inquiry"),
    # Angry, but still a question
    ("why on earth are your crowns so expensive", "inquiry"),
    # Grievance that ends in a request
    ("my filling fell out after two weeks, this is ridiculous", "complaint"),
    # Billing: complaint vs. inquiry
    ("do you take Delta Dental", "inquiry"),
    ("I've been charged twice and nobody has called me back", "complaint"),
    # Mixed intent: the complaint must win, because an unrecorded
    # complaint is lost whereas an unmade booking gets asked for again.
    ("I waited 40 minutes last time and I'm furious, anyway can I book a cleaning Tuesday", "complaint"),
    # Out of scope for a dental practice entirely.
    ("what time does the cinema open", "other"),
    ("can you help me renew my car insurance", "other"),
]

SCENARIOS: dict[str, list[str]] = {
    "booking": [
        "hi",
        "I'd like to book a cleaning",
        "next Tuesday afternoon works",
        "Sarah Chen, 503-555-0180",
        "thanks",
    ],
    "complaint": [
        "I was charged twice for the same filling and nobody has called me back",
        "what is your complaints procedure",
    ],
    "mixed": [
        "I waited 40 minutes last time and I'm furious, anyway can I book a cleaning Tuesday",
    ],
}


def show(turn: dict, elapsed: float) -> None:
    intent = turn.get("intent", "?")
    secondary = turn.get("secondary_intent")
    badge = f"{intent}" + (f" (+{secondary})" if secondary else "")

    print(f"\n  [{badge}  conf={turn.get('confidence', 0):.2f}  "
          f"via={turn.get('routed_by', '?')}  {elapsed * 1000:.0f}ms]")
    if turn.get("routing_reason"):
        print(f"  reason: {turn['routing_reason']}")
    print()
    print(turn.get("answer", ""))
    if turn.get("sources"):
        print("\n  sources:")
        for source in turn["sources"]:
            print(f"    {source['score']}  {source['breadcrumb']}")


def routing_probe() -> int:
    """Measure routing accuracy against hand-labelled expectations."""
    print("Routing accuracy\n")
    correct = 0
    wrong: list[str] = []

    for message, expected in ROUTING_PROBES:
        started = time.perf_counter()
        turn = run(message)
        elapsed = (time.perf_counter() - started) * 1000
        actual = turn.get("intent", "?")
        secondary = turn.get("secondary_intent")

        ok = actual == expected
        correct += ok
        mark = "ok  " if ok else "MISS"
        extra = f" +{secondary}" if secondary else ""
        print(
            f"  {mark}  {actual:<9}{extra:<12} "
            f"conf={turn.get('confidence', 0):.2f} {elapsed:>6.0f}ms  {message[:56]}"
        )
        if not ok:
            wrong.append(f"{message[:60]} -> {actual}, expected {expected}")

    total = len(ROUTING_PROBES)
    print(f"\n{'=' * 70}")
    print(f"{correct}/{total} correct ({correct / total:.0%})")
    if wrong:
        print("\nMisroutes:")
        for item in wrong:
            print(f"  {item}")
    return 0


def scenario(name: str) -> int:
    """Replay a scripted multi-turn conversation."""
    messages = SCENARIOS.get(name)
    if not messages:
        print(f"No scenario {name!r}. Available: {', '.join(SCENARIOS)}")
        return 1

    history: list[dict[str, str]] = []
    for message in messages:
        print(f"\n{'=' * 70}\nYou: {message}")
        started = time.perf_counter()
        turn = run(message, history=history, session_id=f"scenario-{name}")
        show(turn, time.perf_counter() - started)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": turn.get("answer", "")})
    return 0


def interactive() -> int:
    print("Riverbend Dental Care assistant. Ctrl-C or 'quit' to exit.\n")
    history: list[dict[str, str]] = []

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            return 0

        started = time.perf_counter()
        turn = run(message, history=history, session_id="cli")
        show(turn, time.perf_counter() - started)
        print()

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": turn.get("answer", "")})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("message", nargs="*")
    parser.add_argument("--routing", action="store_true", help="Routing accuracy probe.")
    parser.add_argument("--scenario", help="Replay a scripted conversation.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    if args.routing:
        return routing_probe()
    if args.scenario:
        return scenario(args.scenario)
    if args.message:
        message = " ".join(args.message)
        started = time.perf_counter()
        show(run(message), time.perf_counter() - started)
        return 0
    return interactive()


if __name__ == "__main__":
    sys.exit(main())
