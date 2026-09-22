"""Populate analytics with realistic demo traffic.

    python -m scripts.seed_demo            # run every conversation below
    python -m scripts.seed_demo --reset    # clear analytics first

Every message goes through the real graph — real routing, real
retrieval, real model calls — so the resulting numbers are what the
system genuinely does, not invented figures. Takes a few minutes and
roughly 60 model calls, well inside Groq's free tier.

Bookings are forced onto the in-memory calendar. Without that, seeding
would write a dozen fictional patients into whatever real Google
Calendar is connected.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from app.graph.build import run
from app.integrations.calendar import InMemoryCalendar, reset_calendar

# Conversations as a patient would actually write them: typos, fragments,
# follow-ups. Each inner list is one session.
CONVERSATIONS: list[list[str]] = [
    ["hi", "how much is a filling", "is that for one surface?", "thanks"],
    ["do you take cigna"],
    ["what time are you open till on wednesday"],
    ["Can I book a cleaning next Monday morning? Priya Nair, 503-555-0147"],
    ["I need an appointment", "check-up", "thursday afternoon",
     "Tom Becker 503 555 0133"],
    ["my tooth got knocked out playing football what do i do"],
    ["how long does whitening last"],
    ["I was charged twice for my crown and no one has returned my calls. Very poor."],
    ["waited 35 mins past my appointment yesterday, not happy"],
    ["The dentist was really rough during my extraction and I'm still in a lot "
     "of pain four days later. I'm considering reporting this."],
    ["is parking free"],
    ["do you see kids", "what age from"],
    ["whats the cancellation fee if I cancel tomorrow"],
    ["I need to cancel my appointment on Friday"],
    ["do you offer botox"],
    ["is there wifi in the waiting room"],
    ["can I bring my dog to my appointment"],
    ["what brand of implants do you use"],
    ["do you do payment plans", "how long can I spread it over"],
    ["who is the best dentist for nervous patients"],
    ["can I get my teeth cleaned while pregnant"],
    ["what should I not eat after a root canal"],
    ["receptionist Keisha was brilliant today, thank you"],
    ["I waited 40 minutes last time and I'm furious, anyway can I book a "
     "cleaning Tuesday"],
    ["do you accept medicaid"],
    ["how much are veneers"],
    ["what time does the cinema down the road open"],
    ["good morning", "bye"],
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="Clear analytics first.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")

    if args.reset:
        from app.integrations.analytics import db_path

        path = db_path()
        for suffix in ("", "-wal", "-shm"):
            candidate = path.with_name(path.name + suffix)
            if candidate.exists():
                candidate.unlink()
        print(f"Cleared {path.name}")

    # Never write demo patients into a real calendar.
    reset_calendar(InMemoryCalendar())

    total = sum(len(c) for c in CONVERSATIONS)
    print(f"Seeding {total} messages across {len(CONVERSATIONS)} conversations\n")

    done = 0
    started = time.perf_counter()
    for index, conversation in enumerate(CONVERSATIONS, start=1):
        history: list[dict[str, str]] = []
        session = f"demo-{index:03d}"
        for message in conversation:
            turn = run(message, history=history, session_id=session)
            done += 1
            print(f"  [{done:>2}/{total}] {turn.get('intent', '?'):<9} {message[:58]}")
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": turn.get("answer", "")})

    reset_calendar(None)
    print(f"\nDone in {time.perf_counter() - started:.0f}s. "
          f"Run `python -m scripts.analytics_report` to see the results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
