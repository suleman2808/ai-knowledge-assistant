"""Admin dashboard, for the Streamlit deployment.

The real dashboard is the one served by FastAPI at `/admin`. This is the
same data through Streamlit's widgets, and it exists for one reason: the
free hosting that runs this project runs Streamlit apps only, so without
it the deployed link shows the chat and nothing else. The proposal
promises a URL that can be opened and shared, and a dashboard nobody can
reach is not that.

Everything here reads through `app.integrations.analytics`, the same
functions the FastAPI routes call. No queries are written twice.

Nothing here uses `st.dataframe` or `st.bar_chart`. Both are backed by
pandas, whose compiled extension is blocked outright by Windows Smart App
Control on the development machine — the same Application Control policy
that forced the embedding runtime off PyTorch earlier in this project. It
would work on the Linux host, but a dashboard that cannot be opened by
the person demonstrating it is not much use, and asking anyone to switch
off an OS security feature is not an acceptable install step. Tables are
rendered as markdown instead, which also keeps the page honest about how
little machinery it needs.

Two honest differences from the FastAPI version:

- **Storage is ephemeral here.** Streamlit Cloud rebuilds the container
  on each deploy and does not keep a disk, so this dashboard shows
  conversations from the current container's lifetime, not forever.
- **Rate limiting is per-session**, not per-IP, because Streamlit does
  not expose the client address. It slows a person guessing by hand; it
  would not stop a determined script, which is why the password matters
  more here than in the FastAPI app.
"""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(page_title="Admin dashboard — Riverbend", page_icon="🔒", layout="wide")

# Secrets reach the app through st.secrets, not the environment, and
# app.config reads the environment. Bridge before importing it.
try:
    for _key in ("GROQ_API_KEY", "ADMIN_PASSWORD"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass

from app import admin_auth  # noqa: E402
from app.integrations import analytics  # noqa: E402

MAX_ATTEMPTS = 5


def sign_in() -> bool:
    """Show the password gate. Returns True once signed in."""
    if st.session_state.get("admin_ok"):
        return True

    st.title("🔒 Admin dashboard")

    if not admin_auth.is_enabled():
        st.error(
            "No password is configured for this deployment, so the dashboard "
            "is switched off. Set `ADMIN_PASSWORD` in the app's secrets."
        )
        st.caption(
            "It refuses to open rather than defaulting to public, because the "
            "failure people regret is the one where protection was never "
            "switched on."
        )
        return False

    st.caption("Conversations, bookings, complaints and customer records.")

    attempts = st.session_state.get("admin_attempts", 0)
    if attempts >= MAX_ATTEMPTS:
        st.error("Too many attempts. Reload the page to try again.")
        return False

    with st.form("sign_in"):
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if admin_auth.check_password(password):
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.session_state.admin_attempts = attempts + 1
                st.error(f"Incorrect password. {MAX_ATTEMPTS - attempts - 1} left.")
    return False


def table(rows: list[dict[str, object]], columns: dict[str, str]) -> None:
    """Render rows as a markdown table.

    Deliberately not `st.dataframe`: see the module docstring.
    """
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, divider]
    for row in rows:
        cells = []
        for key in columns.values():
            value = row.get(key)
            if isinstance(value, bool):
                value = "yes" if value else "no"
            text = str(value if value not in (None, "") else "—")
            # Escape pipes, or one stray character breaks the whole table.
            cells.append(text.replace("|", r"\|").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join(lines))


def bars(counts: dict[str, int]) -> None:
    """A horizontal bar chart made of text, for the same reason."""
    total = sum(counts.values()) or 1
    width = 32
    lines = []
    for label, count in counts.items():
        filled = round(width * count / total)
        lines.append(
            f"`{label:<10}` `{'█' * filled}{'·' * (width - filled)}` "
            f"**{count}** ({100 * count / total:.0f}%)"
        )
    # Two trailing spaces is a markdown line break, so each bar gets its
    # own line without the gaps a paragraph break would add.
    st.markdown("  \n".join(lines))


def overview() -> None:
    data = analytics.summary(days=30)
    if not data["turns"]:
        st.info("No conversations recorded yet. Use the chat and come back.")
        return

    inquiries = data["inquiries"]
    columns = st.columns(6)
    figures = [
        ("Messages", data["turns"]),
        ("Conversations", data["conversations"]),
        ("Answered", f"{inquiries['answer_rate_pct']}%"
            if inquiries["answer_rate_pct"] is not None else "—"),
        ("Bookings", data["bookings_made"]),
        ("Escalated", data["escalations"]),
        ("Median reply",
         f"{data['latency_ms']['p50'] / 1000:.1f}s" if data["latency_ms"]["p50"] else "—"),
    ]
    for column, (label, value) in zip(columns, figures):
        column.metric(label, value)

    st.subheader("What customers wanted")
    bars(data["intents"])

    st.subheader("Knowledge gaps")
    st.caption(
        "Questions the assistant declined — things customers want to know "
        "that the documents do not cover. Each one is a page worth writing."
    )
    if data["knowledge_gaps"]:
        table(data["knowledge_gaps"], {"Question": "question", "Times asked": "times"})
    else:
        st.success("None — everything asked was answerable from the documents.")


def conversations() -> None:
    rows = analytics.conversations(limit=100)
    if not rows:
        st.info("No conversations yet.")
        return

    table(
        [{**row, "handled_by": ", ".join(row["intents"])} for row in rows],
        {
            "Started": "started_at",
            "Session": "session_id",
            "Turns": "turns",
            "Handled by": "handled_by",
            "Bookings": "bookings",
            "Escalated": "escalations",
            "Declined": "declined",
        },
    )

    st.subheader("Transcript and retrieval trace")
    st.caption(
        "A grounded answer is only a claim until you can see which sections "
        "produced it. Pick a conversation to audit it."
    )
    chosen = st.selectbox(
        "Conversation",
        [row["session_id"] for row in rows],
        format_func=lambda s: f"{s} ({next(r['turns'] for r in rows if r['session_id'] == s)} turns)",
    )

    for turn in analytics.conversation(chosen):
        with st.container(border=True):
            st.markdown(f"**Customer:** {turn['message']}")
            st.markdown(turn["answer"] or "_(answer not recorded)_")

            badges = [turn["intent"], f"{turn['latency_ms']} ms"]
            if turn["grounded"] is True:
                badges.insert(1, "from documents")
            elif turn["grounded"] is False:
                badges.insert(1, "declined")
            if not turn["success"]:
                badges.append("failed")
            st.caption(" · ".join(badges))

            if turn["search_query"] and turn["search_query"] != turn["message"]:
                st.caption(f"Searched as: _{turn['search_query']}_")

            if turn["sources"]:
                with st.expander(f"Answered from {len(turn['sources'])} section(s)"):
                    for source in turn["sources"]:
                        st.markdown(
                            f"**{source['breadcrumb']}** — `{source['source']}` "
                            f"(similarity {source['score']})"
                        )
            elif turn["grounded"] is False:
                st.caption("Answered from: nothing — the documents did not cover it.")


def records(loader, empty: str, columns: dict[str, str]) -> None:
    """Render one of the straightforward record tables."""
    rows = loader()
    if not rows:
        st.info(empty)
        return
    table(rows, columns)


def main() -> None:
    if not sign_in():
        return

    header, action = st.columns([5, 1])
    header.title("Admin dashboard")
    if action.button("Sign out", use_container_width=True):
        st.session_state.admin_ok = False
        st.rerun()

    st.caption(
        "This deployment keeps data only for the lifetime of its container, "
        "so these records reset when the app restarts."
    )

    tabs = st.tabs(["Overview", "Conversations", "Bookings", "Complaints", "Customers"])

    with tabs[0]:
        overview()
    with tabs[1]:
        conversations()
    with tabs[2]:
        records(
            analytics.bookings,
            "No bookings yet.",
            {"Appointment": "starts_at", "Customer": "patient_name", "Phone": "phone",
             "Service": "service", "Calendar": "calendar_backend"},
        )
    with tabs[3]:
        records(
            analytics.complaints,
            "No complaints logged.",
            {"Received": "received_at", "Reference": "reference", "Category": "category",
             "Severity": "severity", "Summary": "summary", "Escalated": "escalated"},
        )
    with tabs[4]:
        records(
            analytics.customers,
            "No returning customers yet.",
            {"Name": "name", "Phone": "phone", "Bookings": "visit_count",
             "First seen": "first_seen", "Last seen": "last_seen"},
        )


main()
