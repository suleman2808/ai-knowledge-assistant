"""Streamlit front-end, for hosting on Streamlit Community Cloud.

The project's real interface is the FastAPI app in `app/main.py`, with
the chat UI in `ui/`. This module exists because Streamlit Community
Cloud is the one host that is free with no card, no quota and no sleep —
and it runs Streamlit apps only.

It is deliberately **thin**. Every decision still happens in
`app.graph.run`: the same router, the same three agents, the same
retrieval, the same analytics. Nothing here knows what an intent is or
how grounding works. If this file and the FastAPI app ever disagree
about behaviour, that is a bug in this file.

Two things differ from a local run, both forced by the platform:

- **Secrets arrive via `st.secrets`, not the environment**, so they are
  bridged into `os.environ` before anything imports `app.config`.
- **There is no build step**, so the vector store is built on first run
  rather than baked into an image.
"""

from __future__ import annotations

import os

import streamlit as st

st.set_page_config(
    page_title="Riverbend Dental Care — Assistant",
    page_icon="🦷",
    layout="centered",
)

# --------------------------------------------------------------------------
# Configuration must be in the environment before app.config is imported.
# --------------------------------------------------------------------------

_SETTINGS_FROM_SECRETS = (
    "GROQ_API_KEY",
    "LLM_MODEL",
    "ROUTER_MODEL",
    "RETRIEVAL_MODE",
    "RETRIEVAL_TOP_K",
    "RETRIEVAL_MIN_SCORE",
)

try:
    for _key in _SETTINGS_FROM_SECRETS:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    # No secrets file at all, which is normal when running locally with
    # a .env. app.config falls back to that.
    pass

from app.config import settings  # noqa: E402
from app.graph.build import run  # noqa: E402

INTENT_LABELS = {
    "booking": "Appointments",
    "inquiry": "Information",
    "complaint": "Complaint",
    "other": "General",
}

GREETING = (
    "Hello — I'm the assistant for Riverbend Dental Care. I can answer "
    "questions about our services, prices, hours and policies, book you an "
    "appointment, or pass on a complaint. What can I help with?"
)


@st.cache_resource(show_spinner="Preparing the knowledge base…")
def prepare_knowledge_base() -> dict:
    """Ensure the vector store exists, building it if this is a cold start.

    Cached for the life of the container, so the cost is paid once. On a
    host with a build step this happens at image build time instead; see
    the Dockerfile.
    """
    from app.rag.ingest import ingest
    from app.rag.store import collection_stats

    stats = collection_stats()
    if stats.get("count"):
        return stats

    ingest(rebuild=True, show_progress=False)
    return collection_stats()


def render_sources(turn: dict) -> None:
    """Show where an answer came from, or that it came from nowhere."""
    meta = turn.get("agent_metadata") or {}
    grounded = meta.get("grounded")
    sources = turn.get("sources") or []

    chips = []
    if INTENT_LABELS.get(turn.get("intent")):
        chips.append(INTENT_LABELS[turn["intent"]])
    if grounded is True:
        chips.append("From clinic documents")
    elif grounded is False:
        chips.append("Not in our documents")
    if turn.get("latency_ms"):
        chips.append(f"{turn['latency_ms'] / 1000:.1f}s")

    if chips:
        st.caption(" · ".join(chips))

    if sources:
        with st.expander(f"{len(sources)} source{'s' if len(sources) > 1 else ''}"):
            for source in sources:
                st.markdown(
                    f"**{source['breadcrumb']}**  \n"
                    f"`{source['source']}` · similarity {source['score']}"
                )


def main() -> None:
    st.title("🦷 Riverbend Dental Care")
    st.caption(
        "2418 SE Hawthorne Blvd, Portland · (503) 555-0142 — "
        "an AI assistant that answers only from the clinic's own documents."
    )

    if not settings.llm_configured:
        st.error(
            "The assistant has no API key. Add `GROQ_API_KEY` under "
            "**Settings → Secrets** for this app, then rerun."
        )
        st.stop()

    stats = prepare_knowledge_base()
    if not stats.get("count"):
        st.error("The knowledge base could not be built. Check the logs.")
        st.stop()

    with st.sidebar:
        st.subheader("Try asking")
        for suggestion in (
            "How much is a root canal on a molar?",
            "Do you take Delta Dental?",
            "What should I avoid after an extraction?",
            "I waited 40 minutes and nobody apologised",
        ):
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending = suggestion

        st.divider()
        st.caption(
            f"{stats['count']} chunks from {len(stats['sources'])} documents  \n"
            f"Agent model: `{settings.llm_model}`  \n"
            f"Router: `{settings.router_model}`  \n"
            f"Retrieval: `{settings.retrieval_mode}`"
        )
        st.caption(
            "[Source on GitHub]"
            "(https://github.com/suleman2808/ai-knowledge-assistant)"
        )
        st.info(
            "Bookings here use an in-memory calendar — the Google Calendar "
            "token is deliberately not committed, so a deployed instance "
            "cannot write to a real diary.",
            icon="📅",
        )

    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.turns = []

    with st.chat_message("assistant"):
        st.markdown(GREETING)

    for message, turn in zip(st.session_state.history[::2], st.session_state.turns):
        with st.chat_message("user"):
            st.markdown(message["content"])
        with st.chat_message("assistant"):
            st.markdown(turn.get("answer", ""))
            render_sources(turn)

    question = st.chat_input("Ask about treatments, prices, hours — or book an appointment")
    if not question:
        question = st.session_state.pop("pending", None)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the clinic's documents…"):
            try:
                turn = run(
                    question,
                    history=st.session_state.history,
                    session_id=st.session_state.get("session_id", "streamlit"),
                )
            except Exception as exc:  # the UI must not show a traceback
                st.error(
                    "Something went wrong at our end. Please try again, or "
                    "call the clinic on (503) 555-0142."
                )
                st.caption(f"({type(exc).__name__})")
                return

        st.markdown(turn.get("answer", ""))
        render_sources(turn)

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append(
        {"role": "assistant", "content": turn.get("answer", "")}
    )
    st.session_state.turns.append(turn)


if __name__ == "__main__":
    main()
