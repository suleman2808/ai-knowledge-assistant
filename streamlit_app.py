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
import uuid

import streamlit as st

st.set_page_config(
    page_title="Riverbend Diagnostics - AI Assistant",
    page_icon="🔬",
    layout="centered",
)

# --------------------------------------------------------------------------
# Configuration must be in the environment before app.config is imported.
# --------------------------------------------------------------------------

_SETTINGS_FROM_SECRETS = (
    "GROQ_API_KEY",
    "ADMIN_PASSWORD",
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
    "Hello, I'm the assistant for Riverbend Diagnostics. I can answer "
    "questions about our services, prices, hours and policies, book you an "
    "appointment, or pass on a complaint. What can I help with?"
)


# ---------------------------------------------------------------------------
# Making this look like the product
# ---------------------------------------------------------------------------

# Streamlit's config.toml sets the palette; everything that palette cannot
# reach is set here. The targets are `data-testid` attributes, which are
# Streamlit's own public hooks and survive its releases far better than
# the generated class names do.
#
# The aim is not a pixel copy of ui/styles.css - Streamlit owns its
# layout - but a visitor moving between the two should feel they are
# looking at one product: same rail, same quiet cards, same rounded
# composer, same typeface.
THEME = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

:root {
  --rb-rail:  #171614;
  --rb-line:  #35332f;
  --rb-faint: #827d75;
  --rb-muted: #a39e96;
  --rb-brand: #6fb3b8;
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
  font-family: Inter, ui-sans-serif, -apple-system, "Segoe UI", Roboto, sans-serif;
}

/* The rail. Streamlit paints the sidebar in secondaryBackgroundColor,
   which the chat bubbles also use, so it is darkened here instead. */
[data-testid="stSidebar"] {
  background: var(--rb-rail);
  border-right: 1px solid var(--rb-line);
}
[data-testid="stSidebar"] h3 {
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--rb-faint) !important;
  margin-bottom: .4rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
  font-size: 12.8px;
  line-height: 1.55;
  color: var(--rb-muted);
}

/* Suggestion buttons, as the quiet cards the rail uses. */
[data-testid="stSidebar"] .stButton button {
  text-align: left;
  justify-content: flex-start;
  border: 1px solid var(--rb-line);
  border-radius: 10px;
  background: #232220;
  color: var(--rb-muted);
  font-size: 12.8px;
  font-weight: 400;
  line-height: 1.4;
  padding: .55rem .7rem;
  transition: border-color .15s ease, color .15s ease;
}
[data-testid="stSidebar"] .stButton button:hover {
  border-color: var(--rb-brand);
  color: var(--rb-brand);
}

/* Heading and strapline, sized like the chat page rather than a report. */
[data-testid="stMainBlockContainer"] h1 {
  font-size: 28px !important;
  font-weight: 600 !important;
  letter-spacing: -.02em;
}
[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] p {
  font-size: 12.5px;
  color: var(--rb-faint);
}

/* The composer: one rounded field with a circular send button. */
[data-testid="stChatInput"] {
  border: 1px solid var(--rb-line);
  border-radius: 24px;
  background: #232220;
}
[data-testid="stChatInput"]:focus-within { border-color: var(--rb-brand); }
[data-testid="stChatInputSubmitButton"] {
  border-radius: 50%;
  background: var(--rb-brand);
  color: #10262a;
}

/* Answers read as prose, the patient's words as a card - the same
   asymmetry the chat page uses. */
[data-testid="stChatMessage"] { background: transparent; padding-left: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  background: #2b2a27;
  border: 1px solid #2c2b28;
  border-radius: 18px 18px 5px 18px;
  padding: .25rem .9rem;
}

/* Streamlit's own fork and share chrome is not part of the demo. */
[data-testid="stToolbarActions"] { display: none; }
[data-testid="stDecoration"] { display: none; }
/* Streamlit's own page switcher is replaced by the brand block and the
   links at the foot of the rail, as the chat page has them. */
[data-testid="stSidebarNav"] { display: none; }

.rb-brand {
  display: flex; align-items: center; gap: 11px;
  padding: 2px 0 14px;
  border-bottom: 1px solid var(--rb-line);
  margin-bottom: 14px;
}
.rb-mark { flex: 0 0 32px; width: 32px; height: 32px; }
.rb-mark svg { width: 100%; height: 100%; display: block; }
.rb-brand-text { display: flex; flex-direction: column; line-height: 1.25; }
.rb-brand-text strong {
  font-size: 14.5px; font-weight: 600; color: #ece9e4;
}
.rb-brand-text em {
  font-style: normal; font-size: 12px; color: var(--rb-faint);
}

/* The header bar: the assistant's name, then where to find the lab. */
.rb-head { display: flex; flex-direction: column; line-height: 1.3; }
.rb-head strong { font-size: 15.5px; font-weight: 600; color: #ece9e4; }
.rb-head span { font-size: 12.5px; color: var(--rb-faint); }

/* "New conversation" is a quiet pill, not a primary action. */
[data-testid="stMainBlockContainer"] .stButton button {
  border: 1px solid var(--rb-line);
  border-radius: 999px;
  background: #232220;
  color: var(--rb-muted);
  font-size: 12.5px;
  font-weight: 500;
  padding: .3rem .85rem;
}
[data-testid="stMainBlockContainer"] .stButton button:hover {
  border-color: var(--rb-brand); color: var(--rb-brand);
}

/* The header row stays in view as the conversation grows. It used to
   scroll away with the first few messages, taking "New conversation" with
   it, so a long chat could only be reset by scrolling all the way back up. */
/* A sticky element cannot travel outside its parent, and Streamlit wraps
   this row in a box exactly as tall as the row. Stickiness on the row
   itself therefore did nothing; it has to be on the wrapper, whose parent
   is the whole page. */
[data-testid="stLayoutWrapper"]:has(.rb-head) {
  position: sticky; top: 0; z-index: 90;
  background: #1b1a18;
  padding: .55rem 0 .65rem;
  margin-bottom: .4rem;
  border-bottom: 1px solid #2c2b28;
}

/* The page link to the dashboard, set like the rail's other small links. */
[data-testid="stSidebar"] [data-testid="stPageLink"] a {
  padding: 0; background: transparent; font-size: 12.5px;
  color: var(--rb-muted);
}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { color: var(--rb-brand); }

/* The header sits against the top of the page, with a hairline under it
   like the chat page's. */
[data-testid="stMainBlockContainer"] { padding-top: 2.2rem; }
/* ---- spacing and alignment fixes ---------------------------------- */

/* The rail started 100px down: a 60px header reserved for the collapse
   chevron, then 24px of padding, then the brand. The header now takes no
   room and the chevron floats in the corner. */
[data-testid="stSidebarHeader"] {
  height: 0; min-height: 0; padding: 0; overflow: visible;
}
[data-testid="stSidebarHeader"] button,
[data-testid="stSidebarCollapseButton"] {
  position: absolute; top: 12px; right: 10px; z-index: 2;
}
[data-testid="stSidebarUserContent"] { padding-top: 22px !important; }

/* Suggestion text was centred because the button's inner wrappers are
   flex containers set to `justify-content: center`; setting it on the
   button alone does nothing. */
[data-testid="stSidebar"] .stButton button > div,
[data-testid="stSidebar"] .stButton button > div > span {
  justify-content: flex-start !important;
  width: 100%;
  text-align: left;
}
[data-testid="stSidebar"] .stButton button p { text-align: left; }

/* Streamlit's fixed top bar is 60px of empty strip with a background.
   It is made transparent and click-through so the header row can sit at
   the top of the page, as on the chat UI; only its own menu stays
   clickable. Without this the row slid underneath it and was cut off. */
[data-testid="stHeader"] {
  background: transparent !important;
  height: 2.6rem;
  pointer-events: none;
}
/* The toolbar is a 1200px-wide box, so making all of it clickable put it
   over the "New conversation" pill and swallowed the clicks. Only the
   buttons inside it take pointer events. */
[data-testid="stToolbar"] { pointer-events: none; }
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] a,
[data-testid="stMainMenu"],
[data-testid="stSidebarCollapseButton"] { pointer-events: auto; }

/* The header row: one line, the pill centred against the title. */
[data-testid="stMainBlockContainer"] { padding-top: 1.6rem !important; }
.rb-head { margin: 0; }
.rb-head strong { display: block; }
/* Streamlit pads every markdown block, so the header's box is taller than
   the text in it and "centred" came out 8px low against the pill. */
[data-testid="stElementContainer"]:has(.rb-head),
[data-testid="stMarkdownContainer"]:has(.rb-head),
[data-testid="stMarkdown"]:has(.rb-head) { margin: 0 !important; padding: 0 !important; }
[data-testid="stMainBlockContainer"] .stButton button {
  min-height: 32px; height: 32px;
  padding: 0 .9rem;
  white-space: nowrap;
}
[data-testid="stMainBlockContainer"] .stButton button p {
  white-space: nowrap; overflow: visible; text-overflow: clip;
}

/* The assistant speaks as the laboratory's drop, as on the chat page;
   the patient has no avatar at all. */
[data-testid="stChatMessageAvatarUser"] { display: none; }
[data-testid="stChatMessageAvatarAssistant"] {
  background: transparent !important;
  width: 26px; height: 26px; min-width: 26px;
  border-radius: 8px;
  position: relative;
}
[data-testid="stChatMessageAvatarAssistant"] > * { display: none !important; }
[data-testid="stChatMessageAvatarAssistant"]::before {
  content: "";
  position: absolute; inset: 0;
  background: var(--rb-brand);
  -webkit-mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M16 3c5.6 6.9 8.8 11.7 8.8 15.6a8.8 8.8 0 0 1-17.6 0C7.2 14.7 10.4 9.9 16 3z' fill='black'/%3E%3C/svg%3E") center / 16px 16px no-repeat;
  mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M16 3c5.6 6.9 8.8 11.7 8.8 15.6a8.8 8.8 0 0 1-17.6 0C7.2 14.7 10.4 9.9 16 3z' fill='black'/%3E%3C/svg%3E") center / 16px 16px no-repeat;
}
</style>
"""


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
        chips.append("From laboratory documents")
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


BRAND = """
<div class="rb-brand">
  <span class="rb-mark">
    <svg viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="#6fb3b8"/>
      <path d="M16 6.5c3.9 4.8 6.1 8.1 6.1 10.8a6.1 6.1 0 0 1-12.2 0c0-2.7 2.2-6 6.1-10.8z" fill="#10262a"/>
    </svg>
  </span>
  <span class="rb-brand-text">
    <strong>Riverbend Diagnostics</strong>
    <em>Medical diagnostic laboratory</em>
  </span>
</div>
"""

HEADER = """
<div class="rb-head">
  <strong>AI Assistant</strong>
  <span>Hawthorne &middot; Beaverton &middot; Gresham &middot; (503) 555-0142</span>
</div>
"""


def new_session_id() -> str:
    """A fresh, unguessable id for one visitor's conversation."""
    return uuid.uuid4().hex[:16]


def main() -> None:
    # One id per browser session. This used to be the constant "streamlit"
    # for everyone, so every visitor to the public demo shared a single
    # conversation: the dashboard showed one session with every turn in it,
    # and - worse - the lookups for "do you have my number" and "cancel it"
    # are keyed by session, so a visitor could be told another visitor's
    # name and phone number, or cancel their booking.
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()

    st.markdown(THEME, unsafe_allow_html=True)

    # The brand sits in the rail and the page title in the header bar, as
    # on the chat page. Streamlit puts a page title in the main column by
    # default, which is why the two looked unrelated.
    with st.sidebar:
        st.markdown(BRAND, unsafe_allow_html=True)

    head, action = st.columns([3, 1], vertical_alignment="center")
    with head:
        st.markdown(HEADER, unsafe_allow_html=True)
    with action:
        if st.button("New conversation", use_container_width=True):
            for key in ("history", "turns", "pending"):
                st.session_state.pop(key, None)
            # A new conversation is a new session: nothing said in the last
            # one - a name, a number, a booking - may be answered from here.
            st.session_state.session_id = new_session_id()
            st.rerun()

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
        st.subheader("About this demo")
        st.caption(
            "A customer-facing assistant built around one rule: it answers "
            "from the laboratory's own documents, or it says it cannot. A "
            "router reads each message and hands it to one of three "
            "specialists - information, appointments, complaints."
        )

        st.subheader("Try asking")
        for suggestion in (
            "How much is a full blood count?",
            "Do I need to fast for a lipid profile?",
            "Book a blood test next Tuesday at 8am",
            "When will my report be ready?",
            "My haemoglobin is 9.2, is that bad?",
            "Do you do allergy testing?",
        ):
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending = suggestion

        st.caption(
            "The last two are the interesting ones - it declines to interpret "
            "a result, and declines a service the documents do not cover."
        )

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
        # Streamlit's own page switcher is hidden so the rail matches the
        # chat UI, which took the only route to the dashboard with it. The
        # chat page has these links at the foot of its rail; so does this.
        st.page_link("pages/Admin_dashboard.py", label="Admin dashboard")
        st.info(
            "**Bookings are held in memory here.** A public demo with no "
            "login should not have write access to anyone's Google "
            "account, so no OAuth token is deployed. The calendar is a "
            "swappable backend: clone the repo, authorise once, and the "
            "same code writes to a real diary. In production this would "
            "be a service account for the business calendar.",
            icon="📅",
        )
        st.warning(
            "It will not interpret a result. That is a doctor's job, and "
            "the laboratory's own documents say so.",
            icon="⚕️",
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

    question = st.chat_input("Message the assistant...")
    if not question:
        question = st.session_state.pop("pending", None)
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Generic on purpose. Nothing is known yet about what kind of
        # message this is: "hi" and "book me a test" both wait here, and
        # telling a greeting that documents are being searched is wrong.
        with st.spinner("Generating a response…"):
            try:
                turn = run(
                    question,
                    history=st.session_state.history,
                    session_id=st.session_state.session_id,
                )
            except Exception as exc:  # the UI must not show a traceback
                st.error(
                    "Something went wrong at our end. Please try again, or "
                    "call the laboratory on (503) 555-0142."
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
