"""Tests for the Streamlit admin dashboard.

Driven by Streamlit's own `AppTest`, which runs the page headlessly and
exposes the elements it produced. That is worth more than a browser
check here: the page is mostly conditional rendering, and the condition
that matters most is whether someone is signed in.

Typing and submitting happen in a single `run()`. Inside an `st.form`
nothing is committed until the submit button is pressed, so a `run()`
between the two throws the typed value away - which is what Streamlit
1.65 started doing, and it turned CI red while every laptop still had
1.64 installed.
"""

from __future__ import annotations

import pytest

from app.config import PROJECT_ROOT, settings
from app.integrations.analytics import log_turn

PASSWORD = "correct horse battery staple"
# Absolute: AppTest resolves a relative path against the test file,
# not the working directory.
PAGE = PROJECT_ROOT / "pages" / "Admin_dashboard.py"


@pytest.fixture(autouse=True)
def _password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADMIN_PASSWORD", PASSWORD)
    monkeypatch.setattr(settings, "admin_password", PASSWORD)


def page(signed_in: bool = False):  # noqa: ANN201
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PAGE), default_timeout=30)
    if signed_in:
        app.session_state["admin_ok"] = True
    return app.run()


def text_of(app) -> str:  # noqa: ANN001
    """Everything the page rendered, as one searchable string."""
    parts = [element.value for element in app.markdown]
    parts += [element.value for element in app.caption]
    # A metric exposes its label and its value separately.
    parts += [f"{element.label} {element.value}" for element in app.metric]
    parts += [element.value for element in app.title]
    parts += [element.value for element in app.subheader]
    parts += [element.value for element in app.error]
    parts += [element.value for element in app.info]
    return "\n".join(str(p) for p in parts)


def record_turn(session_id: str = "audited", **overrides) -> None:  # noqa: ANN003
    base = {
        "session_id": session_id,
        "message": "how much is a full blood count",
        "answer": "A full blood count is $45.",
        "intent": "inquiry",
        "success": True,
        "needs_followup": False,
        "sources": [
            {"breadcrumb": "Tests > Haematology", "source": "catalogue.md", "score": 0.74}
        ],
        "agent_metadata": {"grounded": True},
    }
    base.update(overrides)
    log_turn(base, latency_ms=800)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_the_dashboard_is_closed_until_you_sign_in() -> None:
    app = page()

    rendered = text_of(app)
    assert "Admin dashboard" in rendered
    assert not app.tabs, "no data should be rendered before signing in"


def test_the_wrong_password_does_not_open_it() -> None:
    app = page()
    app.text_input[0].set_value("not the password")
    app.button[0].click().run()

    assert app.session_state.get("admin_ok") is not True
    assert any("Incorrect" in e.value for e in app.error)


def test_the_right_password_opens_it() -> None:
    app = page()
    app.text_input[0].set_value(PASSWORD)
    app.button[0].click().run()

    assert app.session_state["admin_ok"] is True


def test_a_failed_attempt_is_counted() -> None:
    app = page()
    app.text_input[0].set_value("wrong")
    app.button[0].click().run()

    assert app.session_state["admin_attempts"] == 1


def test_guessing_is_limited() -> None:
    """After enough wrong answers the form is withdrawn entirely.

    Driven through session state rather than by clicking five times,
    because the widget tree is rebuilt on every rerun and the test would
    be about Streamlit's internals rather than about the limit.
    """
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PAGE), default_timeout=30)
    app.session_state["admin_attempts"] = 5
    app.run()

    assert not app.text_input, "the form should be withdrawn"
    assert any("Too many attempts" in e.value for e in app.error)


def test_with_no_password_configured_it_refuses_rather_than_opening(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    monkeypatch.setattr(settings, "admin_password", "")

    app = page()

    assert any("switched off" in e.value for e in app.error)
    assert not app.tabs


# ---------------------------------------------------------------------------
# What it shows once open
# ---------------------------------------------------------------------------


def test_all_six_views_are_present() -> None:
    record_turn()

    app = page(signed_in=True)

    labels = [tab.label for tab in app.tabs]
    assert labels == [
        "Overview", "Conversations", "Bookings", "Complaints", "Callbacks", "Customers",
    ]


def test_the_overview_reports_real_figures() -> None:
    record_turn()
    record_turn()

    rendered = text_of(page(signed_in=True))

    assert "Messages" in rendered
    assert "Knowledge gaps" in rendered


def test_a_declined_question_appears_as_a_knowledge_gap() -> None:
    record_turn(
        message="do you do allergy testing",
        answer="I don't have that in the laboratory's information.",
        sources=[],
        agent_metadata={"grounded": False},
    )

    rendered = text_of(page(signed_in=True))

    assert "do you do allergy testing" in rendered


def test_the_transcript_shows_the_sections_behind_an_answer() -> None:
    """The claim this whole project rests on, made auditable."""
    record_turn("audited")

    rendered = text_of(page(signed_in=True))

    assert "A full blood count is $45." in rendered
    assert "Tests > Haematology" in rendered
    assert "catalogue.md" in rendered


def test_the_transcript_redacts_what_the_booking_table_keeps() -> None:
    record_turn(
        "with-contact",
        message="book a test, Sarah Chen, 503-555-0180",
        intent="booking",
        agent_metadata={
            "stage": "booked", "event_id": "e1", "calendar_backend": "in_memory",
            "start": "2026-10-06T09:00:00", "end": "2026-10-06T09:30:00",
            "extracted": {"service": "full blood count",
                          "patient_name": "Sarah Chen", "phone": "503-555-0180"},
        },
    )

    rendered = text_of(page(signed_in=True))

    assert "[phone]" in rendered, "the transcript must not show the number"
    assert "503-555-0180" in rendered, "the bookings table must"


def test_it_says_plainly_that_this_deployment_forgets() -> None:
    """Free hosting has no disk. Better to say so than let someone
    discover it when the records vanish."""
    rendered = text_of(page(signed_in=True))

    assert "reset when the app restarts" in rendered


def test_the_page_does_not_use_pandas_backed_widgets() -> None:
    """pandas is blocked by Application Control on the development
    machine, so the page renders tables and bars as markdown."""
    source = PAGE.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )

    for widget in ("st.dataframe(", "st.bar_chart(", "st.table("):
        assert widget not in code


# ---------------------------------------------------------------------------
# Keeping the two front-ends saying the same thing
# ---------------------------------------------------------------------------


def test_the_streamlit_pages_match_the_house_style() -> None:
    """The Streamlit front-end drifted from the chat UI without anyone
    noticing, because it is a separate file that nothing imports from the
    other one. The live demo still said "Hello —" and offered four
    suggestions weeks after the chat page had been changed.

    This checks the one rule that is mechanical enough to check.
    """
    import re

    for name in ("streamlit_app.py", "pages/Admin_dashboard.py"):
        source = (PROJECT_ROOT / name).read_text(encoding="utf-8")
        # Strip the module docstring: it is prose for developers, and it
        # keeps its punctuation.
        body = re.sub(r'^\s*(""".*?""")', "", source, count=1, flags=re.S)
        offenders = [
            line.strip()
            for line in body.splitlines()
            if "\u2014" in line and not line.strip().startswith("#")
        ]
        assert not offenders, f"{name} has long dashes in user-facing text: {offenders}"


def test_both_front_ends_offer_the_same_questions() -> None:
    """The two suggestions that demonstrate the refusals are the point of
    the demo. They were in the chat UI and missing from the live one."""
    chat = (PROJECT_ROOT / "ui" / "index.html").read_text(encoding="utf-8")
    streamlit = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    for question in ("My haemoglobin is 9.2, is that bad?", "Do you do allergy testing?"):
        assert question in chat, f"{question!r} missing from the chat UI"
        assert question in streamlit, f"{question!r} missing from the Streamlit app"


def test_the_loading_message_does_not_claim_to_search_documents() -> None:
    """Shown before anything is known about the message, so a greeting got
    "Searching the laboratory's documents". It has to stay generic."""
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert "st.spinner(" in source
    assert "Searching the laboratory" not in source


# ---------------------------------------------------------------------------
# One visitor must not see another's conversation
# ---------------------------------------------------------------------------


def test_every_visitor_to_the_hosted_demo_gets_their_own_session() -> None:
    """Found reading the live dashboard: 27 messages, one conversation. The
    session id was the constant "streamlit" for everyone, so every visitor
    shared one conversation - and "do you have my number" and "cancel it"
    are looked up by session, so a visitor could be told another visitor's
    name and phone number, or cancel their booking."""
    from streamlit.testing.v1 import AppTest

    first = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=60).run()
    second = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=60).run()

    a = first.session_state["session_id"]
    b = second.session_state["session_id"]

    assert a != b
    assert a != "streamlit" and b != "streamlit"
    assert len(a) == 16


def test_the_session_id_is_never_a_constant_in_the_source() -> None:
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    assert '"streamlit")' not in source
    assert "session_id=st.session_state.session_id" in source


def test_a_new_conversation_is_a_new_session() -> None:
    """Nothing said in the last one - a name, a number, a booking - may be
    answered from the next."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=60).run()
    before = app.session_state["session_id"]

    # `llm_configured` is false without a key, which stops the page before
    # the button is drawn; the reset is checked in the source instead.
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    reset = source[source.index('if st.button("New conversation"'):]
    assert "st.session_state.session_id = new_session_id()" in reset[:900]
    assert before


def test_the_callbacks_tab_exists_and_lists_a_request() -> None:
    """The live demo records callbacks, and the Streamlit dashboard had no
    tab to show them: the feature existed only on the HTML dashboard."""
    from app.integrations.analytics import record_callback

    record_callback(session_id="cb-test", patient_name="Sam Reed",
                    phone="(503) 555-0122", context="asked to speak to someone")

    app = page(signed_in=True)

    labels = [t.label for t in app.tabs]
    assert "Callbacks" in labels
    assert "Sam Reed" in text_of(app) or any("Sam Reed" in str(e.value) for e in app.main)


def test_a_normalised_phone_key_is_shown_as_a_phone_number() -> None:
    """The customers table showed "5035550155", a matching key."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("admin_page", PAGE)
    source = PAGE.read_text(encoding="utf-8")
    assert "def spoken_phone" in source
    namespace: dict = {}
    exec(  # noqa: S102 - the page's own pure helper, extracted to test it alone
        source[source.index("def spoken_phone"):source.index("def callbacks")], namespace
    )

    assert namespace["spoken_phone"]("5035550155") == "(503) 555-0155"
    assert namespace["spoken_phone"]("503-555-0155") == "(503) 555-0155"
    assert namespace["spoken_phone"]("") == ""
    assert spec is not None
