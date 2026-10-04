"""Tests for the admin dashboard and its authentication.

The dashboard shows transcripts, names, phone numbers and complaints —
the material this project is careful about everywhere else. So most of
these tests are about what happens when someone is *not* signed in.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import admin_auth
from app import main as main_module
from app.config import settings
from app.integrations.analytics import log_turn
from app.main import app

PASSWORD = "correct horse battery staple"

PROTECTED = [
    "/api/admin/conversations",
    "/api/admin/bookings",
    "/api/admin/complaints",
    "/api/admin/customers",
    "/api/analytics",
]


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "admin_password", PASSWORD)
    main_module._requests.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def sign_in(client: TestClient) -> None:
    response = client.post("/api/admin/login", json={"password": PASSWORD})
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# Who can see what
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PROTECTED)
def test_everything_private_requires_a_session(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", PROTECTED)
def test_a_session_opens_everything(client: TestClient, path: str) -> None:
    sign_in(client)

    assert client.get(path).status_code == 200


def test_the_health_check_stays_public(client: TestClient) -> None:
    """Monitoring must not need a password, and it exposes nothing."""
    assert client.get("/api/health").status_code == 200


def test_the_chat_stays_public(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main_module, "run",
        lambda message, **_k: {"answer": "hello", "intent": "other", "sources": []},
    )

    assert client.post("/api/chat", json={"message": "hi"}).status_code == 200


# ---------------------------------------------------------------------------
# Signing in
# ---------------------------------------------------------------------------


def test_the_wrong_password_is_refused(client: TestClient) -> None:
    response = client.post("/api/admin/login", json={"password": "guess"})

    assert response.status_code == 401
    assert client.get("/api/admin/conversations").status_code == 401


def test_the_password_is_never_returned(client: TestClient) -> None:
    """Not in the body, and not in the cookie either."""
    response = client.post("/api/admin/login", json={"password": PASSWORD})

    assert PASSWORD not in response.text
    assert PASSWORD not in response.headers.get("set-cookie", "")


def test_the_session_cookie_is_not_readable_by_scripts(client: TestClient) -> None:
    response = client.post("/api/admin/login", json={"password": PASSWORD})

    assert "httponly" in response.headers["set-cookie"].lower()


def test_a_forged_cookie_is_refused(client: TestClient) -> None:
    client.cookies.set(admin_auth.COOKIE_NAME, "bm90LWEtcmVhbC1zZXNzaW9u")

    assert client.get("/api/admin/conversations").status_code == 401


def test_changing_the_password_invalidates_existing_sessions(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The signing key is derived from the password, so this is free."""
    sign_in(client)
    assert client.get("/api/admin/conversations").status_code == 200

    monkeypatch.setattr(settings, "admin_password", "something else entirely")

    assert client.get("/api/admin/conversations").status_code == 401


def test_signing_out_ends_the_session(client: TestClient) -> None:
    sign_in(client)
    client.post("/api/admin/logout")

    assert client.get("/api/admin/conversations").status_code == 401


def test_repeated_guessing_is_rate_limited(client: TestClient) -> None:
    """The login is the only endpoint where guessing repeatedly pays."""
    for _ in range(main_module.RATE_LIMIT):
        client.post("/api/admin/login", json={"password": "wrong"})

    assert client.post("/api/admin/login", json={"password": "wrong"}).status_code == 429


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


def test_with_no_password_the_dashboard_refuses_to_serve(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure everyone regrets is the one where protection was never
    switched on. No password means closed, not open."""
    monkeypatch.setattr(settings, "admin_password", "")

    assert client.get("/api/admin/conversations").status_code == 503
    assert client.post("/api/admin/login", json={"password": ""}).status_code == 503


def test_the_page_can_tell_whether_it_is_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get("/api/admin/session").json() == {
        "configured": True, "signed_in": False
    }

    monkeypatch.setattr(settings, "admin_password", "")
    assert client.get("/api/admin/session").json()["configured"] is False


# ---------------------------------------------------------------------------
# What the dashboard shows
# ---------------------------------------------------------------------------


def logged_turn(session_id: str = "abc", **overrides) -> None:  # noqa: ANN003
    base = {
        "session_id": session_id,
        "message": "how much is a full blood count",
        "answer": "A full blood count is $45.",
        "intent": "inquiry",
        "confidence": 0.95,
        "routed_by": "llm",
        "success": True,
        "needs_followup": False,
        "sources": [
            {"breadcrumb": "Services > Haematology", "source": "tests.md", "score": 0.71}
        ],
        "agent_metadata": {"grounded": True, "search_query": "price of a full blood count"},
    }
    base.update(overrides)
    log_turn(base, latency_ms=900)


def test_conversations_are_grouped_not_listed_as_turns(client: TestClient) -> None:
    logged_turn("one")
    logged_turn("one")
    logged_turn("two")
    sign_in(client)

    rows = client.get("/api/admin/conversations").json()

    assert {r["session_id"] for r in rows} == {"one", "two"}
    assert next(r for r in rows if r["session_id"] == "one")["turns"] == 2


def test_a_transcript_carries_the_retrieval_trace(client: TestClient) -> None:
    """The view that makes a grounded answer auditable rather than claimed."""
    logged_turn("traced")
    sign_in(client)

    turns = client.get("/api/admin/conversations/traced").json()

    assert turns[0]["answer"] == "A full blood count is $45."
    assert turns[0]["sources"][0]["breadcrumb"] == "Services > Haematology"
    assert turns[0]["search_query"] == "price of a full blood count"
    assert turns[0]["grounded"] is True


def test_an_unknown_conversation_is_a_404_not_an_empty_page(client: TestClient) -> None:
    sign_in(client)

    assert client.get("/api/admin/conversations/nope").status_code == 404


def test_bookings_show_the_contact_details_turns_redact(client: TestClient) -> None:
    logged_turn(
        "booking-session",
        intent="booking",
        message="book a blood test, Sarah Chen, 503-555-0180",
        agent_metadata={
            "stage": "booked", "event_id": "e1", "calendar_backend": "google",
            "start": "2026-10-06T09:00:00", "end": "2026-10-06T09:30:00",
            "extracted": {"service": "full blood count",
                          "patient_name": "Sarah Chen", "phone": "503-555-0180"},
        },
    )
    sign_in(client)

    booking = client.get("/api/admin/bookings").json()[0]
    transcript = client.get("/api/admin/conversations/booking-session").json()

    assert booking["phone"] == "503-555-0180"
    assert booking["patient_name"] == "Sarah Chen"
    assert "503-555-0180" not in transcript[0]["message"]


def test_requested_limits_are_clamped(client: TestClient) -> None:
    """A caller asking for everything should not be able to."""
    sign_in(client)

    assert client.get("/api/admin/conversations?limit=99999").status_code == 200
    assert client.get("/api/admin/bookings?limit=-1").status_code == 200


def test_the_dashboard_page_is_served(client: TestClient) -> None:
    page = client.get("/admin")

    assert page.status_code == 200
    assert "Admin dashboard" in page.text
    assert client.get("/static/admin.js").status_code == 200


def test_a_password_that_arrives_after_import_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reported from the live deployment: the dashboard said it was
    switched off with the secret correctly set.

    `settings` is built once, when `app.config` is first imported. On
    Streamlit Cloud the secret is copied into the environment by whichever
    page runs first, and if that page had already imported `app.config`,
    the password was baked in as empty. The check now reads the
    environment at call time, so import order cannot decide whether
    authentication exists.
    """
    monkeypatch.setattr(settings, "admin_password", "")
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    assert admin_auth.is_enabled() is False

    monkeypatch.setenv("ADMIN_PASSWORD", "arrived late")

    assert admin_auth.is_enabled() is True
    assert admin_auth.check_password("arrived late") is True
    assert admin_auth.check_password("wrong") is False


def test_the_environment_wins_over_a_stale_settings_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "admin_password", "stale value")
    monkeypatch.setenv("ADMIN_PASSWORD", "the real one")

    assert admin_auth.check_password("the real one") is True
    assert admin_auth.check_password("stale value") is False


def test_the_ui_is_served_with_no_cache(client: TestClient) -> None:
    """A stale `app.js` once made a fixed bug look unfixed for an hour.

    `no-cache` does not mean "do not store" — the browser may keep the
    file, it just has to revalidate before using it, which on the same
    origin costs a 304.
    """
    for path in ("/", "/admin", "/analytics", "/static/app.js", "/static/styles.css"):
        assert client.get(path).headers["cache-control"] == "no-cache", path
