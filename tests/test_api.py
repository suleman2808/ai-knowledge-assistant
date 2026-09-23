"""Tests for the HTTP layer.

FastAPI's TestClient runs the real application in-process — real routing,
real validation, real error handlers — with only the graph stubbed. What
is tested is the web layer's own behaviour: validation, session handling,
rate limiting, and the guarantee that no stack trace reaches a client.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app, sessions


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh sessions and rate-limit counters for every test."""
    monkeypatch.setattr(main_module, "sessions", type(sessions)())
    main_module._requests.clear()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A client whose graph is replaced by a recorder."""
    calls: list[dict] = []

    def fake_run(message: str, *, history=None, session_id=""):  # noqa: ANN001, ANN202
        calls.append({"message": message, "history": history, "session_id": session_id})
        return {
            "answer": "A root canal on a molar is $1,250.",
            "intent": "inquiry",
            "secondary_intent": None,
            "confidence": 0.95,
            "sources": [{"breadcrumb": "Services > Restorative", "source": "s.md",
                         "score": 0.62}],
            "needs_followup": False,
            "agent_metadata": {"grounded": True},
            "latency_ms": 900,
        }

    monkeypatch.setattr(main_module, "run", fake_run)
    test_client = TestClient(app)
    test_client.calls = calls  # type: ignore[attr-defined]
    return test_client


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def test_chat_returns_the_answer_and_its_provenance(client: TestClient) -> None:
    response = client.post("/api/chat", json={"message": "how much is a root canal"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("A root canal")
    assert body["intent"] == "inquiry"
    assert body["grounded"] is True
    assert body["sources"][0]["breadcrumb"] == "Services > Restorative"
    assert body["session_id"], "a session id must be issued"


def test_session_id_is_issued_then_reused(client: TestClient) -> None:
    first = client.post("/api/chat", json={"message": "hello"}).json()
    second = client.post(
        "/api/chat", json={"message": "and again", "session_id": first["session_id"]}
    ).json()

    assert second["session_id"] == first["session_id"]


def test_history_is_kept_server_side(client: TestClient) -> None:
    """Follow-up questions need history, and clients must not supply it.

    A client-supplied transcript could be forged to make the assistant
    believe it had already said something it had not.
    """
    first = client.post("/api/chat", json={"message": "how much is a filling"}).json()
    client.post(
        "/api/chat",
        json={"message": "is that for one surface?", "session_id": first["session_id"]},
    )

    second_call = client.calls[1]  # type: ignore[attr-defined]
    assert second_call["history"] == [
        {"role": "user", "content": "how much is a filling"},
        {"role": "assistant", "content": "A root canal on a molar is $1,250."},
    ]


def test_sessions_do_not_leak_into_each_other(client: TestClient) -> None:
    client.post("/api/chat", json={"message": "first", "session_id": "alice"})
    client.post("/api/chat", json={"message": "second", "session_id": "bob"})

    assert client.calls[1]["history"] == []  # type: ignore[attr-defined]


def test_history_is_trimmed(client: TestClient) -> None:
    for i in range(main_module.MAX_HISTORY_TURNS + 5):
        client.post("/api/chat", json={"message": f"message {i}", "session_id": "s"})

    last = client.calls[-1]["history"]  # type: ignore[attr-defined]
    assert len(last) == main_module.MAX_HISTORY_TURNS * 2


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": "x" * 2001},
        {"message": "hi", "session_id": "../../etc/passwd"},
        {"message": "hi", "session_id": "<script>alert(1)</script>"},
    ],
)
def test_bad_requests_are_rejected_with_a_readable_message(
    client: TestClient, payload: dict
) -> None:
    response = client.post("/api/chat", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, str) and detail, "detail must be a readable sentence"


def test_validation_failure_never_reaches_the_graph(client: TestClient) -> None:
    client.post("/api/chat", json={"message": "   "})

    assert client.calls == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_an_unhandled_error_becomes_a_sentence_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _explode(*_a, **_k):  # noqa: ANN002, ANN003
        raise RuntimeError("sqlite3.OperationalError: no such table: turns")

    monkeypatch.setattr(main_module, "run", _explode)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/chat", json={"message": "hello"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "555-0142" in detail
    assert "sqlite3" not in detail
    assert "Traceback" not in detail


def test_rate_limit_returns_429_with_a_polite_message(client: TestClient) -> None:
    for _ in range(main_module.RATE_LIMIT):
        assert client.post("/api/chat", json={"message": "hi"}).status_code == 200

    response = client.post("/api/chat", json={"message": "hi"})

    assert response.status_code == 429
    assert "wait" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_reports_progress_then_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_stream(message, *, history=None, session_id=""):  # noqa: ANN001, ANN202
        yield {"event": "node", "node": "router", "intent": "inquiry"}
        yield {"event": "node", "node": "inquiry"}
        yield {
            "event": "done",
            "turn": {
                "answer": "Yes — we are in-network with Cigna Dental PPO.",
                "intent": "inquiry", "confidence": 0.95, "sources": [],
                "needs_followup": False, "agent_metadata": {"grounded": True},
                "latency_ms": 1200,
            },
        }

    monkeypatch.setattr(main_module, "stream", fake_stream)

    with TestClient(app) as client:
        with client.stream("POST", "/api/chat/stream",
                           json={"message": "do you take cigna"}) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = [
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert [e["event"] for e in events] == ["node", "node", "done"]
    assert events[-1]["answer"].startswith("Yes")
    assert events[-1]["grounded"] is True
    assert events[-1]["session_id"]


def test_stream_failure_is_delivered_as_an_event_not_a_dropped_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The response has already started, so the status code cannot change.

    The client has a 200 and an open stream; the only way to report a
    failure is inside the stream.
    """

    def fake_stream(message, *, history=None, session_id=""):  # noqa: ANN001, ANN202
        yield {"event": "node", "node": "router"}
        raise RuntimeError("groq exploded")

    monkeypatch.setattr(main_module, "stream", fake_stream)

    with TestClient(app) as client:
        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as response:
            events = [
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert events[-1]["event"] == "error"
    assert "555-0142" in events[-1]["detail"]
    assert "groq exploded" not in events[-1]["detail"]


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def test_health_never_fails_even_when_dependencies_do(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A health endpoint that errors when something is broken is useless."""
    monkeypatch.setattr(
        "app.rag.store.collection_stats",
        lambda: (_ for _ in ()).throw(RuntimeError("chroma is gone")),
    )

    with TestClient(app) as client:
        response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["checks"]["knowledge_base"]["ready"] is False


def test_analytics_endpoint_returns_the_summary() -> None:
    with TestClient(app) as client:
        body = client.get("/api/analytics?days=7").json()

    assert body["window_days"] == 7
    assert "knowledge_gaps" in body


def test_analytics_window_is_clamped() -> None:
    with TestClient(app) as client:
        assert client.get("/api/analytics?days=99999").json()["window_days"] == 365
        assert client.get("/api/analytics?days=-5").json()["window_days"] == 1


def test_ui_and_docs_are_served() -> None:
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert "Riverbend" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
