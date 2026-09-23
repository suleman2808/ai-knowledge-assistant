"""HTTP layer: the chat API, health, analytics and the static UI.

The API is thin on purpose. Every route does three things — validate,
call the graph, shape the response — and holds no business logic of its
own. That is what makes the graph runnable from a CLI, a test or a
notebook without the web layer, and it keeps this file readable.

Three things here are not optional in anything customer-facing:

- **No stack trace ever reaches a client.** An unhandled exception
  becomes a plain sentence and a 500; the detail goes to the log.
- **Conversation state lives on the server**, keyed by session id, so a
  client cannot forge history and steer the assistant with invented
  prior turns.
- **Requests are rate limited** per client, because an unauthenticated
  endpoint that spends money on model calls is otherwise an open tab.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import PROJECT_ROOT, settings
from app.graph.build import run, stream
from app.schemas import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

UI_DIR = PROJECT_ROOT / "ui"

# How many turns of a conversation to keep. Enough for a booking to be
# completed across several messages; short enough that memory cannot grow
# without bound.
MAX_HISTORY_TURNS = 12
# Conversations idle for longer than this are discarded.
SESSION_TTL_SECONDS = 60 * 60

# Rate limit: requests per window, per client.
RATE_LIMIT = 30
RATE_WINDOW_SECONDS = 60


# --------------------------------------------------------------------------
# Session storage
# --------------------------------------------------------------------------


class SessionStore:
    """In-memory conversation history.

    Deliberately not persistent. A dental clinic's chat has no need to
    survive a restart, and keeping transcripts — which contain names,
    symptoms and complaints — on disk without a retention policy would
    create a data-protection problem the project does not need. Swapping
    this for Redis behind the same three methods is the scaling step.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._touched: dict[str, float] = {}

    def history(self, session_id: str) -> list[dict[str, str]]:
        self._expire()
        return list(self._sessions.get(session_id, []))

    def append(self, session_id: str, message: str, answer: str) -> None:
        turns = self._sessions.setdefault(session_id, [])
        turns.append({"role": "user", "content": message})
        turns.append({"role": "assistant", "content": answer})
        # Trim oldest first, keeping whole user/assistant pairs.
        del turns[: max(0, len(turns) - MAX_HISTORY_TURNS * 2)]
        self._touched[session_id] = time.time()

    def _expire(self) -> None:
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [s for s, seen in self._touched.items() if seen < cutoff]
        for session_id in stale:
            self._sessions.pop(session_id, None)
            self._touched.pop(session_id, None)

    @property
    def count(self) -> int:
        return len(self._sessions)


sessions = SessionStore()


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------

_requests: dict[str, deque[float]] = defaultdict(deque)


def _rate_limited(client: str) -> bool:
    """Sliding-window limiter, per client.

    In-process, so it protects a single instance rather than a cluster.
    That is the honest scope: behind a load balancer this belongs in the
    proxy or a shared store.
    """
    now = time.time()
    window = _requests[client]
    while window and window[0] < now - RATE_WINDOW_SECONDS:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return True
    window.append(now)
    return False


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201, ARG001
    """Warm the slow dependencies before the first patient waits on them.

    Loading the embedding model takes a few seconds. Doing it on the
    first request would make that request look broken.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
    for noisy in ("httpx", "httpcore", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        from app.rag.embeddings import get_backend

        get_backend()
        logger.info("Embedding model ready")
    except Exception as exc:
        # Not fatal: the API must still start so /health can report why.
        logger.error("Embedding model unavailable at startup: %s", exc)

    yield


app = FastAPI(
    title="Riverbend Dental Care — AI Assistant",
    description=(
        "A RAG assistant with a multi-agent router. An incoming message is "
        "classified and dispatched to a booking, inquiry or complaint "
        "specialist. Inquiry answers are grounded in the clinic's documents "
        "and refused when the documents do not cover the question."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):  # noqa: ANN201, ARG001
    """Turn schema violations into something a person can act on."""
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    logger.info("Rejected request: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": f"{field or 'request'}: {first.get('msg', 'is invalid')}"},
    )


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):  # noqa: ANN201
    """Last line of defence: no stack trace reaches a patient."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Something went wrong at our end. Please try again, or call "
                "the clinic on (503) 555-0142."
            )
        },
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    """Report whether each dependency is usable.

    Never raises, so it is safe to poll. Distinguishes "not configured"
    from "broken", because they need different people to fix them.
    """
    checks: dict[str, Any] = {}

    from app.llm import health as llm_health

    checks["llm"] = llm_health()

    try:
        from app.rag.store import collection_stats

        stats = collection_stats()
        checks["knowledge_base"] = {
            "ready": stats["count"] > 0,
            "chunks": stats["count"],
            "documents": len(stats["sources"]),
        }
    except Exception as exc:
        checks["knowledge_base"] = {"ready": False, "error": str(exc)}

    try:
        from app.rag.embeddings import backend_name

        checks["embeddings"] = {"ready": True, "backend": backend_name()}
    except Exception as exc:
        checks["embeddings"] = {"ready": False, "error": str(exc)}

    from app.integrations.calendar import get_calendar

    try:
        checks["calendar"] = {"ready": True, "backend": get_calendar().name}
    except Exception as exc:
        checks["calendar"] = {"ready": False, "error": str(exc)}

    checks["sessions"] = {"active": sessions.count}

    healthy = (
        checks["llm"]["configured"]
        and checks["knowledge_base"].get("ready")
        and checks["embeddings"].get("ready")
    )
    return HealthResponse(status="ok" if healthy else "degraded", checks=checks)


def _reply(turn: dict[str, Any], session_id: str) -> ChatResponse:
    """Shape a turn record into the public response."""
    meta = turn.get("agent_metadata") or {}
    return ChatResponse(
        answer=turn.get("answer", ""),
        session_id=session_id,
        intent=turn.get("intent", "unknown"),
        secondary_intent=turn.get("secondary_intent"),
        confidence=turn.get("confidence", 0.0),
        sources=turn.get("sources", []),
        needs_followup=turn.get("needs_followup", False),
        grounded=meta.get("grounded"),
        latency_ms=turn.get("latency_ms", 0),
    )


@app.post("/api/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest, request: Request) -> Any:
    """Answer one message.

    History is looked up server-side from `session_id`; clients send only
    the new message. Returns the answer with its citations and the
    routing decision that produced it.
    """
    if _rate_limited(_client_key(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many messages just now — please wait a moment."},
        )

    session_id = payload.session_id or uuid.uuid4().hex[:16]
    turn = run(
        payload.message,
        history=sessions.history(session_id),
        session_id=session_id,
    )
    sessions.append(session_id, payload.message, turn.get("answer", ""))
    return _reply(turn, session_id)


@app.post("/api/chat/stream", tags=["chat"])
def chat_stream(payload: ChatRequest, request: Request) -> Any:
    """Answer one message, reporting progress as each node completes.

    Server-sent events. The agents' own model calls are not streamed, so
    there are no tokens to emit; what is emitted is which stage is
    running. A five-second wait with visible progress reads as work.
    """
    if _rate_limited(_client_key(request)):
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many messages just now — please wait a moment."},
        )

    session_id = payload.session_id or uuid.uuid4().hex[:16]
    history = sessions.history(session_id)

    def events() -> Iterator[str]:
        try:
            for event in stream(payload.message, history=history, session_id=session_id):
                if event.get("event") == "done":
                    turn = event["turn"]
                    sessions.append(session_id, payload.message, turn.get("answer", ""))
                    body = _reply(turn, session_id).model_dump()
                    yield f"data: {json.dumps({'event': 'done', **body})}\n\n"
                else:
                    yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming chat failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "event": "error",
                        "detail": "Something went wrong at our end. Please try "
                                  "again, or call the clinic on (503) 555-0142.",
                    }
                )
                + "\n\n"
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/analytics", tags=["operations"])
def analytics(days: int = 30) -> Any:
    """Summarise what patients asked and how well it went."""
    from app.integrations.analytics import summary

    return summary(days=max(1, min(days, 365)))


# --------------------------------------------------------------------------
# Static UI
# --------------------------------------------------------------------------

if UI_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(UI_DIR / "index.html")

    @app.get("/analytics", include_in_schema=False)
    def analytics_page() -> FileResponse:
        return FileResponse(UI_DIR / "analytics.html")


def main() -> None:
    """Run the development server: `python -m app.main`."""
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.app_host, port=settings.app_port, reload=False
    )


if __name__ == "__main__":
    main()
