"""Application configuration.

Every environment variable the app reads is declared here, once. Modules
import the `settings` singleton rather than calling `os.getenv` themselves,
so there is a single place to see what the app needs in order to run and a
single place where a missing or malformed value is caught.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: app/config.py -> app/ -> <root>
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed view of the environment, loaded from `.env`.

    Pydantic validates types at import time, so a typo like
    `RETRIEVAL_TOP_K=four` fails immediately with a clear message instead of
    surfacing as a confusing error deep inside the retrieval layer.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM ------------------------------------------------------------
    # Default is empty rather than required: the app must still import (and
    # `/health` must still answer) when the key is absent, so that a missing
    # key produces a friendly message instead of a crash on startup.
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    router_model: str = "openai/gpt-oss-20b"
    llm_temperature: float = 0.2
    # Generous by default: these are reasoning models, and the hidden
    # reasoning tokens are drawn from the same budget as the visible reply.
    # Too small a cap yields an empty answer rather than a truncated one.
    llm_max_tokens: int = 1024
    # "low" | "medium" | "high". Classification needs no deliberation, so
    # the router runs at "low" to keep it fast and cheap.
    llm_reasoning_effort: str = "medium"
    router_reasoning_effort: str = "low"

    # --- Embeddings ------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Storage ---------------------------------------------------------
    chroma_dir: Path = Path("data/chroma")
    chroma_collection: str = "clinic_docs"
    documents_dir: Path = Path("data/documents")
    analytics_db: Path = Path("data/analytics.db")

    # --- Retrieval -------------------------------------------------------
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    retrieval_min_score: float = Field(default=0.30, ge=0.0, le=1.0)

    # --- Google Calendar -------------------------------------------------
    google_credentials_file: Path = Path("credentials.json")
    google_token_file: Path = Path("token.json")
    google_calendar_id: str = "primary"

    # --- Server ----------------------------------------------------------
    app_host: str = "127.0.0.1"
    app_port: int = 8000

    # --- Derived helpers -------------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """True when an LLM key is present and looks real."""
        key = self.groq_api_key.strip()
        return bool(key) and not key.startswith("gsk_replace")

    @property
    def calendar_configured(self) -> bool:
        """True when Google OAuth credentials exist on disk.

        When False the Booking Agent falls back to an in-memory calendar so
        a fresh clone still runs end to end.
        """
        return self.abs_path(self.google_credentials_file).is_file()

    def abs_path(self, value: Path) -> Path:
        """Resolve a configured path against the project root.

        Relative paths in `.env` should mean "relative to the repository",
        not "relative to whatever directory the process was started from".
        """
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so `.env` is parsed once and every module sees the same object.
    """
    return Settings()


settings = get_settings()
