"""Tests for configuration and its documentation.

`.env.example` is the only instruction a new user gets about what the
app can be configured with. It drifts silently: a setting added in code
six steps later is invisible until someone needs it. These tests make
that drift fail the build instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.config import PROJECT_ROOT, Settings, settings

EXAMPLE = PROJECT_ROOT / ".env.example"


def documented_keys() -> set[str]:
    text = EXAMPLE.read_text(encoding="utf-8")
    return {m.lower() for m in re.findall(r"^([A-Z_][A-Z0-9_]*)=", text, re.M)}


def test_every_setting_is_documented() -> None:
    """Caught two real omissions: CLINIC_TIMEZONE and RETRIEVAL_MODE."""
    missing = sorted(set(Settings.model_fields) - documented_keys())

    assert not missing, f"present in config.py but absent from .env.example: {missing}"


def test_env_example_documents_nothing_imaginary() -> None:
    """A setting in the example that the app ignores is worse than none."""
    unknown = sorted(documented_keys() - set(Settings.model_fields))

    assert not unknown, f"in .env.example but not read by the app: {unknown}"


def test_env_example_ships_no_real_key() -> None:
    """The file is committed, so it must never contain a working key."""
    text = EXAMPLE.read_text(encoding="utf-8")
    key_line = next(l for l in text.splitlines() if l.startswith("GROQ_API_KEY="))

    assert key_line.split("=", 1)[1].startswith("gsk_replace")


def test_gitignore_covers_every_secret() -> None:
    """The constraint this project is built around."""
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

    for secret in (".env", "credentials.json", "token.json"):
        assert secret in ignored, f"{secret} is not gitignored"


def test_placeholder_key_is_not_treated_as_configured() -> None:
    """A fresh clone that forgot the key must report setup, not an outage."""
    fresh = Settings(groq_api_key="gsk_replace_me")

    assert fresh.llm_configured is False
    assert Settings(groq_api_key="gsk_realish_value_here").llm_configured is True


def test_relative_paths_resolve_against_the_project_root() -> None:
    """Otherwise the data directory depends on where the process started."""
    resolved = settings.abs_path(Path("data/chroma"))

    assert resolved.is_absolute()
    assert resolved.parent.parent == PROJECT_ROOT


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    assert settings.abs_path(tmp_path) == tmp_path


@pytest.mark.parametrize(
    "field,value",
    [("retrieval_top_k", 0), ("retrieval_top_k", 999), ("retrieval_min_score", 1.5)],
)
def test_out_of_range_values_are_rejected_at_startup(field: str, value: object) -> None:
    """Fail loudly at import, not confusingly inside the retrieval layer."""
    with pytest.raises(Exception):
        Settings(**{field: value})
