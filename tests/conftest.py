"""Shared test configuration.

Every test gets its own throwaway analytics database. Without this, any
test that calls `graph.run()` would write into the real
`data/analytics.db` and quietly skew the numbers a laboratory owner sees.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_analytics(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.agents import complaint as complaint_module
    from app.config import settings

    monkeypatch.setattr(settings, "analytics_db", tmp_path / "analytics.db")
    # The complaint store is a lazily built singleton; reset it so it is
    # rebuilt against the temporary database.
    monkeypatch.setattr(complaint_module, "_store", None)


@pytest.fixture(autouse=True)
def _no_real_model_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if a test reaches the provider for real.

    The suite's contract is that it needs no API key and no network: CI
    runs it on three Python versions with neither. A test that forgets to
    stub its model call still passes on a laptop with a key in `.env`,
    and then turns CI red for a reason the author never sees locally -
    which is exactly what happened to two booking tests.

    Stubbing is done at `_chat`, the single provider seam, so every
    public helper above it is covered by this one line.
    """
    from app import llm

    def _refuse(*_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError(
            "This test called the language model for real. Stub it - "
            "`monkeypatch.setattr(\"app.agents.booking.complete_json\", ...)` "
            "or the equivalent for the module under test."
        )

    monkeypatch.setattr(llm, "_chat", _refuse)
