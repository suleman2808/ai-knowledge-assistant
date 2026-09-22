"""Shared test configuration.

Every test gets its own throwaway analytics database. Without this, any
test that calls `graph.run()` would write into the real
`data/analytics.db` and quietly skew the numbers a clinic owner sees.
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
