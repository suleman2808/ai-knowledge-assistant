"""Tests for query handling in the retriever."""

from __future__ import annotations

import pytest




# ---------------------------------------------------------------------------
# Plain "hours", and named tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("what are your hours", "what are your opening hours"),
        ("hours?", "opening hours?"),
        ("what are the hours", "what are the opening hours"),
        # Already says what kind of hours.
        ("what are your opening hours", "what are your opening hours"),
        ("after hours emergency", "after hours emergency"),
        # "Hours" meaning a length of time, not the laboratory's.
        ("how many hours should I fast", "how many hours should I fast"),
        ("results within 24 hours", "results within 24 hours"),
        ("how many hours is the fast for a lipid profile",
         "how many hours is the fast for a lipid profile"),
    ],
)
def test_plain_hours_means_opening_hours_and_nothing_else_changes(
    query: str, expected: str
) -> None:
    """"What are your hours" retrieved "Out of Hours" and never the hours
    section, so the model - correctly - refused to answer one of the most
    asked questions there is."""
    from app.rag.retriever import _plain_hours

    assert _plain_hours(query) == expected
