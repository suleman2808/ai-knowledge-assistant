"""Tests for token streaming.

The risk streaming introduces is not that tokens fail to arrive. It is
that the *wrong* tokens arrive: a sentinel the code was going to turn
into a decline, or the first half of an attempt that was then retried.
Both would be visible to a patient, so both are tested here.
"""

from __future__ import annotations

import pytest

from app import llm
from app.agents.base import guarded_tokens, normalise


def feed(pieces: list[str | None], *, sentinels: tuple[str, ...] = ()) -> str:
    """Push `pieces` through the guard and return what a reader would see."""
    seen: list[str] = []

    def sink(piece: str | None) -> None:
        if piece is None:
            seen.clear()
        else:
            seen.append(piece)

    with llm.token_sink(sink):
        with guarded_tokens(*sentinels):
            inner = llm.active_sink()
            assert inner is not None
            for piece in pieces:
                inner(piece)

    return "".join(seen)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_prose_reaches_the_reader() -> None:
    assert feed(
        ["A lipid ", "profile ", "costs $35."], sentinels=("INSUFFICIENT_CONTEXT",)
    ) == "A lipid profile costs $35."


def test_a_reply_shorter_than_the_sentinel_is_not_swallowed() -> None:
    """Held characters are released when the stream ends, not lost."""
    assert feed(["$45."], sentinels=("INSUFFICIENT_CONTEXT",)) == "$45."


@pytest.mark.parametrize(
    "pieces",
    [
        ["INSUFFICIENT_CONTEXT"],
        ["INSUFFICIENT", "_CONTEXT"],
        ["I", "N", "S", "U", "F", "F", "I", "C", "I", "E", "N", "T", "_CONTEXT"],
        ['"INSUFFICIENT_CONTEXT"'],
        ["**INSUFFICIENT_CONTEXT**"],
        ["  insufficient_context  "],
    ],
)
def test_the_sentinel_never_reaches_the_reader(pieces: list[str]) -> None:
    """However it is chunked or decorated, nobody should watch it appear."""
    assert feed(pieces, sentinels=("INSUFFICIENT_CONTEXT",)) == ""


def test_an_abandoned_attempt_clears_what_was_shown() -> None:
    """A retry starts the reply over; the reader must not see both halves."""
    assert feed(
        ["The Hawthorne branch opens at ", None, "The Hawthorne branch opens at 6:30."],
        sentinels=("INSUFFICIENT_CONTEXT",),
    ) == "The Hawthorne branch opens at 6:30."


def test_without_a_listener_the_guard_is_inert() -> None:
    """Nothing is streaming, so nothing should be buffered or dropped."""
    with guarded_tokens("INSUFFICIENT_CONTEXT"):
        assert llm.active_sink() is None


# ---------------------------------------------------------------------------
# The sink
# ---------------------------------------------------------------------------


def test_streaming_is_off_unless_someone_is_listening() -> None:
    assert llm.streaming() is False
    with llm.token_sink(lambda _piece: None):
        assert llm.streaming() is True
    assert llm.streaming() is False


def test_the_sink_is_restored_after_an_error() -> None:
    with pytest.raises(RuntimeError):
        with llm.token_sink(lambda _piece: None):
            raise RuntimeError("boom")

    assert llm.streaming() is False


# ---------------------------------------------------------------------------
# House style
# ---------------------------------------------------------------------------


def test_long_dashes_are_folded_to_hyphens() -> None:
    """Model prose is normalised at the same boundary as invisibles."""
    assert normalise("open 8am \u2014 6pm") == "open 8am - 6pm"
    assert normalise("Monday \u2013 Friday") == "Monday - Friday"


# ---------------------------------------------------------------------------
# Compound questions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "parts"),
    [
        (
            "what are your opening hours and where are your branches?",
            ["what are your opening hours", "where are your branches"],
        ),
        (
            "do I need to fast for a lipid profile, and how much does it cost?",
            ["do I need to fast for a lipid profile", "how much does it cost"],
        ),
        ("what are your hours? where are you?", ["what are your hours", "where are you"]),
    ],
)
def test_a_compound_question_is_split(question: str, parts: list[str]) -> None:
    from app.agents.inquiry import _split_question

    assert _split_question(question) == parts


@pytest.mark.parametrize(
    "question",
    [
        "how much is a lipid profile?",
        # "and" inside a noun phrase is not a join between two questions.
        "do you test ferritin and iron studies?",
        "is the Beaverton branch open and nearby",
    ],
)
def test_a_single_question_is_left_alone(question: str) -> None:
    """Splitting a simple question would double the model calls for nothing."""
    from app.agents.inquiry import _split_question

    assert _split_question(question) == []
