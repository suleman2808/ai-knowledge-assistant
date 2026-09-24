"""Tests for relative date resolution.

The conventions encoded here are choices, not facts — English is
genuinely ambiguous about "next Tuesday". What matters is that the
choice is consistent and stated, which a model cannot guarantee.

Every case fixes `today` explicitly, so these tests mean the same thing
on any day of the week.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.agents.dates import resolve

# All reference dates below are 2026, chosen so the weekday is explicit.
THURSDAY = date(2026, 9, 24)
TUESDAY = date(2026, 9, 22)
SUNDAY = date(2026, 9, 27)


def test_the_production_bug() -> None:
    """On Thursday 24 September the model booked "next Tuesday" as 6
    October — a week late. It had resolved the same phrase correctly in
    testing, which is exactly why this is code now."""
    assert resolve("next tuesday at 2pm", today=THURSDAY) == date(2026, 9, 29)


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("today", THURSDAY),
        ("tomorrow", date(2026, 9, 25)),
        ("day after tomorrow", date(2026, 9, 26)),
        ("in 3 days", date(2026, 9, 27)),
        ("in 2 weeks", date(2026, 10, 8)),
    ],
)
def test_offsets_from_today(phrase: str, expected: date) -> None:
    assert resolve(phrase, today=THURSDAY) == expected


def test_next_weekday_means_the_following_week() -> None:
    """"Next Friday" on a Thursday is not tomorrow."""
    assert resolve("next friday", today=THURSDAY) == date(2026, 10, 2)


def test_bare_weekday_means_the_soonest_one() -> None:
    assert resolve("can I come in on monday", today=THURSDAY) == date(2026, 9, 28)


def test_bare_weekday_today_means_today() -> None:
    """A patient saying "Thursday" on a Thursday means now, not in a week."""
    assert resolve("thursday", today=THURSDAY) == THURSDAY


def test_this_weekday_after_it_has_passed_moves_forward() -> None:
    """Said on Thursday, "this Tuesday" cannot mean two days ago."""
    assert resolve("this tuesday", today=THURSDAY) == date(2026, 9, 29)


def test_week_on_weekday_is_seven_days_beyond_the_next_one() -> None:
    assert resolve("a week on tuesday", today=THURSDAY) == date(2026, 10, 6)


def test_week_on_is_not_confused_with_a_bare_weekday() -> None:
    """"Week on Tuesday" contains "Tuesday"; specificity must win."""
    assert resolve("week on tuesday", today=THURSDAY) != resolve("tuesday", today=THURSDAY)


def test_next_week_is_the_monday() -> None:
    assert resolve("next week", today=THURSDAY) == date(2026, 9, 28)


def test_next_weekday_from_a_sunday() -> None:
    """Sunday is the end of the week, where off-by-one errors live."""
    assert resolve("next monday", today=SUNDAY) == date(2026, 9, 28)
    assert resolve("monday", today=SUNDAY) == date(2026, 9, 28)


def test_next_weekday_from_that_same_weekday() -> None:
    """"Next Tuesday" said on a Tuesday means a week today."""
    assert resolve("next tuesday", today=TUESDAY) == date(2026, 9, 29)
    assert resolve("tuesday", today=TUESDAY) == TUESDAY


@pytest.mark.parametrize("phrase", ["", None, "sometime soon", "when you have space",
                                    "the 14th", "asap"])
def test_unrecognised_phrases_defer_to_the_model(phrase: str | None) -> None:
    """Returning None is how this module says "not my job" — the caller
    falls back to the date the model extracted."""
    assert resolve(phrase, today=THURSDAY) is None


def test_abbreviations_and_case_are_handled() -> None:
    assert resolve("NEXT TUES", today=THURSDAY) == date(2026, 9, 29)
    assert resolve("next Weds", today=THURSDAY) is None, "Weds is not a spelling we accept"
    assert resolve("next wed", today=THURSDAY) == date(2026, 9, 30)


def test_phrase_embedded_in_a_sentence() -> None:
    assert resolve(
        "could I come in next tuesday afternoon please", today=THURSDAY
    ) == date(2026, 9, 29)
