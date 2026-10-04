"""Tests for recognising returning customers by phone number.

Recognition is not authentication, and these tests are written to keep
that distinction visible: it fills in details the customer already gave
us, and it never unlocks anything.
"""

from __future__ import annotations

import pytest

from app.agents.booking import handle_booking
from app.graph.build import identify
from app.integrations import patients
from app.integrations.analytics import log_turn
from app.integrations.calendar import InMemoryCalendar
from app.integrations.phone import normalise, phone_in
from tests.test_agents import next_weekday, stub_extraction


def record_booking(name: str = "Sarah Chen", phone: str = "503-555-0180",
                   service: str = "full blood count") -> None:
    log_turn(
        {
            "session_id": "s1",
            "message": f"book a {service}, {name}, {phone}",
            "intent": "booking",
            "answer": "booked",
            "sources": [],
            "success": True,
            "needs_followup": False,
            "agent_metadata": {
                "stage": "booked",
                "event_id": "evt-1",
                "start": "2026-10-06T09:00:00",
                "end": "2026-10-06T09:30:00",
                "calendar_backend": "google",
                "extracted": {"service": service, "patient_name": name, "phone": phone},
            },
        }
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "written",
    ["503-555-0180", "(503) 555 0180", "+1 503 555 0180", "5035550180",
     "503.555.0180", "+1-503-555-0180"],
)
def test_one_person_however_they_type_their_number(written: str) -> None:
    """Six spellings, one customer. Without this they are six customers."""
    assert normalise(written) == "5035550180"


@pytest.mark.parametrize("not_a_number", ["12345", "", None, "room 4", "2026"])
def test_things_that_are_not_phone_numbers_are_rejected(not_a_number) -> None:  # noqa: ANN001
    assert normalise(not_a_number) is None


def test_a_number_is_found_inside_a_sentence() -> None:
    assert phone_in("sure, it's 503-555-0180, call any time") == "5035550180"


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


def test_an_unknown_number_is_simply_unknown() -> None:
    assert patients.find("503-555-9999") is None


def test_a_returning_customer_is_found_with_their_history() -> None:
    record_booking(service="full blood count")
    record_booking(service="lipid profile")

    found = patients.find("+1 503 555 0180")

    assert found is not None
    assert found["name"] == "Sarah Chen"
    assert found["visit_count"] == 2
    assert {b["service"] for b in found["bookings"]} == {"full blood count", "lipid profile"}


def test_lookup_failure_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database problem means "not recognised", never a broken reply."""

    def _broken(*_a, **_k):  # noqa: ANN002, ANN003
        raise RuntimeError("database is locked")

    monkeypatch.setattr("app.integrations.patients.connect", _broken)

    assert patients.find("503-555-0180") is None


def test_description_is_terse_enough_to_sit_in_a_prompt() -> None:
    record_booking()

    described = patients.describe(patients.find("503-555-0180"))

    assert "Sarah Chen" in described
    assert len(described.splitlines()) <= 5, "history must not crowd out retrieved documents"
    assert "new customer" in patients.describe(None)


# ---------------------------------------------------------------------------
# The identify node
# ---------------------------------------------------------------------------


def test_identify_recognises_a_number_in_the_message() -> None:
    record_booking()

    result = identify({"message": "it's 503-555-0180, can I rebook?", "history": []})

    assert result["patient"]["name"] == "Sarah Chen"


def test_recognition_persists_for_the_rest_of_the_conversation() -> None:
    """Someone gives their number once and expects to stay known."""
    record_booking()
    history = [
        {"role": "user", "content": "my number is 503-555-0180"},
        {"role": "assistant", "content": "Thanks."},
    ]

    result = identify({"message": "and what time do you open?", "history": history})

    assert result["patient"] is not None


def test_a_stranger_stays_a_stranger() -> None:
    assert identify({"message": "how much is a blood test?", "history": []})["patient"] is None


# ---------------------------------------------------------------------------
# What recognition is for
# ---------------------------------------------------------------------------


def test_a_known_customer_is_not_asked_for_their_name_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the feature: fewer questions, not a login."""
    target = next_weekday(1)
    stub_extraction(
        monkeypatch,
        {"service": "lipid profile", "date": target.isoformat(), "date_phrase": None,
         "time": "14:00", "time_preference": None, "patient_name": None,
         "phone": None, "notes": None},
    )

    response = handle_booking(
        "book a lipid profile",
        backend=InMemoryCalendar(appointments=[]),
        patient={"name": "Sarah Chen", "phone": "5035550180", "visit_count": 2,
                 "bookings": []},
    )

    assert response.metadata["stage"] == "booked", "should not have stopped to ask"
    assert "Sarah Chen" in response.answer


def test_what_the_customer_types_now_beats_the_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """People book on behalf of someone else."""
    target = next_weekday(1)
    stub_extraction(
        monkeypatch,
        {"service": "lipid profile", "date": target.isoformat(), "date_phrase": None,
         "time": "14:00", "time_preference": None, "patient_name": "Imran Khalid",
         "phone": "503-555-0111", "notes": None},
    )

    response = handle_booking(
        "book a lipid profile for Imran Khalid, 503-555-0111",
        backend=InMemoryCalendar(appointments=[]),
        patient={"name": "Sarah Chen", "phone": "5035550180", "visit_count": 2,
                 "bookings": []},
    )

    assert "Imran Khalid" in response.answer
    assert "Sarah Chen" not in response.answer


# ---------------------------------------------------------------------------
# Someone identifying themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected",
    [
        ("hi, this is (503) 555 0180", True),
        ("my number is 503-555-0180", True),
        ("503-555-0180", True),
        ("+1 503 555 0180 thanks", True),
        # Still a real question, and still out of scope.
        ("my number is 503-555-0180, what time does the cinema open?", False),
        # A request with a number attached is not an introduction.
        ("book a test, 503-555-0180", False),
        ("hello there", False),
    ],
)
def test_identifying_yourself_is_told_apart_from_asking_something(
    message: str, expected: bool
) -> None:
    from app.graph.build import _is_mostly_contact_details

    assert _is_mostly_contact_details(message) is expected


def test_a_recognised_customer_giving_their_number_is_welcomed() -> None:
    """Found by using it: "hi, this is 503-555-0180" was answered with
    "that's outside what I can help with" — to someone who had just said
    who they are."""
    from app.graph.build import other_node

    result = other_node(
        {
            "message": "hi, this is 503-555-0180",
            "routed_by": "llm",
            "routing_reason": "shares contact details",
            "patient": {"name": "Sarah Chen", "visit_count": 2, "bookings": []},
        }
    )

    assert "Sarah" in result["answer"]
    assert "outside what I can help with" not in result["answer"]
    assert result["agent_metadata"]["kind"] == "identified"


def test_an_unrecognised_number_is_not_pretended_to_be_known() -> None:
    from app.graph.build import other_node

    result = other_node(
        {"message": "hi, this is 503-555-7777", "routed_by": "llm",
         "routing_reason": "shares contact details", "patient": None}
    )

    assert result["agent_metadata"]["kind"] == "out_of_scope"
