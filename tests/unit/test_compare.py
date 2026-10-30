"""Tests for the deterministic typed field comparators."""

from datetime import date

from callcheck.core.judge.compare import compare_field
from callcheck.core.scenario import FieldSpec


def P(t, e, **k):  # noqa: N802
    """Build a one-field spec of type ``t`` expecting ``e``."""
    return FieldSpec(name="f", type=t, expected=e, **k)


def V(spec, captured):  # noqa: N802
    """Return the verdict string for ``captured`` against ``spec``."""
    return compare_field(spec, captured, "result").verdict


def test_phone_formats_match():
    assert V(P("phone", "+1 415 555 0134"), "(415) 555-0134") == "match"
    assert V(P("phone", "+1 415 555 0134"), "415 555 0143") == "mismatch"


def test_date_forms():
    assert V(P("date", date(1988, 3, 7)), "March 7th, 1988") == "match"
    assert V(P("date", date(1988, 3, 7)), "1988-07-03") == "mismatch"
    assert V(P("date", date(1988, 3, 7)), "sometime in spring") == "undetermined"


def test_name_similarity_bands():
    assert V(P("name", "Okonkwo-Reyes"), "okonkwo reyes") == "match"
    assert V(P("name", "Okonkwo-Reyes"), "Okonkwo Rayes") == "near_miss"
    assert V(P("name", "Okonkwo-Reyes"), "Smith") == "mismatch"
    assert V(P("name", "Maria"), "Mariah") == "near_miss"


def test_missing_email_enum_list_boolean():
    assert V(P("email", "maria.okonkwo@example.com"), None) == "missing"
    assert V(P("email", "maria.okonkwo@example.com"), "Maria.Okonkwo@Example.com") == "match"
    enum = P("enum", "new_patient", synonyms={"new_patient": ["new patient"]})
    assert V(enum, "New Patient") == "match"
    assert V(P("list", ["penicillin", "latex"]), ["Latex", "penicillin"]) == "match"
    assert V(P("boolean", True), "yes") == "match"


def test_free_text_defers():
    assert V(P("free_text", "knee pain"), "sore knee") == "undetermined"


def test_notes_say_what_differed():
    verdict = compare_field(P("phone", "+1 415 555 0134"), "415 555 0143", "result")
    assert "4155550134 vs 4155550143" in verdict.note


def test_short_phone_is_undetermined():
    assert V(P("phone", "+1 415 555 0134"), "555 0134") == "undetermined"


def test_number_alnum_list_string_and_empty():
    assert V(P("number", 3), "3.0") == "match"
    assert V(P("number", 3), "three") == "undetermined"
    assert V(P("alnum", "AB-12"), "ab 12") == "match"
    assert V(P("list", ["penicillin", "latex"]), "Latex and penicillin") == "match"
    assert V(P("text", "x"), "") == "missing"
    assert V(P("list", ["a"]), []) == "missing"
    assert V(P("boolean", True), "maybe") == "undetermined"
    assert compare_field(P("boolean", False), "no", "transcript").method == "transcript"


def test_tolerance_moves_bands():
    assert V(P("text", "knee pain", tolerance=0.99), "knee pains") == "near_miss"
