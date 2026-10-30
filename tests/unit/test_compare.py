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


def test_phone_with_seven_to_nine_digits_is_a_mismatch():
    spec = P("phone", "+1 415 555 0134")
    for captured in ("555 0134", "415 555 013", "(415) 555-013"):
        out = compare_field(spec, captured, "result")
        assert out.verdict == "mismatch" and out.note.startswith("too few digits:")


def test_phone_with_fewer_than_seven_digits_or_none_is_undetermined():
    spec = P("phone", "+1 415 555 0134")
    assert V(spec, "555 013") == "undetermined"
    assert V(spec, "four one five") == "undetermined"


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


def test_partial_dates_are_undetermined_whatever_today_is():
    expected = P("date", date(1988, 3, 7))
    assert V(expected, "March 7th") == "undetermined"
    assert V(expected, "1988") == "undetermined"
    assert V(expected, "March 7th, 1988") == "match"
    note = compare_field(expected, "March 7th", "result").note
    assert "partial" in note and "'March 7th'" in note


def test_doctor_matches_dr():
    assert V(P("name", "Dr. Priya Nair"), "doctor priya nair") == "match"


def test_spoken_digit_run_matches_digits():
    assert V(P("text", "218 Valencia Street"), "two one eight valencia street") == "match"


def test_single_number_word_untouched():
    from callcheck.core.judge.compare import normalize_text

    assert normalize_text("one way street") == "one way street"


def _visit_type():
    return P(
        "enum",
        "new_patient",
        synonyms={
            "new_patient": ["new patient", "first visit"],
            "follow_up": ["follow up", "returning patient"],
        },
    )


def test_enum_phrase_inside_capture():
    assert V(_visit_type(), "new patient visit") == "match"


def test_enum_two_options_stays_mismatch():
    out = compare_field(_visit_type(), "new patient follow up", "result")
    assert out.verdict == "mismatch"
    assert "new_patient" in out.note and "follow_up" in out.note


LONG_ADDRESS = "two one eight valencia street apartment four b san francisco"


def test_long_address_one_digit_off_does_not_match():
    assert V(P("text", LONG_ADDRESS), LONG_ADDRESS) == "match"
    off = LONG_ADDRESS.replace("four", "nine")
    assert V(P("text", LONG_ADDRESS), off) != "match"
    assert "digits differ" in compare_field(P("text", LONG_ADDRESS), off, "result").note


def test_spoken_digit_differs_is_not_match():
    assert V(P("text", "218 valencia street"), "two one nine valencia street") != "match"


def test_dr_smith_not_drive_smith():
    assert V(P("name", "dr smith"), "drive smith") != "match"


def test_enum_negation_is_undetermined():
    assert V(_visit_type(), "not a new patient") == "undetermined"
    assert V(_visit_type(), "this isn't a new patient") == "undetermined"


def test_enum_negation_contractions_with_and_without_apostrophe():
    for lead in ("wasn't", "wasnt", "isn't", "isnt", "don't", "dont", "doesn't", "doesnt", "never"):
        assert V(_visit_type(), f"{lead} a new patient visit") == "undetermined", lead
    assert V(_visit_type(), "no new patient visit") == "undetermined"


def test_enum_only_other_option_phrase_is_mismatch():
    assert V(_visit_type(), "returning patient visit") == "mismatch"
