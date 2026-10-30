from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from callcheck.core.scenario import (
    Behaviours,
    FieldSpec,
    Persona,
    Scenario,
    load_persona,
    load_scenario,
    make_plan,
    persona_path,
    spoken_value,
)

DUP_YAML = """
id: dup
description: Two fields share a name.
caller_goal: Call.
fields:
  - {name: phone, type: phone, expected: "+1 415 555 0134"}
  - {name: phone, type: phone, expected: "+1 415 555 0199"}
"""

UNKNOWN_TYPE_YAML = """
id: unknown
description: A field with a type that does not exist.
caller_goal: Call.
fields:
  - {name: phone, type: phone_number, expected: "+1 415 555 0134"}
"""

SHORT = Path("scenarios/intake_short.yaml")
BASELINE = Path("scenarios/personas/baseline.yaml")
SELF_CORRECTING = Path("scenarios/personas/self_correcting.yaml")


def test_load_short_scenario():
    s = load_scenario(SHORT)
    assert [f.name for f in s.fields] == [
        "first_name",
        "last_name",
        "date_of_birth",
        "phone",
        "email",
        "reason_for_visit",
    ]
    assert s.fields[2].expected == date(1988, 3, 7) and s.fields[1].spell_out is True
    assert s.call.max_seconds == 240


def test_duplicate_or_unknown_field_rejected(tmp_path):
    bad = tmp_path / "s.yaml"
    bad.write_text(DUP_YAML)
    with pytest.raises(ValueError, match="duplicate field name: phone"):
        load_scenario(bad)
    bad.write_text(UNKNOWN_TYPE_YAML)
    with pytest.raises(ValidationError, match="phone_number"):
        load_scenario(bad)


def test_plan_is_deterministic_and_near_miss():
    s, p = load_scenario(SHORT), load_persona(SELF_CORRECTING)
    a, b = make_plan(s, p, 42), make_plan(s, p, 42)
    assert a == b and len(a.corrections) == 2
    for c in a.corrections:
        assert c.wrong_value != c.final_value
    phone = next((c for c in make_plan(s, p, 7).corrections if c.field == "phone"), None)
    if phone:
        assert phone.wrong_value.endswith("0143")


def test_spoken_values():
    birthday = FieldSpec(name="d", type="date", expected=date(1988, 3, 7))
    assert spoken_value(birthday) == "7 March 1988"
    phone = FieldSpec(name="p", type="phone", expected="+1 415 555 0134")
    assert spoken_value(phone) == "415 555 0134"
    assert (
        spoken_value(FieldSpec(name="l", type="list", expected=["penicillin", "latex"]))
        == "penicillin and latex"
    )
    assert spoken_value(FieldSpec(name="b", type="boolean", expected=True)) == "yes"


def test_baseline_plan_has_no_corrections():
    s = load_scenario(SHORT)
    assert make_plan(s, load_persona(BASELINE), 1).corrections == []


def test_every_correctable_type_yields_a_near_miss():
    for spec, wrong in [
        (FieldSpec(name="d", type="date", expected=date(1988, 3, 7)), "8 March 1988"),
        (FieldSpec(name="a", type="alnum", expected="XQ7-4491-B"), "XQ7-4491-C"),
        (FieldSpec(name="a", type="alnum", expected="AB9"), "AB0"),
        (FieldSpec(name="a", type="alnum", expected="AZ"), "AA"),
        (FieldSpec(name="e", type="email", expected="maria.okonkwo@example.com"),
         "maria.okonkwp@example.com"),
        (FieldSpec(name="n", type="number", expected=41), "42"),
    ]:
        scenario = Scenario(id="x", description="x", caller_goal="x", fields=[spec])
        persona = Persona(id="p", style="s", behaviours=Behaviours(corrections=1))
        (c,) = make_plan(scenario, persona, 0).corrections
        assert (c.wrong_value, c.final_value) == (wrong, spoken_value(spec))


def test_persona_path_sits_beside_the_scenario():
    assert persona_path(SHORT, "baseline") == BASELINE
