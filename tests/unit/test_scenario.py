from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from callcheck.core.scenario import (
    OFF_TOPIC_QUESTIONS,
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
NEW_PATIENT = Path("scenarios/intake_new_patient.yaml")
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
        (
            FieldSpec(name="e", type="email", expected="maria.okonkwo@example.com"),
            "maria.okonkwp@example.com",
        ),
        (FieldSpec(name="n", type="number", expected=41), "42"),
    ]:
        scenario = Scenario(id="x", description="x", caller_goal="x", fields=[spec])
        persona = Persona(id="p", style="s", behaviours=Behaviours(corrections=1))
        (c,) = make_plan(scenario, persona, 0).corrections
        assert (c.wrong_value, c.final_value) == (wrong, spoken_value(spec))


def test_persona_path_sits_beside_the_scenario():
    assert persona_path(SHORT, "baseline") == BASELINE


def _persona(**behaviours):
    return Persona(id="p", style="s", behaviours=Behaviours(**behaviours))


def test_plan_is_deterministic_with_all_behaviours():
    s = load_scenario(NEW_PATIENT)
    p = _persona(corrections=2, out_of_order=2, off_topic=1)
    plan = make_plan(s, p, 7)
    assert plan == make_plan(s, p, 7)
    names = [f.name for f in s.fields]
    assert len(plan.volunteer_early) == 2
    assert all(names.index(n) >= 2 for n in plan.volunteer_early)
    assert len(plan.off_topic) == 1
    assert plan.off_topic[0].question in OFF_TOPIC_QUESTIONS
    assert plan.off_topic[0].after_field in names[:-1]
    assert len(OFF_TOPIC_QUESTIONS) >= 6


def test_corrections_keep_their_seeded_draw_when_other_behaviours_are_added():
    s = load_scenario(NEW_PATIENT)
    plain = make_plan(s, _persona(corrections=2), 11)
    busy = make_plan(s, _persona(corrections=2, out_of_order=2, off_topic=2), 11)
    assert plain.corrections == busy.corrections


def test_volunteer_and_corrections_do_not_overlap():
    s = load_scenario(NEW_PATIENT)
    p = _persona(corrections=3, out_of_order=5)
    for seed in range(50):
        plan = make_plan(s, p, seed)
        corrected = {c.field for c in plan.corrections}
        assert not corrected & set(plan.volunteer_early)


def test_shipped_personas_load():
    paths = sorted(Path("scenarios/personas").glob("*.yaml"))
    assert len(paths) == 7
    for path in paths:
        assert load_persona(path).id == path.stem


def test_intake_new_patient_matches_reference_agent():
    s = load_scenario(NEW_PATIENT)
    agent = yaml.safe_load(Path("src/callcheck/reference_agent/intake.yaml").read_text())
    assert [f.name for f in s.fields] == list(agent["fields"])
    assert len(s.fields) == 20
    (dob,) = [f for f in s.fields if f.name == "date_of_birth"]
    assert dob.expected.day <= 12
