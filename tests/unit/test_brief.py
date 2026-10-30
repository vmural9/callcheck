from pathlib import Path

from callcheck.core.brief import build_brief, preflight_brief
from callcheck.core.scenario import OffTopic, load_persona, load_scenario, make_plan

SCENARIOS = Path(__file__).parents[2] / "scenarios"
s = load_scenario(SCENARIOS / "intake_short.yaml")
baseline = load_persona(SCENARIOS / "personas" / "baseline.yaml")
self_correcting = load_persona(SCENARIOS / "personas" / "self_correcting.yaml")


def test_brief_contains_facts_corrections_and_rules():
    b = build_brief(s, self_correcting, make_plan(s, self_correcting, 42))
    assert "- first_name: Maria" in b.system_prompt and "7 March 1988" in b.system_prompt
    assert "When asked for your " in b.system_prompt and "end_call" in b.system_prompt
    assert "Wait for the other person to speak first" in b.system_prompt
    assert b.greeting_timeout_s == 8


def test_baseline_brief_says_no_corrections():
    b = build_brief(s, baseline, make_plan(s, baseline, 1))
    assert "# Corrections to make\nNone." in b.system_prompt


def test_brief_carries_persona_identity_and_voice():
    b = build_brief(s, self_correcting, make_plan(s, self_correcting, 42))
    assert b.persona_id == "self_correcting" and b.voice == self_correcting.voice
    assert "You sometimes give a value, then correct yourself" in b.system_prompt
    assert "8 seconds" in b.system_prompt


def test_preflight_brief_is_a_fixed_script():
    b = preflight_brief()
    assert b.persona_id == "preflight" and b.voice == {} and b.greeting_timeout_s == 8.0
    assert "Do not call any tool." in b.system_prompt


def test_spell_out_note_spells_once_then_accepts_the_read_back():
    # Live run 2026-10-30: the caller hears read-backs through speech recognition, which
    # cannot show spelling ("Okonkwo-Reyes" came back as "oconquo reyes"), so a rule to
    # re-spell whenever the read-back is "wrong" (or does not "sound right") looped for the
    # rest of the call: the caller may spell once, then accepts the next read-back.
    b = build_brief(s, baseline, make_plan(s, baseline, 1))
    assert "- last_name: Okonkwo-Reyes (if asked" in b.system_prompt
    assert "spell it out letter by letter, but only once" in b.system_prompt
    assert "accept the next read-back" in b.system_prompt


def _plan_brief(**plan_fields):
    plan = make_plan(s, baseline, 1).model_copy(update=plan_fields)
    return build_brief(s, baseline, plan).system_prompt


def test_correction_waits_for_read_back():
    plan = make_plan(s, self_correcting, 42)
    text = build_brief(s, self_correcting, plan).system_prompt
    (c, *_) = plan.corrections
    assert f"When asked for your {c.field}, say '{c.wrong_value}'" in text
    assert "after the agent reads it back" in text
    assert "in the same breath" not in text


def test_no_unscripted_corrections():
    text = _plan_brief()
    assert "Do not correct any other value unless the agent reads it back wrong" in text


def test_volunteer_section():
    text = _plan_brief(volunteer_early=["email"])
    assert "# Things to mention early\n" in text
    assert "your email" in text and "maria.okonkwo@example.com" in text.split("# Things")[1]


def test_off_topic_section():
    off_topic = [OffTopic(after_field="phone", question="Is there parking at the clinic?")]
    text = _plan_brief(off_topic=off_topic)
    assert "After you answer your phone, ask: 'Is there parking at the clinic?'" in text


def test_empty_sections_say_none():
    text = _plan_brief()
    assert "# Things to mention early\nNone." in text
    assert "# Questions to ask\nNone." in text
