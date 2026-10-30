from pathlib import Path

from callcheck.core.brief import build_brief, preflight_brief
from callcheck.core.scenario import load_persona, load_scenario, make_plan

SCENARIOS = Path(__file__).parents[2] / "scenarios"
s = load_scenario(SCENARIOS / "intake_short.yaml")
baseline = load_persona(SCENARIOS / "personas" / "baseline.yaml")
self_correcting = load_persona(SCENARIOS / "personas" / "self_correcting.yaml")


def test_brief_contains_facts_corrections_and_rules():
    b = build_brief(s, self_correcting, make_plan(s, self_correcting, 42))
    assert "- first_name: Maria" in b.system_prompt and "7 March 1988" in b.system_prompt
    assert "first say '" in b.system_prompt and "end_call" in b.system_prompt
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


def test_spell_out_note_accepts_a_read_back_that_sounds_right():
    # Live run 2026-10-30: the caller hears read-backs through speech recognition, which
    # cannot show spelling ("Okonkwo-Reyes" came back as "oconquo reyes"), so a rule to
    # re-spell whenever the read-back is "wrong" looped for the rest of the call.
    b = build_brief(s, baseline, make_plan(s, baseline, 1))
    assert "- last_name: Okonkwo-Reyes (spell it out" in b.system_prompt
    assert "accept a read-back that sounds right" in b.system_prompt
