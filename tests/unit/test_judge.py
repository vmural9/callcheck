"""Tests for judge orchestration, driven by a scripted fake chat model."""

import json
from pathlib import Path

import pytest

from callcheck.adapters.chat import make_chat
from callcheck.adapters.chat.fake import FakeChat
from callcheck.core.events import CallEvent, event_to_json
from callcheck.core.judge.judge import judge_run
from callcheck.core.scenario import Correction, Scenario, VariationPlan, load_scenario

SCENARIO_PATH = Path(__file__).parents[2] / "scenarios" / "intake_short.yaml"

QUALITY_KEYS = ("repetition", "confusion", "correction_handling", "clarity", "closing")
GOOD_QUALITY = {k: {"score": 5, "reason": "fine", "evidence": []} for k in QUALITY_KEYS}
MATCH = {"verdict": "match", "reason": "same meaning"}

GOOD_FIELDS = {
    "first_name": "Maria",
    "last_name": "Okonkwo-Reyes",
    "date_of_birth": "March 7th, 1988",
    "phone": "415 555 0134",
    "email": "maria.okonkwo@example.com",
    "reason_for_visit": "my knee hurts since I ran two weeks ago",
}

TRANSCRIPT = [
    ("agent", "Thanks for calling the clinic. What is your first name?"),
    ("caller", "Maria."),
    ("agent", "And your phone number?"),
    ("caller", "It is 415 555 0143. Sorry, I mean 415 555 0134."),
    ("agent", "Thank you, I have 415 555 0143."),
]


def _write_run(
    path: Path,
    *,
    fields: dict | None,
    corrections: list[Correction] = (),
    outcome: dict | None = None,
) -> Path:
    """Write a run directory holding events, plan and outcome, and return it."""
    path.mkdir()
    events = [CallEvent(0, "harness", "call_start", {})]
    for i, (side, text) in enumerate(TRANSCRIPT):
        payload = {"text": text, "final": True, "role": side}
        events.append(CallEvent(1000 * (i + 1), side, "transcript", payload))
    if fields is not None:
        events.append(CallEvent(9000, "agent", "result", {"fields": fields, "submitted": True}))
    (path / "events.jsonl").write_text(
        "".join(event_to_json(ev) + "\n" for ev in events), encoding="utf-8"
    )
    plan = VariationPlan(seed=1, persona_id="self_correcting", corrections=list(corrections))
    (path / "plan.json").write_text(plan.model_dump_json(), encoding="utf-8")
    (path / "outcome.json").write_text(
        json.dumps(outcome or {"reason": "caller_done"}), encoding="utf-8"
    )
    return path


@pytest.fixture
def scenario() -> Scenario:
    """The six-field intake scenario."""
    return load_scenario(SCENARIO_PATH)


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A finished run whose result event matches every field."""
    return _write_run(tmp_path / "run", fields=GOOD_FIELDS)


@pytest.fixture
def run_dir_wrong_phone(tmp_path: Path) -> Path:
    """A run where the caller corrected the phone number and the agent kept the wrong one."""
    correction = Correction(field="phone", wrong_value="415 555 0143", final_value="415 555 0134")
    return _write_run(
        tmp_path / "wrong",
        fields=dict(GOOD_FIELDS, phone="415 555 0143"),
        corrections=[correction],
    )


@pytest.fixture
def run_dir_stalled(tmp_path: Path) -> Path:
    """A run whose fields all match but which ended because the agent stalled."""
    return _write_run(tmp_path / "stalled", fields=GOOD_FIELDS, outcome={"reason": "stalled"})


@pytest.fixture
def run_dir_no_result(tmp_path: Path) -> Path:
    """A run with no result event, so captured fields must come from the transcript."""
    return _write_run(tmp_path / "noresult", fields=None)


async def test_pass_when_fields_match_and_quality_high(run_dir, scenario):
    chat = FakeChat([MATCH, GOOD_QUALITY])
    r = await judge_run(run_dir, scenario, chat)
    assert r.verdict == "pass" and r.captured_method == "result" and r.failure_types == []
    assert (run_dir / "judge.json").exists()
    assert r.summary == "6 of 6 fields matched; quality 5.0/5."


async def test_corrected_phone_ignored_is_fail_with_mismatch(run_dir_wrong_phone, scenario):
    low = dict(GOOD_QUALITY, correction_handling={"score": 1, "reason": "kept it", "evidence": [4]})
    chat = FakeChat([MATCH, low])
    r = await judge_run(run_dir_wrong_phone, scenario, chat)
    assert r.verdict == "fail" and {"field_mismatch", "correction_ignored"} <= set(r.failure_types)
    assert r.failure_types == ["field_mismatch", "correction_ignored"]
    assert r.summary == "5 of 6 fields matched, 1 mismatched; quality 4.2/5."


async def test_quality_prompt_carries_rubric_transcript_and_plan(run_dir_wrong_phone, scenario):
    chat = FakeChat([MATCH, GOOD_QUALITY])
    await judge_run(run_dir_wrong_phone, scenario, chat)
    prompt = "\n".join(m.content for m in chat.calls[-1])
    assert "correction_handling" in prompt and "impartial evaluator" in prompt
    assert "4: AGENT: Thank you, I have 415 555 0143." in prompt
    assert "first say 415 555 0143 then correct to 415 555 0134 for field phone" in prompt


async def test_unparseable_quality_is_undetermined_not_pass(run_dir, scenario):
    bad = dict(GOOD_QUALITY, clarity={"score": 7, "reason": "", "evidence": []})
    chat = FakeChat([MATCH, "not json", bad])
    r = await judge_run(run_dir, scenario, chat)
    assert r.verdict == "undetermined" and r.quality is None and len(r.undetermined_reasons) == 1
    assert r.undetermined_reasons[0].startswith("QualityReport: clarity.score")


async def test_quality_retry_recovers(run_dir, scenario):
    r = await judge_run(run_dir, scenario, FakeChat([MATCH, "oops", GOOD_QUALITY]))
    assert r.verdict == "pass" and r.undetermined_reasons == []


async def test_stalled_outcome_fails(run_dir_stalled, scenario):
    r = await judge_run(run_dir_stalled, scenario, FakeChat([MATCH, GOOD_QUALITY]))
    assert r.verdict == "fail" and r.failure_types == ["stalled"]
    assert r.summary.endswith("; stalled.")


async def test_transcript_extraction_when_no_result_event(run_dir_no_result, scenario):
    extracted = {
        "fields": {
            "first_name": "Maria",
            "last_name": None,
            "date_of_birth": "1988-03-07",
            "phone": "415 555 0134",
            "email": "maria.okonkwo@example.com",
            "reason_for_visit": "knee pain",
        }
    }
    chat = FakeChat([extracted, MATCH, GOOD_QUALITY])
    r = await judge_run(run_dir_no_result, scenario, chat)
    assert r.captured_method == "transcript" and r.verdict == "fail"
    assert r.failure_types == ["field_missing"]
    assert {f.method for f in r.fields} == {"transcript"}
    asked = "\n".join(m.content for m in chat.calls[0])
    assert "first_name" in asked
    assert "the agent last stated" in asked and "caller's final value" not in asked


async def test_failed_extraction_is_undetermined_not_missing(run_dir_no_result, scenario):
    r = await judge_run(run_dir_no_result, scenario, FakeChat(["x", "y", GOOD_QUALITY]))
    assert r.verdict == "undetermined" and r.failure_types == []
    assert {f.verdict for f in r.fields} == {"undetermined"}
    assert r.undetermined_reasons[0].startswith("ExtractedFields")


async def test_free_text_mismatch_fails_and_missing_skips_the_model(tmp_path, scenario):
    mismatch = {"verdict": "mismatch", "reason": "different complaint"}
    run = _write_run(tmp_path / "ft", fields=GOOD_FIELDS)
    r = await judge_run(run, scenario, FakeChat([mismatch, GOOD_QUALITY]))
    free = next(f for f in r.fields if f.name == "reason_for_visit")
    assert (free.verdict, free.note, free.method) == ("mismatch", "different complaint", "result")
    assert r.verdict == "fail" and r.failure_types == ["field_mismatch"]

    run = _write_run(tmp_path / "ft2", fields=dict(GOOD_FIELDS, reason_for_visit=""))
    chat = FakeChat([GOOD_QUALITY])
    r = await judge_run(run, scenario, chat)
    assert r.verdict == "fail" and r.failure_types == ["field_missing"]
    assert len(chat.calls) == 1


async def test_failed_free_text_call_is_undetermined(run_dir, scenario):
    r = await judge_run(run_dir, scenario, FakeChat(["a", "b", GOOD_QUALITY]))
    assert r.verdict == "undetermined"
    assert r.undetermined_reasons[0].startswith("FreeTextJudgement")


async def test_near_miss_counts_as_wrong_unless_allowed(tmp_path, scenario):
    fields = dict(GOOD_FIELDS, last_name="Okonkwo-Reyez")
    run = _write_run(tmp_path / "near", fields=fields)
    r = await judge_run(run, scenario, FakeChat([MATCH, GOOD_QUALITY]))
    assert r.verdict == "fail" and r.failure_types == ["field_near_miss"]

    criteria = scenario.pass_criteria.model_copy(update={"allow_near_miss": True})
    lenient = scenario.model_copy(update={"pass_criteria": criteria})
    r = await judge_run(run, lenient, FakeChat([MATCH, GOOD_QUALITY]))
    assert r.verdict == "pass" and r.failure_types == []


async def test_fake_chat_records_calls_and_raises_when_exhausted():
    chat = FakeChat([{"a": 1}, "text"])
    first = await chat.complete([])
    second = await chat.complete([])
    assert first.json == {"a": 1} and second.json is None and second.text == "text"
    assert len(chat.calls) == 2
    with pytest.raises(AssertionError, match="FakeChat exhausted"):
        await chat.complete([])


async def test_max_wrong_tolerates_a_mismatch_without_listing_it(tmp_path, scenario):
    run = _write_run(tmp_path / "tolerated", fields=dict(GOOD_FIELDS, phone="415 555 0143"))
    criteria = scenario.pass_criteria.model_copy(update={"max_wrong": 1})
    tolerant = scenario.model_copy(update={"pass_criteria": criteria})
    r = await judge_run(run, tolerant, FakeChat([MATCH, GOOD_QUALITY]))
    assert r.verdict == "pass" and r.failure_types == []
    assert next(f for f in r.fields if f.name == "phone").verdict == "mismatch"


async def test_unknown_required_field_is_rejected(run_dir, scenario):
    criteria = scenario.pass_criteria.model_copy(update={"required": ["phone", "fone"]})
    typo = scenario.model_copy(update={"pass_criteria": criteria})
    chat = FakeChat([])
    with pytest.raises(ValueError, match="unknown required field: fone"):
        await judge_run(run_dir, typo, chat)
    assert chat.calls == []


def test_make_chat_builds_fake_and_rejects_unknown():
    assert isinstance(make_chat("fake", responses=[]), FakeChat)
    with pytest.raises(KeyError, match="unknown chat"):
        make_chat("nope")
