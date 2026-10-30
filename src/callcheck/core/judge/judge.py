"""Judge orchestration: turn a finished run directory into a ``JudgeResult``.

Typed fields are graded by the deterministic comparators; free-text fields and conversation
quality go to a chat model behind the ``ChatModel`` protocol. Every model reply is validated
against its pydantic schema and retried once; a reply that never validates leaves that part of
the grade undetermined, and nothing undetermined ever counts as a pass.
"""

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, TypeAdapter, ValidationError

from callcheck.core.chat import ChatModel, ChatResult, Message
from callcheck.core.events import CallEvent
from callcheck.core.judge.compare import compare_field
from callcheck.core.judge.prompts import (
    extraction_messages,
    free_text_messages,
    load_rubric,
    quality_messages,
    render_transcript,
)
from callcheck.core.judge.schema import (
    ExtractedFields,
    FieldVerdict,
    FreeTextJudgement,
    JudgeResult,
    Method,
    QualityReport,
    RunVerdict,
)
from callcheck.core.recorder import build_transcript, read_events
from callcheck.core.scenario import FieldSpec, Scenario, VariationPlan
from callcheck.core.turns import Incident, decisive_incident

_ATTEMPTS = 2
_QUALITY_FAILURES = {
    "repetition": "repetition",
    "confusion": "confusion",
    "correction_handling": "correction_ignored",
    "clarity": "unclear_question",
    "closing": "bad_closing",
}
_FIELD_FAILURES = (
    ("mismatch", "field_mismatch"),
    ("near_miss", "field_near_miss"),
    ("missing", "field_missing"),
)
# Incident labels that, when the agent caused them, fail the run whether or not the call ended.
_AGENT_FAULTS = ("no_response", "dead_end")
_SUMMARY_PHRASES = (
    ("near_miss", "near miss", "near misses"),
    ("mismatch", "mismatched", "mismatched"),
    ("missing", "missing", "missing"),
    ("undetermined", "undetermined", "undetermined"),
)


class _Metered:
    """A ``ChatModel`` wrapper that records the token usage of every call it forwards."""

    def __init__(self, chat: ChatModel) -> None:
        """Wrap ``chat``; the model name is its ``model_id`` attribute, else ``"unknown"``."""
        self._chat = chat
        self._model = str(getattr(chat, "model_id", "unknown"))
        self.usage: list[dict] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """Forward the call, then record its model and input and output token counts."""
        reply = await self._chat.complete(
            messages, json_schema=json_schema, temperature=temperature
        )
        self.usage.append(
            {
                "model": self._model,
                "input_tokens": int(reply.usage.get("input_tokens", 0)),
                "output_tokens": int(reply.usage.get("output_tokens", 0)),
            }
        )
        return reply


def _read_incidents(run_dir: Path) -> list[Incident]:
    """Return the run's ``incidents.json``; a run without one (before M2) has none."""
    path = run_dir / "incidents.json"
    if not path.exists():
        return []
    return TypeAdapter(list[Incident]).validate_json(path.read_text(encoding="utf-8"))


async def _ask[M: BaseModel](
    chat: ChatModel, messages: list[Message], model: type[M], reasons: list[str]
) -> M | None:
    """Ask ``chat`` for a ``model``, retrying once; if it fails twice, note why and return None."""
    error = ""
    for _ in range(_ATTEMPTS):
        reply = await chat.complete(
            messages, json_schema=model.model_json_schema(), temperature=None
        )
        if reply.json is None:
            error = "reply was not JSON"
            continue
        try:
            return model.model_validate(reply.json)
        except ValidationError as exc:
            first = exc.errors()[0]
            error = f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    reasons.append(f"{model.__name__}: {error}")
    return None


def _result_fields(events: list[CallEvent]) -> dict | None:
    """Return the fields of the last ``result`` event that carries any, else ``None``."""
    found = [
        ev.payload["fields"]
        for ev in events
        if ev.kind == "result" and isinstance(ev.payload.get("fields"), dict)
    ]
    return found[-1] if found else None


def _undetermined(spec: FieldSpec, method: Method, note: str) -> FieldVerdict:
    """Build an undetermined verdict for ``spec`` with nothing captured."""
    return FieldVerdict(
        name=spec.name,
        expected=spec.expected,
        captured=None,
        verdict="undetermined",
        method=method,
        note=note,
    )


async def _grade_field(
    spec: FieldSpec, captured: object, method: Method, chat: ChatModel, reasons: list[str]
) -> FieldVerdict:
    """Grade one field, handing a present free-text value to the chat model."""
    typed = compare_field(spec, captured, method)
    if spec.type != "free_text" or typed.verdict == "missing":
        return typed
    ruling = await _ask(chat, free_text_messages(spec, captured), FreeTextJudgement, reasons)
    if ruling is None:
        return typed.model_copy(update={"note": "the judge model gave no usable answer"})
    return typed.model_copy(update={"verdict": ruling.verdict, "note": ruling.reason})


def _required_names(scenario: Scenario) -> set[str]:
    """Return the names of the fields the verdict depends on.

    Raises ``ValueError`` when ``pass_criteria.required`` names a field the scenario lacks,
    since a typo there would let a run pass without checking that field.
    """
    required = scenario.pass_criteria.required
    if required == "all":
        return {spec.name for spec in scenario.fields if spec.required}
    known = {spec.name for spec in scenario.fields}
    for name in required:
        if name not in known:
            raise ValueError(f"unknown required field: {name}")
    return set(required)


def _summary(fields: list[FieldVerdict], quality: QualityReport | None, stalled: bool) -> str:
    """Build one deterministic sentence from the field counts, quality mean and stall flag."""
    tally = Counter(f.verdict for f in fields)
    parts = [f"{tally['match']} of {len(fields)} fields matched"]
    for verdict, singular, plural in _SUMMARY_PHRASES:
        if tally[verdict]:
            parts.append(f"{tally[verdict]} {singular if tally[verdict] == 1 else plural}")
    text = ", ".join(parts)
    if quality is None:
        text += "; quality undetermined"
    else:
        scores = [getattr(quality, dimension).score for dimension in _QUALITY_FAILURES]
        text += f"; quality {sum(scores) / len(scores):.1f}/5"
    return text + ("; stalled." if stalled else ".")


def _is_wrong(field: FieldVerdict, allow_near_miss: bool) -> bool:
    """Report whether ``field`` counts against the run (a near miss only if not allowed)."""
    if field.verdict == "near_miss":
        return not allow_near_miss
    return field.verdict in ("mismatch", "missing")


def _decide(
    scenario: Scenario,
    required_names: set[str],
    fields: list[FieldVerdict],
    quality: QualityReport | None,
    stalled: bool,
    incidents: list[Incident],
) -> tuple[RunVerdict, list[str]]:
    """Apply the verdict algebra and return the verdict with its failure types in order.

    A stall the simulator caused makes the run an ``artifact`` whose only failure type is the
    decisive incident's label: the call never finished for a reason that is not the agent's, so
    field failures are not added. Otherwise a run fails on determinate evidence (too many wrong
    required fields, a stall, an agent ``no_response`` or ``dead_end`` incident, or a quality
    dimension below the floor); it is undetermined if a required field or the quality call is
    open, and passes only when nothing is open. Failure types name only what decided the
    verdict, so wrong fields within ``max_wrong`` are tolerated and not listed. Other incidents
    never change the verdict.
    """
    decisive = decisive_incident(incidents)
    if stalled and decisive is not None and decisive.attribution == "simulator":
        return "artifact", [decisive.label]
    criteria = scenario.pass_criteria
    required = [f for f in fields if f.name in required_names]
    wrong = [f for f in required if _is_wrong(f, criteria.allow_near_miss)]
    failures: list[str] = []
    if len(wrong) > criteria.max_wrong:
        failures = [
            name for verdict, name in _FIELD_FAILURES if any(f.verdict == verdict for f in wrong)
        ]
    low_quality = [
        failure
        for dimension, failure in _QUALITY_FAILURES.items()
        if quality is not None and getattr(quality, dimension).score < criteria.quality_floor
    ]
    agent_labels = [
        i.label for i in incidents if i.attribution == "agent" and i.label in _AGENT_FAULTS
    ]
    if stalled and decisive is not None and decisive.attribution == "agent":
        agent_labels.insert(0, decisive.label)
    if stalled:
        failures.append("stalled")
    failures += dict.fromkeys(agent_labels)
    failures += low_quality
    if failures:
        return "fail", failures
    still_open = quality is None or any(f.verdict == "undetermined" for f in required)
    return ("undetermined" if still_open else "pass"), failures


async def judge_run(run_dir: Path, scenario: Scenario, chat: ChatModel) -> JudgeResult:
    """Grade the finished run in ``run_dir`` against ``scenario`` and write ``judge.json`` there.

    Reads ``events.jsonl``, ``plan.json``, ``outcome.json`` and, when present, ``incidents.json``.
    Records the token usage of every chat call in the result. Captured fields come from the
    last ``result`` event, or from the chat model reading the transcript when there is none.
    A call that ended with reason ``error`` is always ``undetermined`` (fields are still
    compared and reported), with the error as the first undetermined reason.
    """
    required_names = _required_names(scenario)
    events = read_events(run_dir)
    plan = VariationPlan.model_validate_json((run_dir / "plan.json").read_text(encoding="utf-8"))
    outcome = json.loads((run_dir / "outcome.json").read_text(encoding="utf-8"))
    transcript = render_transcript(build_transcript(events))
    incidents = _read_incidents(run_dir)
    reasons: list[str] = []
    chat = _Metered(chat)

    captured = _result_fields(events)
    method: Method = "result"
    if captured is None:
        method = "transcript"
        extracted = await _ask(
            chat, extraction_messages(scenario, transcript), ExtractedFields, reasons
        )
        captured = extracted.fields if extracted else None

    if captured is None:
        fields = [_undetermined(spec, method, "extraction failed") for spec in scenario.fields]
    else:
        fields = [
            await _grade_field(spec, captured.get(spec.name), method, chat, reasons)
            for spec in scenario.fields
        ]
    for field in fields:
        if field.verdict == "undetermined" and field.name in required_names:
            reasons.append(f"field {field.name}: {field.note}")

    quality = await _ask(
        chat,
        quality_messages(scenario, transcript, plan, load_rubric(), incidents),
        QualityReport,
        reasons,
    )
    stalled = outcome.get("reason") == "stalled"
    verdict, failures = _decide(scenario, required_names, fields, quality, stalled, incidents)
    if outcome.get("reason") == "error":
        # A crashed call says nothing about the agent: never a pass, never a blamed field.
        verdict, failures = "undetermined", []
        reasons.insert(0, f"call error: {outcome.get('error') or 'unknown'}")
    result = JudgeResult(
        verdict=verdict,
        failure_types=failures,
        fields=fields,
        quality=quality,
        captured_method=method,
        summary=_summary(fields, quality, stalled),
        undetermined_reasons=reasons,
        incidents=incidents,
        usage=chat.usage,
    )
    (run_dir / "judge.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return result
