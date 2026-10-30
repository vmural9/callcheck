"""The prompts the judge sends to its chat model.

Each ``*_messages`` function builds the full message list for one kind of question: extracting
field values from a transcript, judging a free-text field by meaning, and scoring conversation
quality. All of them tell the model to answer with JSON only, since the judge validates every
reply against a pydantic schema. The rubric text lives in ``rubric.md`` so it can be edited
without touching code.
"""

from collections.abc import Sequence
from importlib import resources

from callcheck.core.chat import Message
from callcheck.core.scenario import FieldSpec, Scenario, VariationPlan
from callcheck.core.turns import Incident

_SYSTEM = (
    "You are an impartial evaluator of phone calls between a simulated caller and a voice agent. "
    "Judge only what the transcript shows. Answer only with JSON matching the schema you are "
    "given, with no text before or after it."
)


def load_rubric() -> str:
    """Return the text of the editable quality rubric shipped beside this module."""
    return resources.files("callcheck.core.judge").joinpath("rubric.md").read_text("utf-8")


def render_transcript(transcript: list[dict]) -> str:
    """Render utterances as numbered lines such as ``0: CALLER: Hello``, one per line.

    The numbers are the indices the model must cite as evidence.
    """
    return "\n".join(f"{i}: {u['side'].upper()}: {u['text']}" for i, u in enumerate(transcript))


def _messages(user: str) -> list[Message]:
    """Wrap ``user`` with the shared system message."""
    return [Message("system", _SYSTEM), Message("user", user)]


def extraction_messages(scenario: Scenario, transcript: str) -> list[Message]:
    """Ask for the value the agent recorded or confirmed for each scenario field."""
    names = "\n".join(f"- {spec.name} ({spec.type})" for spec in scenario.fields)
    return _messages(
        "Below is a transcript of a call in which the AGENT collects information from the "
        "CALLER. For each field listed, give the value the agent last stated, or confirmed it had "
        "recorded, by the end of the call. Take it from the AGENT's words, even if the caller "
        "corrected it and the agent ignored the correction, and even if it is wrong. Never "
        "substitute what the caller said. Use null for a field the agent never stated or "
        "confirmed.\n\n"
        f"Fields:\n{names}\n\n"
        f"Transcript:\n{transcript}\n\n"
        'Answer with JSON of the form {"fields": {"<field name>": "<value or null>"}}, '
        "with every field name above present."
    )


def free_text_messages(spec: FieldSpec, captured: object) -> list[Message]:
    """Ask whether the captured value conveys the expected meaning, ignoring wording."""
    return _messages(
        "Decide whether the captured text conveys the same meaning as the expected text. Judge "
        "meaning, not wording: a paraphrase that keeps every essential fact is a match; a "
        "different, contradictory or much vaguer statement is a mismatch. Answer undetermined "
        "only if the texts cannot be compared.\n\n"
        f"Field: {spec.name}\n"
        f"Expected: {spec.expected}\n"
        f"Captured: {captured}\n\n"
        'Answer with JSON of the form {"verdict": "match" | "mismatch" | "undetermined", '
        '"reason": "<one sentence>"}.'
    )


def _plan_lines(plan: VariationPlan) -> str:
    """Describe everything the caller was instructed to do, one sentence per instruction."""
    lines = [
        f"The caller was instructed to first say {c.wrong_value} then correct to "
        f"{c.final_value} for field {c.field}."
        for c in plan.corrections
    ]
    if not plan.corrections:
        lines.append("The caller was not instructed to correct anything.")
    if plan.volunteer_early:
        lines.append(
            "The caller volunteered these fields before being asked: "
            f"{', '.join(plan.volunteer_early)}."
        )
    lines += [
        f'The caller was instructed to ask "{q.question}" after giving {q.after_field}.'
        for q in plan.off_topic
    ]
    return "\n".join(lines)


def _incident_lines(incidents: Sequence[Incident]) -> str:
    """List the detected turn-taking incidents, one per line, or say there were none."""
    if not incidents:
        return "No turn-taking incidents were detected."
    return (
        "Turn-taking incidents detected from the call timing (t_ms is milliseconds):\n"
        + "\n".join(
            f"- {i.label} ({i.attribution}) at t_ms={i.t_ms:.0f}: {i.reason}" for i in incidents
        )
    )


def quality_messages(
    scenario: Scenario,
    transcript: str,
    plan: VariationPlan,
    rubric_text: str,
    incidents: Sequence[Incident] = (),
) -> list[Message]:
    """Ask for the five 0 to 5 quality scores, each with a reason and transcript line evidence."""
    return _messages(
        "Score the AGENT's conversation quality on the five dimensions of the rubric below.\n\n"
        f"Rubric:\n{rubric_text}\n\n"
        f"The caller's goal: {' '.join(scenario.caller_goal.split())}\n"
        f"{_plan_lines(plan)}\n\n"
        f"{_incident_lines(incidents)}\n\n"
        f"Transcript, one numbered line per utterance:\n{transcript}\n\n"
        "Answer with a JSON object whose keys are repetition, confusion, correction_handling, "
        'clarity and closing. Each value is an object with an integer "score" from 0 to 5, a '
        'short "reason", and "evidence": the 0-based numbers of the transcript lines that '
        "justify the score (an empty list if none apply)."
    )
