"""Scenarios, personas and the per-run variation plan.

A scenario says what the simulated caller knows and what the agent should capture. A persona
says how the caller behaves. The variation plan is generated from a seed and decides which
facts the caller will state wrong before correcting them, so a run is reproducible and the
judge can be told exactly what the caller was supposed to say.
"""

import random
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, model_validator

FieldType = Literal[
    "name",
    "text",
    "date",
    "phone",
    "email",
    "alnum",
    "number",
    "boolean",
    "enum",
    "list",
    "free_text",
]

_CORRECTABLE: tuple[FieldType, ...] = ("phone", "date", "alnum", "email", "number")
_NANP_LENGTH = 11
_FIRST_VOLUNTEERABLE = 2  # fields 0 and 1 come first anyway, so volunteering them is no test

OFF_TOPIC_QUESTIONS: tuple[str, ...] = (
    "Is there parking at the clinic?",
    "Do you take walk-ins on Saturdays?",
    "What are your opening hours on weekdays?",
    "Is the clinic close to a bus stop?",
    "Can I bring my sister with me to the visit?",
    "Do I need to bring a printed copy of my insurance card?",
    "How long does a first appointment usually take?",
)


class FieldSpec(BaseModel):
    """One fact the caller holds and the agent is expected to capture."""

    name: str
    type: FieldType
    expected: Any
    required: bool = True
    spell_out: bool = False
    synonyms: dict[str, list[str]] | None = None
    tolerance: float | None = None


class PassCriteria(BaseModel):
    """The thresholds a call must meet to pass."""

    required: Literal["all"] | list[str] = "all"
    max_wrong: int = 0
    quality_floor: int = 3
    allow_near_miss: bool = False


class CallConfig(BaseModel):
    """Time limits for one call, in seconds."""

    max_seconds: float = 360
    stall_s: float = 6
    greeting_timeout_s: float = 8
    nudge_after_s: float = 8
    max_nudges: int = 2
    slow_response_s: float = 4
    double_talk_s: float = 1.5


class Scenario(BaseModel):
    """A caller's goal and fact sheet, with the criteria for judging the call."""

    id: str
    description: str
    caller_goal: str
    fields: list[FieldSpec]
    pass_criteria: PassCriteria = PassCriteria()
    call: CallConfig = CallConfig()

    @model_validator(mode="after")
    def _unique_field_names(self) -> "Scenario":
        """Reject a scenario in which two fields share a name."""
        seen: set[str] = set()
        for spec in self.fields:
            if spec.name in seen:
                raise ValueError(f"duplicate field name: {spec.name}")
            seen.add(spec.name)
        return self


class Behaviours(BaseModel):
    """How a persona deviates from a plain, cooperative caller."""

    corrections: int = 0
    out_of_order: int = 0
    off_topic: int = 0
    interrupts: bool = False
    verbosity: Literal["terse", "medium", "rambling"] = "medium"
    clarity: Literal["clear", "mumbled"] = "clear"


class Persona(BaseModel):
    """How the simulated caller speaks and behaves, independent of the facts."""

    id: str
    style: str
    behaviours: Behaviours = Behaviours()
    voice: dict[str, str] = {}


class Correction(BaseModel):
    """A field the caller states wrong first and then corrects."""

    field: str
    wrong_value: str
    final_value: str


class OffTopic(BaseModel):
    """An unrelated question the caller asks right after answering a field."""

    after_field: str
    question: str


class VariationPlan(BaseModel):
    """What varies in one run, fixed by the seed."""

    seed: int
    persona_id: str
    corrections: list[Correction] = []
    volunteer_early: list[str] = []
    off_topic: list[OffTopic] = []


def _as_date(value: Any) -> date:
    """Return ``value`` as a date, parsing an ISO string if that is what the YAML held."""
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def _spoken_date(value: date) -> str:
    """Render a date the way a caller says it, for example ``7 March 1988``."""
    return f"{value.day} {value:%B %Y}"


def _spoken_phone(value: str) -> str:
    """Render a phone number as grouped digits, dropping a leading US country code."""
    digits = re.sub(r"\D", "", value)
    if len(digits) == _NANP_LENGTH and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"{digits[:3]} {digits[3:6]} {digits[6:]}"
    return digits


def spoken_value(spec: FieldSpec) -> str:
    """Return the form in which a caller says the field's expected value aloud."""
    match spec.type:
        case "date":
            return _spoken_date(_as_date(spec.expected))
        case "phone":
            return _spoken_phone(str(spec.expected))
        case "list":
            return " and ".join(str(item) for item in spec.expected)
        case "boolean":
            return "yes" if spec.expected else "no"
        case _:
            return str(spec.expected)


def _next_char(char: str) -> str:
    """Return the next letter or digit, wrapping 9 to 0 and Z to A (case preserved)."""
    if char == "9":
        return "0"
    if char in "zZ":
        return "a" if char == "z" else "A"
    return chr(ord(char) + 1)


def _wrong_phone(spoken: str) -> str:
    """Swap the last digit with the nearest earlier digit that differs from it."""
    chars = list(spoken)
    last = max(i for i, c in enumerate(chars) if c.isdigit())
    for i in range(last - 1, -1, -1):
        if chars[i].isdigit() and chars[i] != chars[last]:
            chars[i], chars[last] = chars[last], chars[i]
            break
    return "".join(chars)


def _wrong_alnum(spoken: str) -> str:
    """Replace the last letter or digit with the next one."""
    last = max(i for i, c in enumerate(spoken) if c.isascii() and c.isalnum())
    return spoken[:last] + _next_char(spoken[last]) + spoken[last + 1 :]


def _wrong_email(spoken: str) -> str:
    """Change the last letter before the ``@`` to the next letter."""
    local, at, domain = spoken.partition("@")
    last = max(i for i, c in enumerate(local) if c.isascii() and c.isalpha())
    return local[:last] + _next_char(local[last]) + local[last + 1 :] + at + domain


def _wrong_value(spec: FieldSpec) -> str:
    """Return a plausible near miss for a correctable field's spoken value."""
    spoken = spoken_value(spec)
    match spec.type:
        case "phone":
            return _wrong_phone(spoken)
        case "date":
            return _spoken_date(_as_date(spec.expected) + timedelta(days=1))
        case "alnum":
            return _wrong_alnum(spoken)
        case "email":
            return _wrong_email(spoken)
        case _:
            return str(int(spec.expected) + 1)


def make_plan(scenario: Scenario, persona: Persona, seed: int) -> VariationPlan:
    """Choose, from ``seed``, what varies: corrections, early volunteers and off-topic questions.

    One ``random.Random(seed)`` makes the draws in that order, so corrections keep the values
    they had before the other behaviours existed.
    """
    rng = random.Random(seed)
    b = persona.behaviours
    candidates = [spec for spec in scenario.fields if spec.type in _CORRECTABLE]
    count = min(b.corrections, len(candidates))
    chosen = set(rng.sample(range(len(candidates)), count))
    corrections = [
        Correction(field=spec.name, wrong_value=_wrong_value(spec), final_value=spoken_value(spec))
        for index, spec in enumerate(candidates)
        if index in chosen
    ]
    corrected = {c.field for c in corrections}
    names = [spec.name for spec in scenario.fields]
    volunteerable = [n for n in names[_FIRST_VOLUNTEERABLE:] if n not in corrected]
    volunteered = set(rng.sample(volunteerable, min(b.out_of_order, len(volunteerable))))
    volunteer_early = [n for n in names if n in volunteered]
    after_fields = rng.sample(names[:-1], min(b.off_topic, len(names) - 1))
    questions = rng.sample(OFF_TOPIC_QUESTIONS, len(after_fields))
    off_topic = [
        OffTopic(after_field=f, question=q) for f, q in zip(after_fields, questions, strict=True)
    ]
    return VariationPlan(
        seed=seed,
        persona_id=persona.id,
        corrections=corrections,
        volunteer_early=volunteer_early,
        off_topic=off_topic,
    )


def load_scenario(path: Path) -> Scenario:
    """Read and validate a scenario YAML file."""
    return Scenario.model_validate(yaml.safe_load(path.read_text()))


def load_persona(path: Path) -> Persona:
    """Read and validate a persona YAML file."""
    return Persona.model_validate(yaml.safe_load(path.read_text()))


def persona_path(scenario_path: Path, persona_id: str) -> Path:
    """Return the path of persona ``persona_id``: ``personas/<id>.yaml`` beside the scenario."""
    return scenario_path.parent / "personas" / f"{persona_id}.yaml"
