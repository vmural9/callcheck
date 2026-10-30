"""Builds the reference intake agent from ``intake.yaml`` for a chosen set of fields and bugs.

The YAML holds the voice, the system prompt template and the twenty intake fields. A run picks
which of those fields to collect and which planted bugs to switch on; the loader validates the
choice, renders the prompt, and wires the tools to a fresh ``FormStore``.
"""

import re
from functools import cache
from importlib.resources import files

import yaml

from callcheck.core.endpoints import AgentDefinition, ToolSpec
from callcheck.reference_agent.tools import BUGS, FormStore

INTAKE_YAML = files("callcheck.reference_agent") / "intake.yaml"


@cache
def _spec() -> dict:
    """Read and cache the parsed ``intake.yaml``."""
    return yaml.safe_load(INTAKE_YAML.read_text())


def _tidy(prompt: str) -> str:
    """Collapse runs of blank lines into a single paragraph break and end with one newline."""
    return re.sub(r"\n{3,}", "\n\n", prompt).strip() + "\n"


def load_reference_agent(
    field_names: list[str], bugs: frozenset[str] = frozenset(BUGS)
) -> AgentDefinition:
    """Return the intake agent collecting ``field_names`` in order, with ``bugs`` planted.

    Raises ``KeyError`` naming the first field that ``intake.yaml`` does not define.
    """
    spec = _spec()
    for name in field_names:
        if name not in spec["fields"]:
            raise KeyError(name)
    fields_block = "\n".join(f"- {n}: {spec['fields'][n]['question']}" for n in field_names)
    order_key = "strict_order_sentence" if "strict_order" in bugs else "early_field_sentence"
    store = FormStore(field_names, {n: spec["fields"][n]["kind"] for n in field_names}, bugs)
    tools = [
        ToolSpec(
            name="save_field",
            description="Save the value the caller gave for one intake field.",
            input_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": list(field_names)},
                    "value": {"type": "string"},
                },
                "required": ["name", "value"],
            },
        ),
        ToolSpec(
            name="submit_form",
            description="Submit the form once every field has been saved.",
            input_schema={"type": "object", "properties": {}},
        ),
    ]
    return AgentDefinition(
        system_prompt=_tidy(
            spec["system_prompt_template"].format(
                fields_block=fields_block, strict_order_block=spec[order_key]
            )
        ),
        tools=tools,
        tool_executor=store.execute,
        voice=spec["voice"],
        result=store.snapshot,
    )
