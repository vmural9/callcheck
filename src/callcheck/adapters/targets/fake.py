"""A scripted target agent that fills in a form field by field, with no network.

``FakeTarget`` greets the caller, asks for each field in turn, and stores whatever
the caller says. Behaviours let a test provoke the failures the harness must catch:
``ignore_correction`` keeps the first value of a self-correction, and
``stall_after:<field>`` goes silent once that field is stored.
"""

from callcheck.adapters.fake_channel import (
    CORRECTION_MARKER,
    FakeChannel,
    FakeEndpoint,
    spoken_name,
)
from callcheck.core.endpoints import TargetAgent


class FakeTarget(FakeEndpoint, TargetAgent):
    """Target that collects ``fields`` one question at a time and reports what it stored."""

    def __init__(
        self,
        channel: FakeChannel,
        fields: list[str],
        *,
        behaviours: set[str] = frozenset(),
    ) -> None:
        """Ask for ``fields`` in order; ``behaviours`` selects the deliberate faults to show."""
        super().__init__("agent", channel.to_agent, channel.to_caller, 660.0)
        self._fields = list(fields)
        self._behaviours = behaviours
        self._stored: dict[str, str] = {}
        self._silenced = False

    async def start(self) -> None:
        """Greet the caller and ask for the first field."""
        self._say(f"Thanks for calling the clinic. {self._question(self._fields[0])}")

    def result(self) -> dict | None:
        """Return the fields stored so far."""
        return dict(self._stored)

    def _hear(self, text: str) -> None:
        """Store the caller's answer to the current question, then ask the next or finish."""
        if self._silenced:
            return
        self._emit("heard_speech_end")
        self._emit("transcript", role="caller", text=text, final=True)
        field = self._fields[len(self._stored)]
        value = self._value_from(text)
        self._stored[field] = value
        self._emit("tool_call", name="save_field", arguments={"field": field, "value": value})
        self._emit("tool_result", name="save_field", result={"ok": True})
        if f"stall_after:{field}" in self._behaviours:
            self._silenced = True
        elif len(self._stored) < len(self._fields):
            self._say(self._question(self._fields[len(self._stored)]))
        else:
            self._silenced = True
            self._say("Your form is complete. Goodbye.", finished=True)

    def _value_from(self, text: str) -> str:
        """Apply the self-correction rule: keep the corrected value unless told to ignore it."""
        first, marker, corrected = text.partition(CORRECTION_MARKER)
        if not marker:
            return text
        return first if "ignore_correction" in self._behaviours else corrected

    def _question(self, field: str) -> str:
        """Return the question that asks for ``field``."""
        return f"What is your {spoken_name(field)}?"

    def _say(self, text: str, *, finished: bool = False) -> None:
        """Speak one agent turn, reporting the stored result when the form is finished."""
        self._emit("turn_start")
        self._speak(text)
        if finished:
            self._emit("result", fields=dict(self._stored), submitted=True)
        self._emit("turn_end")
