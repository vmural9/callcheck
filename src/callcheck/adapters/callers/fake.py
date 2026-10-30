"""A scripted caller that answers a form's questions from a fixed set of answers.

``FakeCaller`` listens for a question naming one of its fields, answers it, and says
goodbye when the agent reports the form complete. A field listed in ``corrections``
is first answered wrongly and then corrected ("415 555 0143, I mean 415 555 0134").
"""

from callcheck.adapters.fake_channel import (
    CORRECTION_MARKER,
    FakeChannel,
    FakeEndpoint,
    spoken_name,
)
from callcheck.core.endpoints import Caller


class FakeCaller(FakeEndpoint, Caller):
    """Caller that answers each question from ``answers`` and ends the call on completion."""

    def __init__(
        self,
        channel: FakeChannel,
        answers: dict[str, str],
        *,
        corrections: dict[str, str] | None = None,
    ) -> None:
        """Answer from ``answers``; for fields in ``corrections`` give that value first."""
        super().__init__("caller", channel.to_caller, channel.to_agent, 440.0)
        self._answers = answers
        self._corrections = corrections or {}

    async def nudge(self, instruction: str) -> bool:
        """Prompt the agent with "Hello?" whatever the instruction says; always accepted."""
        self._speak("Hello?")
        return True

    def _hear(self, text: str) -> None:
        """Answer the field the agent just asked about, or say goodbye once it is complete."""
        lowered = text.lower()
        if "complete" in lowered:
            self._speak("Goodbye.")
            self._emit("done")
            return
        asked = [f for f in self._answers if spoken_name(f).lower() in lowered]
        if not asked:
            return
        field = max(asked, key=len)
        if field in self._corrections:
            self._speak(f"{self._corrections[field]}{CORRECTION_MARKER}{self._answers[field]}")
        else:
            self._speak(self._answers[field])
