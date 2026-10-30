"""A scripted caller that answers a form's questions from a fixed set of answers.

``FakeCaller`` listens for a question naming one of its fields, answers it, and says
goodbye when the agent reports the form complete. A field listed in ``corrections``
is first answered wrongly and then corrected ("415 555 0143, I mean 415 555 0134").
With a ``script`` it instead speaks the next scripted line after each agent utterance
and, once the lines run out, stays silent. A nudge makes it say "Hello?", unless it
was built with ``nudgeable=False``, when it refuses and stays quiet.
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
        script: list[str] | None = None,
        nudgeable: bool = True,
    ) -> None:
        """Answer from ``answers``; for fields in ``corrections`` give that value first.

        When ``script`` is given, speak its lines in order instead of answering. With
        ``nudgeable=False`` a nudge is refused.
        """
        super().__init__("caller", channel.to_caller, channel.to_agent, 440.0)
        self._answers = answers
        self._corrections = corrections or {}
        self._script = None if script is None else list(script)
        self._nudgeable = nudgeable

    async def nudge(self, instruction: str) -> bool:
        """Prompt the agent with "Hello?" whatever the instruction says; refuse if not nudgeable."""
        if not self._nudgeable:
            return False
        self._speak("Hello?")
        return True

    def _hear(self, text: str) -> None:
        """Answer the field the agent just asked about, or say goodbye once it is complete."""
        self._emit("transcript", role="agent", text=text, final=True)
        if self._script is not None:
            if self._script:
                self._speak(self._script.pop(0))
            return
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
