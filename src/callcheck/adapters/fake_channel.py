"""Shared machinery for the offline fake caller and fake target.

The fakes cannot recognise speech, so each one speaks a synthetic tone and, at the
same moment, pushes the words onto a ``FakeChannel``. The listener takes those words
off the channel only once its own audio input shows the speaker's tone has ended,
which keeps the text and audio timelines aligned. Everything is driven by the 20 ms
frames passed to ``send_audio``; nothing sleeps, so a whole call runs deterministically
under a virtual clock.
"""

import asyncio
from collections.abc import AsyncIterator

from callcheck.core.audio import EnergyVad, frame_bytes, tone
from callcheck.core.endpoints import VoiceEndpoint
from callcheck.core.events import CallEvent, Side, endpoint_event

CORRECTION_MARKER = ", I mean "
MS_PER_WORD = 60
MIN_UTTERANCE_MS = 300


def spoken_name(field: str) -> str:
    """Return a field name as it is spoken in a question: underscores become spaces."""
    return field.replace("_", " ")


class FakeChannel:
    """Two text queues carrying what each fake says to the other, one entry per utterance."""

    def __init__(self) -> None:
        """Create empty queues for the agent's input and the caller's input."""
        self.to_agent: asyncio.Queue[str] = asyncio.Queue()
        self.to_caller: asyncio.Queue[str] = asyncio.Queue()


class FakeEndpoint(VoiceEndpoint):
    """Base for the fakes: queues events, speaks tones, and hears the other side's text."""

    input_rate = 16000
    output_rate = 24000

    def __init__(
        self,
        side: Side,
        inbox: asyncio.Queue[str],
        outbox: asyncio.Queue[str],
        freq_hz: float,
    ) -> None:
        """Bind the speaker's side, the channel queues it reads and writes, and its tone pitch."""
        self._side = side
        self._inbox = inbox
        self._outbox = outbox
        self._freq_hz = freq_hz
        self._queue: asyncio.Queue[CallEvent] = asyncio.Queue()
        self._vad = EnergyVad()

    async def start(self) -> None:
        """Begin the session; a fake has nothing to open unless a subclass speaks first."""

    async def send_audio(self, pcm: bytes) -> None:
        """Hear one frame; when the other side's tone ends, take its words off the channel."""
        if self._vad.feed(pcm) == "end" and not self._inbox.empty():
            self._hear(self._inbox.get_nowait())

    async def events(self) -> AsyncIterator[CallEvent]:
        """Yield queued events, waiting for more, until the ``closed`` event has been yielded."""
        while True:
            ev = await self._queue.get()
            yield ev
            if ev.kind == "closed":
                return

    def drain(self) -> list[CallEvent]:
        """Return and clear the events queued so far without waiting, for frame-stepped drivers."""
        drained: list[CallEvent] = []
        while not self._queue.empty():
            drained.append(self._queue.get_nowait())
        return drained

    async def close(self) -> None:
        """End the session by queueing the final ``closed`` event."""
        self._emit("closed")

    def _hear(self, text: str) -> None:
        """React to the words of one finished utterance from the other side."""
        raise NotImplementedError

    def _emit(self, kind: str, **payload: object) -> None:
        """Queue an event on this fake's own side."""
        self._queue.put_nowait(endpoint_event(self._side, kind, **payload))

    def _speak(self, text: str) -> None:
        """Emit the transcript, push the words to the other side, then emit the tone as frames."""
        self._emit("transcript", role=self._side, text=text, final=True)
        self._outbox.put_nowait(text)
        ms = max(MIN_UTTERANCE_MS, MS_PER_WORD * len(text.split()))
        pcm = tone(self._freq_hz, ms, self.output_rate)
        step = frame_bytes(self.output_rate)
        for start in range(0, len(pcm), step):
            self._emit("audio_out", pcm=pcm[start : start + step])
