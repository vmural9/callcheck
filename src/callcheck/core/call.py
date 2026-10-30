"""The call mixer: one clock, two endpoints, and a timestamped record of everything.

``Call`` places a single call between a caller and a target. Endpoints deliver audio in
bursts and report events as they happen; the mixer paces the audio to real time in 20 ms
ticks, sends each side what the other said (silence when nothing is queued), watches both
directions for speech, and records every event and every frame through a ``Sink``. It
also decides when the call is over: the caller is done, the target closed, nobody has
made a sound for too long, the time limit passed, or something broke. It does not judge
the call or decide what is stored.
"""

import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Literal

from callcheck.core.audio import FRAME_MS, EnergyVad, Resampler, frame_bytes, rms_dbfs, silence
from callcheck.core.clock import Clock
from callcheck.core.endpoints import Caller, TargetAgent, VoiceEndpoint
from callcheck.core.events import CallEvent, Side, Sink
from callcheck.core.playout import PlayoutBuffer

TICK_S = FRAME_MS / 1000

EndReason = Literal["caller_done", "target_closed", "stalled", "max_duration", "error"]


@dataclass
class CallSettings:
    """Limits and timings for one call."""

    max_seconds: float = 360.0
    stall_s: float = 6.0
    grace_ms: int = 1500
    record_rate: int = 24000
    audio_gap_ms: int = 300


@dataclass
class CallOutcome:
    """How a call ended, how long it lasted, and the error text when it failed."""

    reason: EndReason
    duration_ms: float
    error: str | None = None

    def as_dict(self) -> dict:
        """Return the outcome as a plain dict for the sink and the run summary."""
        return dataclasses.asdict(self)


@dataclass
class _Lane:
    """Everything the mixer keeps for the audio one side speaks.

    Audio from the speaker is resampled once for the listener's playout buffer and once
    for the recording buffer, so what is recorded is paced exactly like what is heard.
    """

    to_listener: Resampler
    playout: PlayoutBuffer
    to_record: Resampler
    recording: PlayoutBuffer
    vad: EnergyVad = field(default_factory=EnergyVad)
    last_audio_at: float | None = None


class Call:
    """Places one call between ``caller`` and ``target`` and returns how it ended."""

    def __init__(
        self,
        *,
        caller: Caller,
        target: TargetAgent,
        sink: Sink,
        clock: Clock,
        settings: CallSettings | None = None,
    ) -> None:
        """Bind the two endpoints, the sink that records the call, and the clock that paces it.

        ``settings`` defaults to a fresh ``CallSettings()``.
        """
        settings = settings or CallSettings()
        self._caller = caller
        self._target = target
        self._sink = sink
        self._clock = clock
        self._settings = settings
        self._endpoints: dict[Side, VoiceEndpoint] = {"caller": caller, "agent": target}
        self._lanes = {
            "caller": self._lane(speaker=caller, listener=target),
            "agent": self._lane(speaker=target, listener=caller),
        }
        self._t0 = 0.0
        self._last_activity = 0.0
        self._done_at: float | None = None
        self._ending: tuple[EndReason, str | None] | None = None

    async def run(self) -> CallOutcome:
        """Run the call to its end, close everything, and return the outcome."""
        self._t0 = self._last_activity = self._clock.now()
        consumers: list[asyncio.Task[None]] = []
        outcome: CallOutcome | None = None
        try:
            self._record("harness", "call_start")
            await self._target.start()
            await self._caller.start()
            next_tick = self._write_connect_gap()
            consumers = [
                asyncio.create_task(self._consume("agent")),
                asyncio.create_task(self._consume("caller")),
            ]
            outcome = await self._tick_until_end(next_tick)
        except Exception as exc:
            self._record("harness", "error", message=str(exc), recoverable=False)
            outcome = self._conclude("error", str(exc))
        finally:
            await self._shutdown(consumers)
            self._sink.close((outcome or self._conclude("error", "call cancelled")).as_dict())
        return outcome

    async def _shutdown(self, consumers: list[asyncio.Task[None]]) -> None:
        """Stop the consumers and close both endpoints, whatever any one close does."""
        for task in consumers:
            task.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)
        for side, endpoint in self._endpoints.items():
            try:
                await endpoint.close()
            except Exception as exc:
                self._record(side, "error", message=f"close failed: {exc}", recoverable=True)

    def _write_connect_gap(self) -> float:
        """Record the time spent connecting as silence and return when the ticker resumes.

        The recording starts at call start, so a frame's position in it equals the
        ``t_ms`` of the events around it; the endpoints hear nothing until both are up.
        """
        frames = round((self._clock.now() - self._t0) / TICK_S)
        gap = silence(self._settings.record_rate)
        for _ in range(frames):
            self._sink.write_audio(gap, gap)
        return self._t0 + frames * TICK_S

    def _lane(self, *, speaker: VoiceEndpoint, listener: VoiceEndpoint) -> _Lane:
        """Build the lane for ``speaker``, whose audio is heard by ``listener``."""
        record_rate = self._settings.record_rate
        return _Lane(
            to_listener=Resampler(speaker.output_rate, listener.input_rate),
            playout=PlayoutBuffer(listener.input_rate),
            to_record=Resampler(speaker.output_rate, record_rate),
            recording=PlayoutBuffer(record_rate),
        )

    def _now_ms(self) -> float:
        """Return milliseconds since the call began."""
        return (self._clock.now() - self._t0) * 1000

    def _record(self, side: Side, kind: str, **payload: object) -> None:
        """Stamp an event with the current call time and hand it to the sink."""
        self._sink.record(CallEvent(self._now_ms(), side, kind, dict(payload)))

    def _conclude(self, reason: EndReason, error: str | None = None) -> CallOutcome:
        """Record ``call_end`` and build the outcome for a call that ended now."""
        outcome = CallOutcome(reason, self._now_ms(), error)
        self._record("harness", "call_end", reason=reason)
        return outcome

    def _request_end(self, reason: EndReason, error: str | None = None) -> None:
        """Ask the ticker to end the call; the first request wins."""
        if self._ending is None:
            self._ending = (reason, error)

    async def _consume(self, side: Side) -> None:
        """Handle every event ``side`` emits, ending the call if the consumer itself fails."""
        try:
            async for ev in self._endpoints[side].events():
                self._handle(side, ev)
        except Exception as exc:
            self._record("harness", "error", message=str(exc), recoverable=False)
            self._request_end("error", str(exc))

    def _handle(self, side: Side, ev: CallEvent) -> None:
        """Route one endpoint event: audio to the lanes, everything else to the record."""
        if ev.kind == "audio_out":
            self._take_audio(side, ev.payload["pcm"])
            return
        self._sink.record(dataclasses.replace(ev, t_ms=self._now_ms()))
        if ev.kind == "interrupted":
            lane = self._lanes[side]
            lane.playout.flush()
            lane.recording.flush()
        elif ev.kind == "done" and side == "caller":
            self._done_at = self._clock.now()
        elif ev.kind == "closed" and side == "agent":
            self._request_end("target_closed")
        elif ev.kind == "error" and not ev.payload.get("recoverable", True):
            self._request_end("error", ev.payload.get("message"))

    def _take_audio(self, side: Side, pcm: bytes) -> None:
        """Queue a burst of audio from ``side`` for playout and recording."""
        lane = self._lanes[side]
        now = self._clock.now()
        gap_s = self._settings.audio_gap_ms / 1000
        if lane.last_audio_at is None or now - lane.last_audio_at >= gap_s:
            self._record(side, "audio_start")
        lane.last_audio_at = self._last_activity = now
        dropped = lane.playout.push(lane.to_listener.process(pcm))
        lane.recording.push(lane.to_record.process(pcm))
        if dropped:
            self._record("harness", "error", message="playout overflow", dropped=dropped)

    async def _tick_until_end(self, next_tick: float) -> CallOutcome:
        """Mix one 20 ms frame per tick, starting after ``next_tick``, until the call ends."""
        while True:
            next_tick += TICK_S
            await self._clock.sleep_until(next_tick)
            outcome = self._check_end()
            if outcome:
                return outcome
            await self._mix_frame()

    def _check_end(self) -> CallOutcome | None:
        """Return the outcome if the call should end on this tick, else ``None``."""
        if self._ending:
            return self._conclude(*self._ending)
        now = self._clock.now()
        if any(lane.playout.pending_ms > 0 for lane in self._lanes.values()):
            self._last_activity = now
        if now - self._t0 >= self._settings.max_seconds:
            return self._conclude("max_duration")
        if now - self._last_activity > 2 * self._settings.stall_s:
            self._record("harness", "watchdog", reason="stalled")
            return self._conclude("stalled")
        if self._done_at is not None and self._finished_speaking(now):
            return self._conclude("caller_done")
        return None

    def _finished_speaking(self, now: float) -> bool:
        """Report whether, after ``done``, both sides were heard out or the grace ran out."""
        drained = all(lane.playout.pending_ms == 0 for lane in self._lanes.values())
        return drained or (now - self._done_at) * 1000 >= self._settings.grace_ms

    async def _mix_frame(self) -> None:
        """Send each endpoint its frame, run the speech detectors, and record the audio."""
        heard_by_caller = self._lanes["agent"].playout.pull(frame_bytes(self._caller.input_rate))
        heard_by_target = self._lanes["caller"].playout.pull(frame_bytes(self._target.input_rate))
        await self._caller.send_audio(heard_by_caller)
        await self._target.send_audio(heard_by_target)
        recorded = {
            side: lane.recording.pull(frame_bytes(self._settings.record_rate))
            for side, lane in self._lanes.items()
        }
        for side, frame in recorded.items():
            transition = self._lanes[side].vad.feed(frame)
            if transition:
                self._record(side, f"speech_{transition}", energy=rms_dbfs(frame))
        self._sink.write_audio(recorded["caller"], recorded["agent"])
