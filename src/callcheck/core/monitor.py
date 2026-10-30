"""Live turn monitor: decides, while a call runs, when the caller should be nudged.

``TurnMonitor`` is fed every event the mixer records. It tracks the current quiet
spell and says, through ``due``, when it has lasted long enough that the caller
should speak up. It follows the same timing and labelling rules as the offline
detector in ``turns.py`` so a nudge and the incident it later produces agree:

* after the caller's turn ends (``speech_end``) the agent has ``nudge_after_s`` to start
  answering; the spell is ``no_response`` if the agent's endpoint registered the caller
  (heard speech, or a transcript of the caller) and ``caller_not_heard`` if it did not;
* after an agent ``turn_end`` both sides have ``stall_s`` to speak; the spell is
  ``caller_stall`` if the agent's last words were a question and ``dead_end`` if not.

Each spell is reported once. Any new speech or audio, or a nudge, ends it. Timestamps
are milliseconds; the ``CallConfig`` timers are seconds.
"""

from typing import Literal

from callcheck.core.events import CallEvent
from callcheck.core.scenario import CallConfig

NUDGE_TEXT = "The other person has gone quiet. Say: hello, are you still there?"

# How long after an agent turn_end its final transcript may still arrive (as in turns.py).
_ASR_LAG_MS = 1500.0
_NEVER = float("-inf")


class TurnMonitor:
    """Watches a call's events and reports when a nudge is due."""

    def __init__(self, config: CallConfig) -> None:
        """Take the timers from ``config``; the call starts with nothing to wait for."""
        self.max_nudges = config.max_nudges
        self._reply_ms = config.nudge_after_s * 1000
        self._stall_ms = config.stall_s * 1000
        self._since: float | None = None
        self._kind: Literal["reply", "turn"] = "reply"
        self._fired = False
        self._over = False
        self._caller_speaking = False
        self._caller_turn_start = _NEVER
        self._last_heard = _NEVER
        self._agent_open = False
        self._agent_spoke = False
        self._turn_ended = False
        self._turn_end = _NEVER
        self._previous_turn_end = _NEVER
        self._agent_finals: list[tuple[float, str]] = []

    def observe(self, ev: CallEvent) -> None:
        """Take in one recorded event."""
        if self._over:
            return
        if ev.side == "caller":
            self._observe_caller(ev)
        elif ev.side == "agent":
            self._observe_agent(ev)
        elif ev.kind == "nudge":
            self._since = None

    def hold(self, now_ms: float) -> None:
        """Say audio is still playing at ``now_ms``, so a quiet spell cannot have begun yet."""
        if self._since is not None and not self._fired:
            self._since = max(self._since, now_ms)

    def due(self, now_ms: float) -> str | None:
        """Return the label of the nudge that is due at ``now_ms``, or ``None``.

        A spell is reported once; the next one needs new speech or audio first.
        """
        if self._since is None or self._fired or self._over:
            return None
        limit = self._reply_ms if self._kind == "reply" else self._stall_ms
        if now_ms - self._since < limit:
            return None
        self._fired = True
        return self._reply_label() if self._kind == "reply" else self._turn_label()

    def _begin(self, kind: Literal["reply", "turn"], t_ms: float) -> None:
        """Start a quiet spell of ``kind`` at ``t_ms``."""
        self._since, self._kind, self._fired = t_ms, kind, False

    def _observe_caller(self, ev: CallEvent) -> None:
        """Track the caller's speech: it ends any spell, and its end starts a reply wait."""
        if ev.kind == "done":
            self._over = True
            self._since = None
        elif ev.kind in ("speech_start", "audio_start"):
            self._since = None
            self._turn_ended = False
            if ev.kind == "speech_start":
                self._caller_speaking = True
                self._caller_turn_start = ev.t_ms
        elif ev.kind == "speech_end":
            self._caller_speaking = False
            if not self._agent_open:
                self._begin("reply", ev.t_ms)

    def _observe_agent(self, ev: CallEvent) -> None:
        """Track the agent: its audio ends a spell, its turn end starts a stall wait."""
        if ev.kind in ("audio_start", "speech_start"):
            self._agent_open = self._agent_spoke = True
            self._since = None
        elif ev.kind in ("turn_end", "speech_end"):
            self._agent_open = False
            if ev.kind == "turn_end" and self._agent_spoke:
                self._previous_turn_end, self._turn_end = self._turn_end, ev.t_ms
                self._agent_spoke, self._turn_ended = False, True
            if self._turn_ended and not self._caller_speaking:
                self._begin("turn", ev.t_ms)
        elif ev.kind in ("heard_speech_start", "heard_speech_end"):
            self._last_heard = ev.t_ms
        elif ev.kind == "transcript":
            if ev.payload.get("role") == "caller":
                self._last_heard = ev.t_ms
            elif ev.payload.get("role") == "agent" and ev.payload.get("final") is True:
                self._agent_finals.append((ev.t_ms, str(ev.payload.get("text", ""))))

    def _reply_label(self) -> str:
        """Name the wait for an answer: did the agent's endpoint register the caller's turn?"""
        if self._last_heard >= self._caller_turn_start:
            return "no_response"
        return "caller_not_heard"

    def _turn_label(self) -> str:
        """Name the quiet after an agent turn: was its last final transcript a question?"""
        spoken = [
            text
            for t_ms, text in self._agent_finals
            if self._previous_turn_end < t_ms <= self._turn_end + _ASR_LAG_MS
        ]
        if spoken and spoken[-1].rstrip().endswith("?"):
            return "caller_stall"
        return "dead_end"
