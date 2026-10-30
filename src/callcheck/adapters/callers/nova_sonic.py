"""The simulated caller, played by an Amazon Nova Sonic session.

``NovaSonicCaller`` gives a Nova Sonic session a ``CallerBrief`` and one tool, ``end_call``,
which the caller uses to hang up once the agent has finished; the call ends on the ``done``
event that follows it. Mid-call nudges reach the session as text turns.
"""

from collections.abc import AsyncIterator, Callable

from callcheck.adapters.providers.nova_sonic_client import NovaSonicSession
from callcheck.core.endpoints import Caller, CallerBrief, ToolSpec
from callcheck.core.events import CallEvent, endpoint_event

DEFAULT_VOICE = "tiffany"
END_CALL = ToolSpec(
    "end_call",
    "Call this once the agent has confirmed the form is complete and you have said goodbye.",
    {"type": "object", "properties": {}},
)


async def _execute(name: str, arguments: dict) -> dict:
    """Answer the model's ``end_call``; the ``done`` event is emitted from the result."""
    return {"ok": True}


class NovaSonicCaller(Caller):
    """Caller backed by a Nova Sonic session playing ``brief``."""

    def __init__(
        self,
        brief: CallerBrief,
        *,
        model_id: str,
        region: str,
        profile: str | None,
        session_factory: Callable[..., NovaSonicSession] = NovaSonicSession,
    ) -> None:
        """Build the session for ``brief``; nothing is opened until ``start``."""
        self._session = session_factory(
            side="caller",
            model_id=model_id,
            region=region,
            profile=profile,
            system_prompt=brief.system_prompt,
            voice_id=brief.voice.get("nova_sonic", DEFAULT_VOICE),
            tools=[END_CALL],
            tool_executor=_execute,
        )
        self.input_rate = self._session.input_rate
        self.output_rate = self._session.output_rate
        self._done = False
        self._closed = False

    async def start(self) -> None:
        """Open the session; the caller then waits to hear the agent."""
        await self._session.start()

    async def send_audio(self, pcm: bytes) -> None:
        """Deliver one frame of the agent's audio to the caller."""
        await self._session.send_audio(pcm)

    async def events(self) -> AsyncIterator[CallEvent]:
        """Yield the session's events, adding one ``done`` right after ``end_call`` returns."""
        async for ev in self._session.events():
            yield ev
            if ev.kind == "closed":
                self._closed = True
            elif ev.kind == "tool_result" and ev.payload.get("name") == END_CALL.name:
                if not self._done:
                    self._done = True
                    yield endpoint_event("caller", "done")

    async def nudge(self, instruction: str) -> bool:
        """Send ``instruction`` to the caller as a text turn; ``False`` if the session is over."""
        if self._closed:
            return False
        try:
            await self._session.send_text(instruction)
        except RuntimeError:
            return False
        return True

    async def close(self) -> None:
        """End the session."""
        self._closed = True
        await self._session.close()
