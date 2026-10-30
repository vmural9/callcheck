"""The agent under test, hosted on an Amazon Nova Sonic session.

``NovaSonicTarget`` gives a Nova Sonic session an ``AgentDefinition``: its prompt, its
tools, and local execution of those tools. The captured form is reported as a ``result``
event after each tool result, so the call's log shows when the form changed.
"""

from collections.abc import AsyncIterator, Callable

from callcheck.adapters.providers.nova_sonic_client import NovaSonicSession
from callcheck.core.endpoints import AgentDefinition, TargetAgent
from callcheck.core.events import CallEvent, endpoint_event

DEFAULT_VOICE = "matthew"
CONNECTED_TURN = "(call connected)"


class NovaSonicTarget(TargetAgent):
    """Target agent backed by a Nova Sonic session that runs ``agent``'s tools locally."""

    def __init__(
        self,
        agent: AgentDefinition,
        *,
        model_id: str,
        region: str,
        profile: str | None,
        session_factory: Callable[..., NovaSonicSession] = NovaSonicSession,
    ) -> None:
        """Build the session for ``agent``; nothing is opened until ``start``."""
        self._agent = agent
        self._session = session_factory(
            side="agent",
            model_id=model_id,
            region=region,
            profile=profile,
            system_prompt=agent.system_prompt,
            voice_id=agent.voice.get("nova_sonic", DEFAULT_VOICE),
            tools=agent.tools,
            tool_executor=agent.tool_executor,
        )
        self.input_rate = self._session.input_rate
        self.output_rate = self._session.output_rate

    async def start(self) -> None:
        """Open the session, then send the text turn that makes the agent greet the caller.

        The model answers once the call mixer's audio frames arrive, so no audio is sent here.
        """
        await self._session.start()
        await self._session.send_text(CONNECTED_TURN)

    async def send_audio(self, pcm: bytes) -> None:
        """Deliver one frame of the caller's audio to the agent."""
        await self._session.send_audio(pcm)

    async def events(self) -> AsyncIterator[CallEvent]:
        """Yield the session's events, adding a ``result`` event after each ``tool_result``."""
        async for ev in self._session.events():
            yield ev
            if ev.kind == "tool_result" and (form := self._agent.result()) is not None:
                yield endpoint_event("agent", "result", **form)

    def result(self) -> dict | None:
        """Return the form the agent has captured so far, or ``None`` if it has none."""
        return self._agent.result()

    async def close(self) -> None:
        """End the session."""
        await self._session.close()
