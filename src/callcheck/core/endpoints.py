"""Protocols for the two ends of a call and the data that configures them.

The caller (a simulated customer) and the target (the agent under test) both
speak PCM16 audio and emit ``CallEvent`` records; they differ only in what else
they expose. Concrete implementations live in ``callcheck.adapters``.
"""

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from callcheck.core.events import CallEvent


class VoiceEndpoint(Protocol):
    """A party on the call that takes audio in and reports events out."""

    input_rate: int
    output_rate: int

    async def start(self) -> None:
        """Open the connection and begin the session."""
        ...

    async def send_audio(self, pcm: bytes) -> None:
        """Deliver one frame of PCM16 audio heard by this party."""
        ...

    def events(self) -> AsyncIterator[CallEvent]:
        """Stream the events this party emits until it closes."""
        ...

    async def close(self) -> None:
        """End the session and release resources."""
        ...


class TargetAgent(VoiceEndpoint, Protocol):
    """The agent under test, which may record a structured result."""

    def result(self) -> dict | None:
        """Return the result the agent recorded, or ``None`` if it has not."""
        ...


class Caller(VoiceEndpoint, Protocol):
    """The simulated customer, which can be nudged back on script."""

    async def nudge(self, instruction: str) -> bool:
        """Steer the caller with an instruction; return whether it was accepted."""
        ...


@dataclass
class ToolSpec:
    """Description of a tool the agent may call, with its JSON input schema."""

    name: str
    description: str
    input_schema: dict


ToolExecutor = Callable[[str, dict], Awaitable[dict]]


@dataclass
class AgentDefinition:
    """Everything needed to stand up the agent under test."""

    system_prompt: str
    tools: list[ToolSpec]
    tool_executor: ToolExecutor
    voice: dict[str, str]
    result: Callable[[], dict | None]


@dataclass
class CallerBrief:
    """The persona and voice a simulated caller plays."""

    persona_id: str
    system_prompt: str
    voice: dict[str, str]
    greeting_timeout_s: float = 8.0
