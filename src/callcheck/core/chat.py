"""Text-chat model protocol used by the judge.

The judge needs one capability from a language model: complete a conversation,
optionally constrained to a JSON schema. Providers implement ``ChatModel`` in
``callcheck.adapters``; the core depends only on this shape.
"""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass
class Message:
    """One turn of a chat conversation."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatResult:
    """A model's reply: raw text, parsed JSON when a schema was given, and usage."""

    text: str
    json: dict | None
    usage: dict


class ChatModel(Protocol):
    """A language model that completes a conversation."""

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """Return the model's reply to ``messages``, parsed against ``json_schema`` if given.

        ``temperature=None`` means the provider's default; adapters omit the parameter.
        """
        ...
