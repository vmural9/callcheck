"""A scripted chat model for tests: it replays canned replies and records what it was asked."""

import json

from callcheck.core.chat import ChatModel, ChatResult, Message


class FakeChat(ChatModel):
    """Chat model that answers each call with the next scripted response."""

    def __init__(self, responses: list[dict | str]) -> None:
        """Script the replies: a ``dict`` is a JSON reply, a ``str`` is text that is not JSON."""
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """Record ``messages`` and return the next scripted reply; fail when none is left."""
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeChat exhausted")
        response = self._responses.pop(0)
        if isinstance(response, dict):
            return ChatResult(text=json.dumps(response), json=response, usage={})
        return ChatResult(text=response, json=None, usage={})
