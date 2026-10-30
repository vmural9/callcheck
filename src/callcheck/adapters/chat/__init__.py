"""Registry of chat models (the judge's language model) selectable by name."""

from typing import Any

from callcheck.adapters.registry import build
from callcheck.core.chat import ChatModel

CHAT: dict[str, str] = {
    "fake": "callcheck.adapters.chat.fake:FakeChat",
    "bedrock": "callcheck.adapters.chat.bedrock:BedrockChat",
}


def make_chat(name: str, **kwargs: Any) -> ChatModel:
    """Instantiate the registered chat model ``name`` with ``kwargs``; ``KeyError`` if unknown."""
    return build(CHAT, "chat", name, **kwargs)
