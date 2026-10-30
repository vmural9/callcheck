"""Claude on Amazon Bedrock as a ``ChatModel``, via the Converse API.

Structured output comes from an ``emit`` tool whose input schema is the requested JSON
schema; the model is told to call it. Forced tool choice is rejected by current Claude
models on Bedrock, so ``auto`` is used, with a fallback that parses JSON from the text.
"""

import asyncio
import json
import re
from typing import Any

from callcheck.core.chat import ChatModel, ChatResult, Message

_TOOL_NAME = "emit"
_MAX_TOKENS = 4096
_EMIT_INSTRUCTION = "Answer only by calling the emit tool with your answer as its input."
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _parse_text_json(text: str) -> dict | None:
    """Parse ``text`` as a JSON object, tolerating a surrounding code fence; else ``None``."""
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class BedrockChat(ChatModel):
    """Chat model backed by a Bedrock Converse model such as a Claude inference profile."""

    def __init__(
        self,
        *,
        model_id: str,
        region: str,
        profile: str | None = None,
        client: Any = None,
    ) -> None:
        """Use ``client`` if given; otherwise build a ``bedrock-runtime`` client for ``region``."""
        self.model_id = model_id
        if client is None:
            import boto3

            session = boto3.Session(profile_name=profile)
            client = session.client("bedrock-runtime", region_name=region)
        self._client = client

    async def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """Call Converse off the event loop; ``json_schema`` offers an ``emit`` tool for it."""
        request: dict[str, Any] = {
            "modelId": self.model_id,
            "messages": [
                {"role": m.role, "content": [{"text": m.content}]}
                for m in messages
                if m.role != "system"
            ],
            "inferenceConfig": {"maxTokens": _MAX_TOKENS},
        }
        if temperature is not None:
            request["inferenceConfig"]["temperature"] = temperature
        system = [{"text": m.content} for m in messages if m.role == "system"]
        if json_schema is not None:
            system.append({"text": _EMIT_INSTRUCTION})
        if system:
            request["system"] = system
        if json_schema is not None:
            request["toolConfig"] = {
                "tools": [
                    {
                        "toolSpec": {
                            "name": _TOOL_NAME,
                            "description": "Return the answer",
                            "inputSchema": {"json": json_schema},
                        }
                    }
                ],
                "toolChoice": {"auto": {}},
            }
        response = await asyncio.to_thread(self._client.converse, **request)
        blocks = response["output"]["message"]["content"]
        text = "".join(b["text"] for b in blocks if "text" in b)
        parsed = next((b["toolUse"]["input"] for b in blocks if "toolUse" in b), None)
        if parsed is None and json_schema is not None:
            parsed = _parse_text_json(text)
        usage = response["usage"]
        return ChatResult(
            text=text,
            json=parsed,
            usage={"input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"]},
        )
