"""A fake chat model: it replays canned replies, or answers by schema with no script at all.

With a script it is the test double used across the suite. Without one it is the offline judge
behind ``--judge fake``: a deterministic responder that needs no network and no credentials.
"""

import json

from callcheck.core.chat import ChatModel, ChatResult, Message

_FIVE = {"score": 5, "reason": "fake judge", "evidence": []}
_BY_SCHEMA_TITLE: dict[str, dict] = {
    "ExtractedFields": {"fields": {}},
    "FreeTextJudgement": {"verdict": "match", "reason": "fake judge"},
    "QualityReport": {
        name: _FIVE
        for name in ("repetition", "confusion", "correction_handling", "clarity", "closing")
    },
}


class FakeChat(ChatModel):
    """Chat model that answers each call with the next scripted response.

    Built with ``responses=None`` it is scriptless: it answers by the ``title`` of the
    requested schema, extracting nothing, ruling that free text matches, and scoring every
    conversational quality 5.
    """

    model_id = "fake"

    def __init__(
        self, responses: list[dict | str] | None = None, *, usage: dict | None = None
    ) -> None:
        """Script the replies: a ``dict`` is a JSON reply, a ``str`` is text that is not JSON.

        ``None`` (the default) selects the scriptless mode described on the class. ``usage``
        is the token usage every reply reports (empty by default).
        """
        self._usage = dict(usage or {})
        self._scripted = responses is not None
        self._responses = list(responses or [])
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
        if not self._scripted:
            title = (json_schema or {}).get("title", "")
            if title not in _BY_SCHEMA_TITLE:
                raise AssertionError(f"scriptless FakeChat has no answer for schema {title!r}")
            response = _BY_SCHEMA_TITLE[title]
            return ChatResult(text=json.dumps(response), json=response, usage=dict(self._usage))
        if not self._responses:
            raise AssertionError("FakeChat exhausted")
        response = self._responses.pop(0)
        if isinstance(response, dict):
            return ChatResult(text=json.dumps(response), json=response, usage=dict(self._usage))
        return ChatResult(text=response, json=None, usage=dict(self._usage))
