"""Unit tests for the call event record and its JSON serialisation."""

import json

from callcheck.core.events import CallEvent, endpoint_event, event_to_json


def test_event_json_drops_pcm_and_keeps_rest():
    ev = CallEvent(12.5, "agent", "audio_out", {"pcm": b"\x00\x01", "n": 1})
    d = json.loads(event_to_json(ev))
    assert d == {"t_ms": 12.5, "side": "agent", "kind": "audio_out", "payload": {"n": 1}}


def test_endpoint_event_is_unstamped():
    assert endpoint_event("caller", "done").t_ms == 0.0
