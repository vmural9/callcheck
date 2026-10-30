import json

from callcheck.core.events import CallEvent, event_to_json
from callcheck.core.metrics import Metrics, compute_metrics, write_metrics


def _events() -> list[CallEvent]:
    return [
        CallEvent(900, "agent", "audio_start", {}),
        CallEvent(1000, "agent", "speech_start", {}),
        CallEvent(3000, "agent", "speech_end", {}),
        CallEvent(3500, "caller", "speech_start", {}),
        CallEvent(5000, "caller", "speech_end", {}),
        CallEvent(6200, "agent", "audio_start", {}),
        CallEvent(6300, "agent", "speech_start", {}),
    ]


def test_metrics_from_events():
    m = compute_metrics(_events(), {"reason": "caller_done", "duration_ms": 9000})
    assert m.greeting_latency_ms == 900 and m.ttfa_ms == [1200] and m.ttfa_p50_ms == 1200
    assert (m.caller_turns, m.agent_turns) == (1, 2)
    assert m.ttfa_max_ms == 1200 and m.duration_ms == 9000 and m.cost_usd is None


def test_ttfa_skips_a_turn_the_caller_interrupted():
    evs = [
        CallEvent(1000, "caller", "speech_end", {}),
        CallEvent(1500, "caller", "speech_start", {}),
        CallEvent(2000, "caller", "speech_end", {}),
        CallEvent(2800, "agent", "audio_start", {}),
    ]
    m = compute_metrics(evs, {"duration_ms": 3000})
    assert m.ttfa_ms == [800]


def test_no_agent_audio_leaves_latencies_empty():
    m = compute_metrics([CallEvent(100, "caller", "speech_end", {})], {"duration_ms": 500})
    assert m.greeting_latency_ms is None and m.ttfa_ms == []
    assert m.ttfa_p50_ms is None and m.ttfa_max_ms is None


def test_write_metrics_reads_the_run_directory(tmp_path):
    (tmp_path / "events.jsonl").write_text(
        "".join(event_to_json(ev) + "\n" for ev in _events()), encoding="utf-8"
    )
    (tmp_path / "outcome.json").write_text(json.dumps({"duration_ms": 9000}), encoding="utf-8")
    m = write_metrics(tmp_path)
    assert m == Metrics.model_validate_json((tmp_path / "metrics.json").read_text())
    assert m.ttfa_ms == [1200]
