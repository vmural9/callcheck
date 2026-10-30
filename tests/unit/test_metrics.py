import json

from callcheck.core.events import CallEvent, event_to_json
from callcheck.core.metrics import Metrics, compute_metrics, percentile, write_metrics


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
    assert m.ttfa_max_ms == 1200 and m.duration_ms == 9000 and m.cost is None


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


def test_percentile_nearest_rank():
    vals = [100, 200, 300, 400]
    assert (percentile(vals, 50), percentile(vals, 90), percentile(vals, 99)) == (200, 400, 400)
    assert percentile([], 50) is None


def test_metrics_includes_cost(tmp_path):
    usage = {
        "sessionId": "s",
        "details": {"total": {"input": {"speechTokens": 1000}, "output": {"speechTokens": 500}}},
    }
    events = _events() + [CallEvent(7000, "agent", "usage", usage)]
    (tmp_path / "events.jsonl").write_text(
        "\n".join(event_to_json(ev) for ev in events) + "\n", encoding="utf-8"
    )
    (tmp_path / "outcome.json").write_text(json.dumps({"duration_ms": 9000}), encoding="utf-8")
    (tmp_path / "judge.json").write_text(
        json.dumps({"usage": [{"model": "j", "input_tokens": 1000, "output_tokens": 0}]}),
        encoding="utf-8",
    )
    pricing = {
        "models": {
            "nova-sonic": {"per_1k_tokens": {"speech_in": 1.0, "speech_out": 2.0}},
            "bedrock:j": {"per_1k_tokens": {"text_in": 4.0, "text_out": 0.0}},
        }
    }
    m = write_metrics(tmp_path, pricing)
    assert m.cost is not None and m.cost.agent.usd == 2.0 and m.cost.judge.usd == 4.0
    assert m.ttfa_p90_ms == 1200 and m.ttfa_p99_ms == 1200
    assert write_metrics(tmp_path).cost is None
