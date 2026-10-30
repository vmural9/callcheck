from pathlib import Path

import pytest

from callcheck.core.cost import cost_of, load_pricing, nova_usage
from callcheck.core.events import CallEvent

PRICING = {
    "models": {
        "nova-sonic": {
            "unverified": True,
            "per_1k_tokens": {
                "speech_in": 1.0,
                "speech_out": 2.0,
                "text_in": None,
                "text_out": None,
            },
        },
        "bedrock:judge-a": {
            "unverified": False,
            "per_1k_tokens": {"text_in": 1.0, "text_out": 3.0},
        },
    }
}


def _usage(t, session, si, ti, so, to, side="agent"):
    total = {
        "input": {"speechTokens": si, "textTokens": ti},
        "output": {"speechTokens": so, "textTokens": to},
    }
    return CallEvent(t, side, "usage", {"sessionId": session, "details": {"total": total}})


def test_nova_usage_takes_last_total_per_session():
    events = [
        _usage(1, "s1", 100, 10, 50, 5),
        _usage(2, "s1", 300, 30, 150, 15),
        _usage(3, "s2", 1000, 100, 500, 50),
        _usage(4, "s1", 400, 40, 200, 20, side="caller"),
    ]
    assert nova_usage(events, "agent") == {
        "speech_in": 1300,
        "speech_out": 650,
        "text_in": 130,
        "text_out": 65,
    }
    assert nova_usage(events, "caller")["speech_in"] == 400
    assert nova_usage([], "agent") is None


def test_cost_of_known_and_unknown_models():
    events = [_usage(1, "s", 2000, 0, 1000, 0)]
    judge = [{"model": "nope", "input_tokens": 1000, "output_tokens": 1000}]
    cost = cost_of(events, judge, PRICING, agent_model="nova-sonic", caller_model="nova-sonic")
    assert cost.agent.usd == pytest.approx(2.0 + 2.0)
    assert cost.judge.usd is None and cost.judge.note.startswith("n/a")
    assert cost.caller.usd is None and cost.caller.note.startswith("n/a")  # no caller usage
    assert cost.total_usd == pytest.approx(4.0)


def test_total_is_sum_of_known_lines():
    events = [_usage(1, "s", 1000, 0, 0, 0), _usage(2, "s", 1000, 0, 0, 0, side="caller")]
    judge = [{"model": "judge-a", "input_tokens": 2000, "output_tokens": 1000}]
    cost = cost_of(events, judge, PRICING, agent_model="nova-sonic", caller_model="nova-sonic")
    assert cost.judge.usd == pytest.approx(5.0) and not cost.judge.unverified
    assert cost.total_usd == pytest.approx(1.0 + 1.0 + 5.0)
    assert cost.partial is False


def test_unverified_flag_propagates():
    cost = cost_of(
        [_usage(1, "s", 1000, 0, 0, 0)], [], PRICING, agent_model="nova-sonic", caller_model="x"
    )
    assert cost.agent.unverified is True


def test_null_unit_price_makes_partial():
    cost = cost_of(
        [_usage(1, "s", 1000, 500, 0, 0)], [], PRICING, agent_model="nova-sonic", caller_model="x"
    )
    assert cost.agent.usd == pytest.approx(1.0)
    assert cost.agent.partial is True and "text_in" in cost.agent.note
    assert cost.partial is True
    # zero usage in a null-priced unit is not partial
    cost = cost_of(
        [_usage(1, "s", 1000, 0, 0, 0)], [], PRICING, agent_model="nova-sonic", caller_model="x"
    )
    assert cost.agent.partial is False


def test_shipped_pricing_yaml_loads():
    path = Path(__file__).resolve().parents[2] / "pricing.yaml"
    models = load_pricing(path)["models"]
    assert "nova-sonic" in models and "bedrock:us.anthropic.claude-sonnet-5-5" in models
    assert str(models["nova-sonic"]["checked"]) == "2026-10-31"
