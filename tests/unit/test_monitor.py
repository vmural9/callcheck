"""Unit tests for the live turn monitor: when a nudge is due and which label it carries."""

from callcheck.core.events import CallEvent
from callcheck.core.monitor import NUDGE_TEXT, TurnMonitor
from callcheck.core.scenario import CallConfig


def ev(t, side, kind, **payload):
    return CallEvent(float(t), side, kind, payload)


def monitor(*events):
    mon = TurnMonitor(CallConfig())
    for event in events:
        mon.observe(event)
    return mon


def test_nudge_text_is_the_agreed_instruction():
    assert NUDGE_TEXT == "The other person has gone quiet. Say: hello, are you still there?"


def test_due_after_nudge_after_s_when_agent_silent():
    mon = monitor(
        ev(0, "caller", "speech_start"),
        ev(1000, "agent", "heard_speech_end"),
        ev(1000, "caller", "speech_end"),
    )
    assert mon.due(8999) is None
    assert mon.due(9001) == "no_response"


def test_agent_transcript_of_the_caller_counts_as_registering_them():
    mon = monitor(
        ev(0, "caller", "speech_start"),
        ev(900, "agent", "transcript", role="caller", text="hi", final=True),
        ev(1000, "caller", "speech_end"),
    )
    assert mon.due(9001) == "no_response"


def test_caller_not_heard_when_the_agent_never_registered_the_caller():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    assert mon.due(9001) == "caller_not_heard"


def test_agent_registering_an_earlier_turn_does_not_count():
    mon = monitor(
        ev(0, "caller", "speech_start"),
        ev(500, "agent", "heard_speech_end"),
        ev(1000, "caller", "speech_end"),
        ev(2000, "agent", "audio_start"),
        ev(3000, "agent", "speech_end"),
        ev(4000, "caller", "speech_start"),
        ev(5000, "caller", "speech_end"),
    )
    assert mon.due(13001) == "caller_not_heard"


def test_resets_on_agent_audio():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    mon.observe(ev(5000, "agent", "audio_start"))
    assert mon.due(20000) is None


def test_resets_when_the_caller_speaks_again():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    mon.observe(ev(6000, "caller", "speech_start"))
    assert mon.due(20000) is None


def test_fires_once_per_episode():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    assert mon.due(9001) == "caller_not_heard"
    assert mon.due(9021) is None
    assert mon.due(30000) is None


def test_a_new_episode_fires_again():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    assert mon.due(9001) is not None
    mon.observe(ev(9100, "caller", "speech_start"))
    mon.observe(ev(9500, "caller", "speech_end"))
    assert mon.due(17000) is None
    assert mon.due(17501) == "caller_not_heard"


def test_a_nudge_event_ends_the_episode():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    mon.observe(ev(9001, "harness", "nudge", label="caller_not_heard", n=1))
    assert mon.due(30000) is None


def test_dead_end_after_stall():
    mon = monitor(ev(0, "agent", "audio_start"), ev(1000, "agent", "turn_end"))
    assert mon.due(6999) is None
    assert mon.due(7001) == "dead_end"


def test_caller_stall_when_the_agents_last_words_were_a_question():
    mon = monitor(
        ev(0, "agent", "audio_start"),
        ev(100, "agent", "transcript", role="agent", text="What is your name?", final=True),
        ev(1000, "agent", "turn_end"),
    )
    assert mon.due(7001) == "caller_stall"


def test_a_statement_is_a_dead_end():
    mon = monitor(
        ev(0, "agent", "audio_start"),
        ev(100, "agent", "transcript", role="agent", text="One moment.", final=True),
        ev(1000, "agent", "turn_end"),
    )
    assert mon.due(7001) == "dead_end"


def test_only_the_latest_agent_turn_decides_the_question():
    mon = monitor(
        ev(0, "agent", "audio_start"),
        ev(100, "agent", "transcript", role="agent", text="Name?", final=True),
        ev(1000, "agent", "turn_end"),
        ev(1500, "caller", "speech_start"),
        ev(2500, "caller", "speech_end"),
        ev(3000, "agent", "audio_start"),
        ev(3100, "agent", "transcript", role="agent", text="Thanks.", final=True),
        ev(4000, "agent", "turn_end"),
    )
    assert mon.due(10001) == "dead_end"


def test_agent_speech_end_after_an_early_turn_end_starts_the_silence():
    mon = monitor(
        ev(0, "agent", "audio_start"),
        ev(0, "agent", "turn_end"),
        ev(60, "agent", "speech_start"),
        ev(3000, "agent", "speech_end"),
    )
    assert mon.due(8999) is None
    assert mon.due(9001) == "dead_end"


def test_hold_keeps_the_silence_from_starting_while_audio_plays():
    mon = monitor(ev(0, "agent", "audio_start"), ev(0, "agent", "turn_end"))
    mon.hold(5000)
    assert mon.due(10999) is None
    assert mon.due(11001) == "dead_end"


def test_caller_speaking_at_turn_end_defers_to_the_callers_own_turn():
    mon = monitor(
        ev(0, "agent", "audio_start"),
        ev(500, "caller", "speech_start"),
        ev(1000, "agent", "turn_end"),
        ev(2000, "caller", "speech_end"),
    )
    assert mon.due(7001) is None
    assert mon.due(10001) == "caller_not_heard"


def test_nothing_is_due_once_the_caller_is_done():
    mon = monitor(ev(0, "caller", "speech_start"), ev(1000, "caller", "speech_end"))
    mon.observe(ev(1100, "caller", "done"))
    assert mon.due(60000) is None


def test_nothing_is_due_before_anyone_has_spoken():
    assert monitor().due(60000) is None


def test_max_nudges_comes_from_the_config():
    assert TurnMonitor(CallConfig(max_nudges=5)).max_nudges == 5
