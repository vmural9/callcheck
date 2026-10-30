import json

from callcheck.core.events import CallEvent
from callcheck.core.scenario import CallConfig
from callcheck.core.turns import Incident, decisive_incident, detect_incidents, write_incidents


def ev(t, side, kind, **payload):
    return CallEvent(float(t), side, kind, payload)


def labels(incidents):
    return [i.label for i in incidents]


def detect(events, expected=False):
    return detect_incidents(events, CallConfig(), caller_interrupts_expected=expected)


def caller_turn(start, end):
    return [ev(start, "caller", "speech_start", energy=0.5), ev(end, "caller", "speech_end")]


def test_no_response_when_agent_heard_caller():
    events = [
        *caller_turn(1000, 3000),
        ev(1500, "agent", "heard_speech_start"),
        ev(20000, "harness", "call_end"),
    ]
    found = detect(events)
    assert labels(found) == ["no_response"]
    assert (found[0].attribution, found[0].confidence, found[0].t_ms) == ("agent", "high", 3000)


def test_agent_transcript_of_the_caller_counts_as_registering_them():
    events = [
        *caller_turn(1000, 3000),
        ev(3500, "agent", "transcript", role="caller", text="hi", final=True),
        ev(20000, "harness", "call_end"),
    ]
    assert labels(detect(events)) == ["no_response"]


def test_caller_not_heard_when_agent_never_registered():
    events = [*caller_turn(1000, 3000), ev(20000, "harness", "call_end")]
    found = detect(events)
    assert labels(found) == ["caller_not_heard"]
    assert (found[0].attribution, found[0].confidence) == ("simulator", "high")


def test_agent_answering_within_the_window_is_not_flagged():
    events = [*caller_turn(1000, 3000), ev(6000, "agent", "audio_start")]
    assert detect(events) == []


def test_call_ending_before_the_window_elapses_is_not_flagged():
    events = [*caller_turn(1000, 3000), ev(5000, "harness", "call_end")]
    assert detect(events) == []


def test_caller_resuming_within_the_window_is_not_a_missed_turn():
    events = [*caller_turn(1000, 2000), *caller_turn(3000, 4000), ev(5000, "agent", "audio_start")]
    assert detect(events) == []


def test_dead_end_vs_caller_stall_by_question_mark():
    def silence_after(text):
        return [
            ev(1000, "agent", "transcript", role="agent", text=text, final=True),
            ev(2000, "agent", "turn_end"),
            ev(12000, "harness", "call_end"),
        ]

    stall = detect(silence_after("Anything else?"))
    assert labels(stall) == ["caller_stall"]
    assert (stall[0].attribution, stall[0].confidence) == ("simulator", "high")
    dead = detect(silence_after("Goodbye."))
    assert labels(dead) == ["dead_end"]
    assert (dead[0].attribution, dead[0].confidence) == ("agent", "medium")


def test_partial_transcripts_do_not_decide_the_question():
    events = [
        ev(500, "agent", "transcript", role="agent", text="Ready.", final=True),
        ev(900, "agent", "transcript", role="agent", text="Ready?", final=False),
        ev(2000, "agent", "turn_end"),
        ev(12000, "harness", "call_end"),
    ]
    assert labels(detect(events)) == ["dead_end"]


def test_speech_inside_the_stall_window_prevents_an_episode():
    events = [
        ev(2000, "agent", "turn_end"),
        ev(4000, "caller", "speech_start"),
        ev(5000, "caller", "speech_end"),
        ev(6000, "agent", "audio_start"),
    ]
    assert detect(events) == []


def test_agent_interrupts_vs_turn_edge():
    base = [ev(3000, "agent", "speech_end")]
    interrupted = [*caller_turn(1000, 2000), ev(1500, "agent", "audio_start"), *base]
    found = detect(interrupted)
    assert labels(found) == ["agent_interrupts"]
    assert (found[0].attribution, found[0].confidence) == ("agent", "medium")

    edge = [
        *caller_turn(1000, 1500),
        ev(2200, "caller", "speech_start"),
        ev(2400, "agent", "audio_start"),
        ev(3000, "caller", "speech_end"),
        ev(3500, "agent", "speech_end"),
    ]
    found = detect(edge)
    assert labels(found) == ["turn_edge"]
    assert (found[0].attribution, found[0].confidence) == ("unclear", "low")


def test_agent_audio_and_speech_start_make_one_interruption():
    events = [
        *caller_turn(1000, 2000),
        ev(1500, "agent", "audio_start"),
        ev(1600, "agent", "speech_start"),
        ev(3000, "agent", "speech_end"),
    ]
    assert labels(detect(events)) == ["agent_interrupts"]


def test_caller_interrupts_expected_for_interrupting_persona():
    events = [
        ev(1000, "agent", "speech_start"),
        *caller_turn(2000, 2500),
        ev(4000, "agent", "speech_end"),
    ]
    expected = detect(events, expected=True)
    assert labels(expected) == ["caller_interrupts"]
    assert (expected[0].attribution, expected[0].confidence) == ("expected", "medium")
    unexpected = detect(events, expected=False)
    assert labels(unexpected) == ["caller_interrupts"]
    assert unexpected[0].attribution == "simulator"


def test_double_talk_attributed_to_second_starter():
    caller_second = [
        ev(1000, "agent", "speech_start"),
        *caller_turn(2000, 4500),
        ev(5000, "agent", "speech_end"),
    ]
    found = [i for i in detect(caller_second) if i.label == "double_talk"]
    assert [(i.attribution, i.confidence, i.t_ms) for i in found] == [("simulator", "medium", 2000)]

    agent_second = [
        ev(1000, "caller", "speech_start"),
        ev(2000, "agent", "audio_start"),
        ev(4500, "caller", "speech_end"),
        ev(5000, "agent", "speech_end"),
    ]
    found = [i for i in detect(agent_second) if i.label == "double_talk"]
    assert [i.attribution for i in found] == ["agent"]


def test_short_overlap_is_not_double_talk():
    events = [
        ev(1000, "agent", "speech_start"),
        *caller_turn(2000, 3000),
        ev(5000, "agent", "speech_end"),
    ]
    assert "double_talk" not in labels(detect(events))


def test_slow_response_over_threshold():
    slow = [ev(1000, "caller", "speech_end"), ev(5500, "agent", "audio_start")]
    found = detect(slow)
    assert labels(found) == ["slow_response"]
    assert (found[0].attribution, found[0].confidence, found[0].t_ms) == ("agent", "high", 1000)
    fine = [ev(1000, "caller", "speech_end"), ev(4900, "agent", "audio_start")]
    assert detect(fine) == []


def test_transport_error_ends_call():
    fatal = [ev(5000, "harness", "error", message="socket closed", recoverable=False)]
    found = detect(fatal)
    assert labels(found) == ["transport_error"]
    assert (found[0].attribution, found[0].confidence, found[0].ends_call) == (
        "simulator",
        "high",
        True,
    )
    assert "socket closed" in found[0].reason
    assert detect([ev(5000, "harness", "error", message="blip", recoverable=True)]) == []


def test_unsorted_events_are_sorted():
    events = [
        ev(2000, "agent", "turn_end"),
        ev(12000, "harness", "call_end"),
        ev(1000, "agent", "transcript", role="agent", text="Anything else?", final=True),
    ]
    snapshot = list(events)
    assert labels(detect(events)) == ["caller_stall"]
    assert events == snapshot


def test_incidents_come_back_sorted_by_time():
    events = [
        ev(9000, "harness", "error", message="x", recoverable=False),
        ev(2000, "agent", "turn_end"),
        ev(12000, "harness", "call_end"),
    ]
    found = detect(events)
    assert [i.t_ms for i in found] == sorted(i.t_ms for i in found)
    assert labels(found) == ["dead_end", "transport_error"]


def test_agent_never_speaks():
    found = detect(caller_turn(1000, 3000))
    assert len(found) == 1 and found[0].label in {"no_response", "caller_not_heard"}


def test_no_events_no_incidents():
    assert detect([]) == []


def test_watchdog_marks_decisive():
    events = [
        *caller_turn(1000, 3000),
        ev(1500, "agent", "heard_speech_start"),
        ev(19000, "harness", "watchdog", reason="max_nudges"),
        ev(19500, "harness", "call_end"),
    ]
    found = detect(events)
    assert labels(found) == ["no_response"]
    decisive = decisive_incident(found)
    assert decisive is not None and decisive.label == "no_response" and decisive.ends_call


def test_watchdog_marks_the_most_recent_nudge_incident():
    events = [
        ev(1000, "agent", "turn_end"),
        *caller_turn(8000, 9000),
        ev(9500, "agent", "audio_start"),
        ev(10000, "agent", "turn_end"),
        ev(30000, "harness", "watchdog", reason="nudge_refused"),
    ]
    found = detect(events)
    assert [(i.label, i.ends_call) for i in found] == [("dead_end", False), ("dead_end", True)]
    assert decisive_incident(found).t_ms == 10000


def test_watchdog_without_prior_incident_emits_one_from_its_label():
    events = [ev(9000, "harness", "watchdog", reason="max_nudges", label="caller_stall")]
    found = detect(events)
    assert [(i.label, i.attribution, i.ends_call) for i in found] == [
        ("caller_stall", "simulator", True)
    ]


def test_other_watchdog_reasons_do_not_end_the_call():
    events = [
        ev(2000, "agent", "turn_end"),
        ev(12000, "harness", "watchdog", reason="max_seconds"),
    ]
    assert not any(i.ends_call for i in detect(events))


def test_decisive_incident_is_none_without_an_ending():
    quiet = Incident(t_ms=1, label="turn_edge", attribution="unclear", confidence="low", reason="r")
    assert decisive_incident([quiet]) is None
    assert decisive_incident([]) is None


def test_write_incidents_saves_a_json_list(tmp_path):
    incident = Incident(
        t_ms=3000,
        label="no_response",
        attribution="agent",
        confidence="high",
        reason="r",
        ends_call=True,
    )
    write_incidents(tmp_path, [incident])
    rows = json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8"))
    assert rows == [incident.model_dump()]
    write_incidents(tmp_path, [])
    assert json.loads((tmp_path / "incidents.json").read_text(encoding="utf-8")) == []
