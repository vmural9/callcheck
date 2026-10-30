from callcheck.core.events import CallEvent
from callcheck.core.recorder import Recorder, build_transcript, render_transcript_md


def test_recorder_writes_all_files_and_dedupes_dir(tmp_path):
    r1 = Recorder(tmp_path / "run")
    r2 = Recorder(tmp_path / "run")
    assert r2.run_dir.name == "run-2"
    r1.record(CallEvent(0, "harness", "call_start", {}))
    r1.write_audio(b"\x00" * 960, b"\x00" * 960)
    r1.close({"reason": "caller_done", "duration_ms": 20})
    assert {p.name for p in r1.run_dir.iterdir()} >= {
        "events.jsonl",
        "call.wav",
        "outcome.json",
        "transcript.jsonl",
        "transcript.md",
    }


def test_build_transcript_attaches_heard_as():
    evs = [
        CallEvent(
            1000,
            "agent",
            "transcript",
            {"role": "agent", "text": "What is your name?", "final": True},
        ),
        CallEvent(
            1400,
            "caller",
            "transcript",
            {"role": "agent", "text": "what is your name", "final": True},
        ),
        CallEvent(3000, "caller", "transcript", {"role": "caller", "text": "Maria", "final": True}),
        CallEvent(3600, "agent", "transcript", {"role": "caller", "text": "Mariah", "final": True}),
    ]
    t = build_transcript(evs)
    assert [(u["side"], u["text"], u["heard_as"]) for u in t] == [
        ("agent", "What is your name?", "what is your name"),
        ("caller", "Maria", "Mariah"),
    ]


def test_build_transcript_falls_back_to_heard_when_no_self_transcript():
    evs = [
        CallEvent(
            1400, "caller", "transcript", {"role": "agent", "text": "hello there", "final": True}
        )
    ]
    assert build_transcript(evs) == [
        {
            "t_ms": 1400,
            "side": "agent",
            "text": "hello there",
            "final": True,
            "heard_as": None,
            "source": "heard",
        }
    ]


def test_render_transcript_md_inlines_watchdog():
    md = render_transcript_md(
        [
            {
                "t_ms": 500,
                "side": "agent",
                "text": "Hi",
                "final": True,
                "heard_as": None,
                "source": "self",
            }
        ],
        [CallEvent(12000, "harness", "watchdog", {"reason": "stalled"})],
    )
    assert "[00:00.500] AGENT: Hi" in md and "[00:12.000] -- watchdog: stalled" in md
