# callcheck

callcheck is an open-source evaluation harness for voice agents. A simulated caller, driven by a
speech-to-speech model, places a real call to an agent over a streaming audio connection. The
call is recorded and transcribed with timestamps, and a judge scores it against the scenario's
written expectations and shows its reasons.

**Status:** milestone 2 is done: turn-taking detection with nudges, persona matrices run in
parallel, per-call cost, and an aggregate report over a batch. Today the agent under test is the
bundled reference intake agent (with planted bugs) running on Nova Sonic, and the caller is Nova
Sonic too. Planned next: your own agents as targets.

## Quickstart

Needs Python 3.12, [uv](https://docs.astral.sh/uv/), and an AWS account with Amazon Bedrock model
access for Nova Sonic and a Claude model (the judge) in `us-west-2`.

```
git clone <this repo> && cd callcheck
uv sync --extra bedrock
cp .env.example .env        # then set AWS_PROFILE (and AWS_REGION if not us-west-2)
uv run callcheck preflight --target nova-sonic --caller nova-sonic
uv run callcheck run scenarios/intake_short.yaml --target nova-sonic --caller nova-sonic \
    --personas self_correcting
```

A batch is a persona matrix: `-n` calls cycling through the listed personas, `--parallel` calls at a
time, and one aggregate report for the lot. This is the command behind the sample report below.
Add `--fix-bugs` to switch the reference agent's planted bugs off and measure the clean agent.

```
uv run callcheck run scenarios/intake_new_patient.yaml --target nova-sonic --caller nova-sonic \
    -n 8 --parallel 4 --personas baseline,terse,self_correcting,out_of_order
uv run callcheck run scenarios/intake_new_patient.yaml --target nova-sonic --caller nova-sonic \
    -n 4 --parallel 4 --personas baseline --fix-bugs
```

`preflight` checks that the two sides can hear each other. `run` places the call, then writes a
run folder under `runs/` holding `call.wav`, `transcript.jsonl`, `judge.json` and `report.md`, and
prints the verdict. It exits 1 if a call crashed or judging failed, and 0 otherwise, so an agent
that fails its scenario still exits 0. Live calls cost a few cents each. A batch
also writes `report.md` and `report.json` at the top of its run folder.

## Personas

Seven callers ship in `scenarios/personas/`:

- `baseline`: cooperative and clear; the control.
- `terse`: answers in a word or a few words and adds nothing unasked.
- `rambling`: wraps every answer in a sentence or two of unrelated story.
- `self_correcting`: states some facts wrong first, then corrects them.
- `out_of_order`: volunteers later details before the agent asks for them.
- `off_topic`: asks unrelated practical questions in the middle of the form.
- `mumbler`: nervous and quiet, with hedges, restarts and "um", but the facts stay correct.

## Sample report

This is real output, not a mock-up: an excerpt of the `report.md` from a live batch of 8 calls
(Nova Sonic calling Nova Sonic, the reference intake agent with all three planted bugs on, four
personas, judged by Claude on Bedrock). Rows and sections are omitted and the values are unedited. The sample was rendered before
the `timed_out` failure type existed: one run that ran out of time appears under `field_missing`
in the original file and is omitted here. The links point into that batch's run directory, which is not committed.

The sample shows two of the three planted bugs: a date of birth that never matches
(`day_month_swap`) and corrections that never reach the form (`ignore_corrections`: in 5 runs
the agent's last email save was correct but the store kept the first, wrong one; the wrong email
shown in the table below is a mishearing, not that case). `strict_order` lives in
the form store, where a save for a field ahead of the first unsaved one is silently dropped. It
is shown by a separate run of the `out_of_order` persona, not by this sample:

```
uv run callcheck run scenarios/intake_new_patient.yaml --target nova-sonic --caller nova-sonic \
    -n 2 --parallel 2 --personas out_of_order --seed 11
```

See [docs/research/m2-live-notes.md](docs/research/m2-live-notes.md).

````markdown
# callcheck batch report

## Summary

- Scenario: intake-new-patient
- Target: nova-sonic
- Caller: nova-sonic
- Judge: bedrock
- Bugs: day_month_swap, ignore_corrections, strict_order
- Runs: 8
- Pass: 0, fail: 8, undetermined: 0, artifact: 0, errored: 0
- Pass rate: 0/8 (0%) (pass / (pass + fail); undetermined and artifact are not counted)
- Total cost: ≥ $1.13
- Wall time: 13m 55s

## Failures by type

| type | runs | example |
| --- | --- | --- |
| field_mismatch | 8 | [baseline-1](baseline-1/transcript.md) |
| confusion | 6 | [baseline-1](baseline-1/transcript.md) |
| field_near_miss | 5 | [baseline-1](baseline-1/transcript.md) |
| repetition | 5 | [baseline-1](baseline-1/transcript.md) |
| correction_ignored | 5 | [baseline-1](baseline-1/transcript.md) |
| bad_closing | 3 | [out_of_order-8](out_of_order-8/transcript.md) |

## Field accuracy

| field | match rate | most common wrong value |
| --- | --- | --- |
| date_of_birth | 0/7 (0%) | 1988-07-03 |
| email | 0/8 (0%) | maria.oconquo@example.com |
| address_line | 8/8 (100%) | - |

## Latency

- Turns measured: 316
- TTFA p50: 1725 ms
- TTFA p90: 2340 ms
- TTFA p99: 2916 ms
- TTFA max: 3025 ms (slowest turn: [terse-6](terse-6/transcript.md))

## Cost

| party | total | mean per run |
| --- | --- | --- |
| agent | ≥ $0.50 | ≥ $0.06 |
| caller | ≥ $0.43 | ≥ $0.05 |
| judge | $0.20 | $0.02 |
| total | ≥ $1.13 | - |

≥ marks a partial sum: some usage had no price and is left out.
Prices are unverified: some rates in the pricing table have not been checked.
````
