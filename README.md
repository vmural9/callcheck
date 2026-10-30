# callcheck

callcheck is an open-source evaluation harness for voice agents. A simulated caller, driven by a
speech-to-speech model, places a real call to an agent over a streaming audio connection. The
call is recorded and transcribed with timestamps, and a judge scores it against the scenario's
written expectations and shows its reasons.

**Status:** milestone 1 is done: one real call, end to end, on Nova Sonic. Today the agent under
test is the bundled reference intake agent (with planted bugs) running on Nova Sonic, and the
caller is Nova Sonic too. Planned next: a detector that tells agent bugs from simulator artifacts
in turn-taking (milestone 2), then your own agents as targets and a parallel runner.

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

`preflight` checks that the two sides can hear each other. `run` places the call, then writes a
run folder under `runs/` holding `call.wav`, `transcript.jsonl`, `judge.json` and `report.md`, and
prints the verdict. It exits 1 if a call crashed or judging failed, and 0 otherwise, so an agent
that fails its scenario still exits 0. Live calls cost a few cents each.

## Sample report

From the first live run that caught a planted bug: Nova Sonic calling Nova Sonic, the reference
intake agent with its bugs switched on, the `self_correcting` caller, seed 1, judged by Claude on
Bedrock. The agent read the birth date back correctly ("March seventh, nineteen eighty eight"),
but its form stored it with day and month swapped (the planted `day_month_swap` bug).

- Verdict: **fail**
- Failure types: field_mismatch

| field | expected | captured | verdict | method | note |
| --- | --- | --- | --- | --- | --- |
| first_name | Maria | maria | match | result | similarity 1.00 (match at 0.95, near miss at 0.75) |
| last_name | Okonkwo-Reyes | okonkwo reyes | match | result | similarity 1.00 (match at 0.95, near miss at 0.75) |
| date_of_birth | 1988-03-07 | 1988-07-03 | mismatch | result | dates 1988-03-07 vs 1988-07-03 |
| phone | +1 415 555 0134 | 415-555-0134 | match | result | last 10 digits 4155550134 vs 4155550134 |
| email | maria.okonkwo@example.com | maria.okonkwo@example.com | match | result | casefolded maria.okonkwo@example.com vs maria.okonkwo@example.com |
| reason_for_visit | knee pain after running, started two weeks ago | knee pain after running that started about two weeks ago | match | result | Both describe knee pain after running that began about two weeks ago; "about" is a negligible hedge and no essential fact differs. |
