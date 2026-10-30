# callcheck

callcheck is an open-source evaluation harness for voice agents. A simulated caller, driven by a
speech-to-speech model, places real calls to your agent over a streaming audio connection. Each
call is recorded and transcribed with timestamps, a judge scores it against a written rubric with
reasons, and a detector classifies turn-taking incidents as agent bugs or simulator artifacts.

**Status:** Milestone 1 complete: one real call end to end on Nova Sonic.

Quickstart (coming):

```
callcheck run scenarios/intake_short.yaml --target nova-sonic --caller nova-sonic
```

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
