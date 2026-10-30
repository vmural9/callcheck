# Milestone 1 live verification

Run 2026-10-30 in us-west-2: Nova Sonic (`amazon.nova-2-5-sonic`) as the agent under test
(voice `matthew`) and as the caller (voice `tiffany`), judged by Claude Sonnet on Bedrock.
Every command was run from the repo root after `set -a && source .env && set +a`. Run
directories are under `runs/` (git-ignored). Greeting latency is the time from call start to
the agent's first audio; TTFA is the time from the end of a caller turn to the agent's first
audio.

The `run` commands pass `--skip-preflight`: `run` places its own preflight call by default,
and the preflight had already passed (attempt 1), so skipping it kept each run to one call.

## Attempts

### 1. Preflight

- Command: `uv run callcheck preflight --target nova-sonic --caller nova-sonic`
- Outcome: exit 0. `heard by agent:   yes   "hello can you hear me"   (6.9 s)`,
  `heard by caller:  yes   "hello what is your first name"   (4.4 s)`; the call itself ended on
  `max_duration` at 20 s, as a preflight does once the caller has said its line.
- Verdict: passed. Failure types: none.
- Greeting latency: 1620 ms. TTFA p50: not computed for a preflight.
- Run directory: `20261030-210130-preflight`
- Surprising: the agent split the test phrase into two FINAL transcripts ("hello can you hear
  me", then "testing one two three"), so the preflight line shows only the first half. It still
  passes, because two of the test words are in it (risk d, as expected).

### 2. Bugs on, `self_correcting`, seed 1

- Command: `uv run callcheck run scenarios/intake_short.yaml --target nova-sonic --caller nova-sonic --persona self_correcting --seed 1 --skip-preflight`
- Outcome: `max_duration` (240 s).
- Verdict: `fail`, 2 of 6 fields matched, 4 missing, quality 0.8/5.
- Failure types: field_missing, repetition, confusion, correction_ignored, unclear_question,
  bad_closing.
- Greeting latency: 1540 ms. TTFA p50: 572 ms (27 agent turns).
- Run directory: `20261030-210158-intake-short/self_correcting-1`
- Surprising: the call never got past the surname. Nova's recogniser, on the caller side, hears
  the agent's correct read-back "Okonkwo-Reyes" as "oconquo reyes" every time. The brief told the
  caller to spell the name "if it is read back wrong", so it re-spelled after every read-back, about 20
  times. That was a simulator artifact, but the judge blamed the agent. Fixed in the caller brief
  (aaf79c0). `ignore_corrections` did drop the second and third `last_name` saves, but the first
  save was already right, so it could not show.

### 3. Bugs on, `self_correcting`, seed 1, after the first brief fix

- Command: as attempt 2.
- Outcome: `caller_done` at 178 s. The caller hung up with `end_call` after the agent's "The form
  is complete. Goodbye!". The last FINAL transcript is in `transcript.md` (risk b did not lose it).
- Verdict: `fail`, 5 of 6 fields matched, 1 mismatched, quality 3.8/5.
- Failure types: field_mismatch.
- The planted bug was caught: `date_of_birth | 1988-03-07 | 1988-07-03 | mismatch | dates
  1988-03-07 vs 1988-07-03`. The caller said "August eighth ... no wait, March seventh, nineteen
  eighty eight". The agent saved "March 7th, 1988" and read back "March seventh, nineteen eighty
  eight", so the corruption was silent in the conversation. This is `day_month_swap`.
- Greeting latency: 1545 ms. TTFA p50: 1027 ms (14 agent turns).
- Run directory: `20261030-210634-intake-short/self_correcting-1`
- Surprising:
  - The caller corrected the phone number unprompted ("zero one four three, no wait, zero one
    three four"; risk c). The plan only scheduled corrections for date_of_birth and email. The
    persona's style text gives exactly that phone example.
  - Both corrections were made "in the same breath", so the agent saved only the final value
    each time. That means `ignore_corrections` cannot fire on a scheduled correction: it needs
    a second `save_field` for the field.
  - Both off-topic questions ("parking", "walk ins") were asked and ignored.
  - `transcript.md` shows empty `-- interrupted: ` lines where the two sides overlapped.

### 4. Bugs off, `baseline`, seed 1

- Command: `uv run callcheck run scenarios/intake_short.yaml --target nova-sonic --caller nova-sonic --persona baseline --seed 1 --fix-bugs --skip-preflight`
- Outcome: `max_duration` (240 s).
- Verdict: `fail`, 2 of 6 fields matched, 4 missing, quality 0.6/5.
- Failure types: field_missing, repetition, confusion, correction_ignored, unclear_question,
  bad_closing.
- Greeting latency: 1582 ms. TTFA p50: 635 ms (27 agent turns).
- Run directory: `20261030-210956-intake-short/baseline-1`
- Surprising: the same surname loop came back, even with the first fix ("accept a read-back that
  sounds right"). The caller only sees its own transcript, "oconquo reyes is that correct", so it
  cannot judge sound. Fixed with a count it can follow, "spell it out ... but only once ... after
  spelling it once, accept the next read-back" (b11fa02).

### 5. Bugs off, `baseline`, seed 1, after the second brief fix

- Command: as attempt 4.
- Outcome: `caller_done` at 126 s.
- Verdict: `fail`, 4 of 6 fields matched, 1 near miss, 1 mismatched, quality 3.0/5.
- Failure types: field_mismatch, field_near_miss, confusion, correction_ignored.
- These are agent mishearings, recorded as findings and not fixed in this milestone:
  - `last_name`: captured `oconquo reyes`, a near miss at similarity 0.77.
  - `email`: captured `maria.okonqo@s.example.com`, a mismatch. The agent's own transcript of
    the caller reads "my email address is maria dot oconquo at s. example dot com".
  - `date_of_birth` matched (1988-03-07), as expected with `day_month_swap` off.
- Greeting latency: 1542 ms. TTFA p50: 1128 ms (14 agent turns).
- Run directory: `20261030-211439-intake-short/baseline-1`
- Surprising:
  - The caller accepted the surname read-back "Oconquo Reyes" at once, without spelling it.
    Under the new rule it also accepted the mangled email after spelling it once. A correct
    acceptance and a wrong one now look alike in the transcript; the form comparison is what
    catches the error.
  - The caller said "No" to a correct phone read-back, then repeated the same number.
  - The judge's `correction_ignored` comes from the quality score, not from a planted bug: they
    were all off.

## Calls placed and estimated spend

- This task: 5 live calls of the 10 allowed (1 preflight + 4 runs). There were 4 graded runs, so
  the judge made about 6 Claude calls (one quality call per run, plus one free-text call for
  each run that captured `reason_for_visit`).
- Nova Sonic usage, summed from the `usage` events of both sessions in each call:
  - speech: 20 632 input tokens, 16 013 output tokens
  - text: 12 762 input tokens, 7 977 output tokens
- Estimated spend: about 0.29 USD for Nova Sonic. This assumes the Nova Sonic v1 list prices
  (speech 0.0034 / 0.0136 USD per 1k tokens in / out, text 0.00006 / 0.00024); the
  `amazon.nova-2-5-sonic` prices were not checked. Judge tokens were not measured.
- Against the milestone ceiling of 30 calls and 15 USD: this task added 5 calls and well under
  1 USD.
