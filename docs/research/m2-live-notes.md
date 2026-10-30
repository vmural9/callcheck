# Milestone 2 live verification

Run 2026-10-31 in us-west-2: Nova Sonic as the agent under test and as the caller, judged by
Claude on Bedrock, on `scenarios/intake_new_patient.yaml` (20 fields). Every command was run from
the repo root. Run directories are under `runs/` (git-ignored). TTFA is the time from the end of
a caller turn to the agent's first audio. All dollar figures come from the pricing table in the
repo, whose rates are unverified; a leading "≥" marks a sum that leaves out usage with no price.

## What was run

### 1. Preflight

- Passed. The agent heard the caller in 9.7 s, the caller heard the agent in 5.4 s, and the
  first agent audio arrived after 1663 ms.

### 2. Bugs on, four personas

- Command: `uv run callcheck run scenarios/intake_new_patient.yaml --target nova-sonic --caller nova-sonic -n 8 --parallel 4 --personas baseline,terse,self_correcting,out_of_order`
- Planted bugs on: `day_month_swap`, `ignore_corrections`, `strict_order`.
- Run directory: `20261031-014518-intake-new-patient`
- Result: 0 of 8 passed. Wall time 13m 55s. Cost ≥ $1.13 (agent ≥ $0.50, caller ≥ $0.43,
  judge $0.20).
- TTFA over 316 turns: p50 1725 ms, p90 2340 ms, p99 2916 ms, max 3025 ms.
- Incidents: `caller_interrupts` (simulator) 13, `turn_edge` (unclear) 2, `double_talk`
  (simulator) 1. No stalls, and no nudge was needed.

### 3. Bugs off, `baseline`

- Command: `uv run callcheck run scenarios/intake_new_patient.yaml --target nova-sonic --caller nova-sonic -n 4 --parallel 4 --personas baseline --fix-bugs`
- Run directory: `20261031-020121-intake-new-patient`
- Result: 0 of 4 passed. Wall time 6m 37s. Cost ≥ $0.57.
- TTFA over 184 turns: p50 1533 ms, p90 1823 ms, p99 2510 ms, max 2704 ms.
- Incidents: `caller_interrupts` (simulator) 5, `agent_interrupts` (agent) 3, `double_talk`
  (simulator) 2, `double_talk` (agent) 2.

## How each planted bug showed up

- `day_month_swap`: `date_of_birth` matched in 0 of 7 runs with a captured value; the most
  common wrong value was `1988-07-03`. With the bugs off it matched 4 of 4.
- `ignore_corrections`: `field_mismatch` on corrected fields where the agent's last save was
  right but the store kept the first value. For example, `email`: the last save was correct in 5
  runs of the first batch and the stored value was wrong. The judge's `correction_ignored` label
  (5 of 8 runs after re-judging) also appears with the bugs off, so it is not by itself evidence
  of the planted bug.
- `strict_order`: not shown by the first batch. The `field_missing` in `out_of_order-4` was not
  this bug: that call hit the 480 s maximum duration stuck on `emergency_contact_phone`, and it
  is now graded with the failure type `timed_out`. Getting a live proof took three attempts:
  1. The caller never volunteered a field early, because the brief never said when. The brief
     now tells it to give the field together with the previous one.
  2. With the bug written only in the agent's prompt, the live Nova Sonic agent ignored it: it
     said it would get to the field, then asked anyway.
  3. The bug moved into the form store. A save for a field ahead of the first unsaved field is
     silently dropped, and so is every later save of that field, while the prompt tells the agent
     to save early answers and not ask again.

  Live proof: run `20261031-024240-intake-new-patient`, persona `out_of_order`, 2 runs, seed 11
  (`uv run callcheck run scenarios/intake_new_patient.yaml --target nova-sonic --caller nova-sonic -n 2 --parallel 2 --personas out_of_order --seed 11`).
  Both runs ended in `field_missing` on `email`: the agent saved the caller's early email at
  12.8 s together with `first_name`, before `last_name`, and the store dropped it and every
  later email save.

## The clean agent does not reach 100%

With every planted bug off, 0 of 4 runs passed. The cause is on the agent side: Nova Sonic
mishears spoken alphanumerics even when the caller spells them out and corrects them.
Examples from the four runs:

- phone: 0134 heard as 0000, 0137 or 50134; the emergency contact phone the same way
- email: okonkwo heard as oconquo or okonqwo
- insurance member ID: `XQ7-4491-B` captured as `XQ7-4491` or `XQ7-441`
- last name, and preferred name Mari captured as Mary

The judge's `correction_handling` reasons show the agent repeating the wrong value after being
corrected. So `correction_ignored` with the planted bugs off is a real agent failure and not a
judge false positive. That settles the milestone 1 carry-over, where it was left open.

One mismatch is the harness's: the clean-run report shows `address_line` at 2 of 4 because the
agent captured "two hundred eighteen valencia street" against the expected 218. Only
digit-by-digit runs are normalised, so compound number words count as a mismatch.

## Harness fixes the live run drove

Found in the first batch and fixed afterwards:

- "doctor" is normalised to "dr".
- Spoken digit runs are converted to digits.
- An enum phrase inside a longer answer is found, with false-pass guards: the digits must match,
  and a negated enum phrase gives `undetermined` rather than a match.

Re-judging the first batch with these fixes moved `address_line` from 2 of 8 to 8 of 8. The
clean-run batch was re-judged too; its `address_line` row stays at 2 of 4 because those agents
said "two hundred eighteen", and compound number words are not normalised (only digit-by-digit
runs are).

Fixes from the final review:

- A caller who accepts a nudge and then stays silent is an artifact, not an agent failure.
- Running out of time is a failure of type `timed_out`.
- An agent `no_response` or `dead_end` is never erased by an artifact ending.
- A refused nudge is an artifact of type `simulator_error`.
- TTFA skips gaps where the agent was already speaking.
- Retried attempts count in the batch cost.
- Phone captures of 7 to 9 digits count as mismatches.
- More negation words are recognised.

## Calls and spend

- 23 live calls against the ceiling of 30: 4 preflights and 19 graded calls.
- About $2.60 for the calls plus about $0.35 for re-judging, an estimate. The ceiling was 15 USD.
- Nova text tokens are not priced and the other rates are unverified, so the call figure may be
  low.

## Open follow-ups

- Compound number words ("two hundred eighteen") are not normalised.
- The Nova Sonic text-token price is unknown, so every cost is a lower bound.
- All prices in the pricing table are unverified.
