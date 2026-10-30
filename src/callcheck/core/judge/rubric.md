# Conversation quality rubric

You score the agent's side of the call on five dimensions, each from 0 to 5. A 5 is a call a
careful human receptionist would be happy with; a 0 is a call that failed on that dimension.
Scores of 1, 2 and 4 sit between the anchors. Judge only what the transcript shows, and cite the
transcript lines that justify each score.

Edit the definitions and anchors below to change how strict the judge is. The names of the five
dimensions are fixed; the wording is yours.

## repetition

Did the agent ask for something it already had, or repeat itself?

- 0: Asked for the same fact three or more times, or looped on one question.
- 3: Asked for one fact twice without a reason, or repeated a sentence once.
- 5: Never asked for anything it already had; every repeat was a deliberate read-back.

## confusion

Did the agent misunderstand, loop, or lose the thread?

- 0: Lost the thread entirely; answers had no relation to what the caller said.
- 3: Misunderstood once and recovered, or needed the caller to rephrase.
- 5: Followed every turn without misunderstanding or backtracking.

## correction_handling

When the caller corrected a value, did the agent acknowledge it and use the corrected value?

If the caller made no correction, score 5.

- 0: Ignored the correction and kept the wrong value to the end.
- 3: Acknowledged the correction but wavered, or needed it repeated before using it.
- 5: Acknowledged the correction at once and used the corrected value from then on.

## clarity

Were the agent's questions clear and asked one at a time?

- 0: Questions were unintelligible, or several were piled into one turn throughout.
- 3: Mostly clear, but at least one question was vague or bundled with another.
- 5: Every question was clear, short and asked on its own.

## closing

Did the agent confirm the form and end the call cleanly?

- 0: Never closed: the call ended abruptly, in silence, or mid-question.
- 3: Closed, but without confirming what was captured, or with an awkward ending.
- 5: Confirmed the form was complete, thanked the caller, and said goodbye.
