"""The system prompt for the simulated caller.

A caller brief tells the speech-to-speech model playing the caller who it is, how it speaks,
the facts it knows, which of them it will first state wrong and then correct, and how the call
ends. It is rendered from a scenario, a persona and a variation plan, so the same inputs always
produce the same prompt. The preflight brief is a fixed script used to check the audio path.
"""

from callcheck.core.endpoints import CallerBrief
from callcheck.core.scenario import Persona, Scenario, VariationPlan, spoken_value

_SPELL_OUT_NOTE = (
    " (if asked, or if it is read back wrong, spell it out letter by letter, but only once:"
    " you hear a read-back as sound, not letters, so after spelling it once, accept the next"
    " read-back)"
)

_NO_UNSCRIPTED_CORRECTIONS = (
    "Do not correct any other value unless the agent reads it back wrong; then spell it or"
    " repeat it once and accept the next read-back."
)

_PREFLIGHT_PROMPT = (
    "Wait for the other person to speak. Then say exactly: "
    "'Hello, can you hear me? Testing one two three.' "
    "Then say nothing else, whatever happens. Do not call any tool."
)


def _behaviour_sentences(persona: Persona) -> list[str]:
    """Return one sentence for each way the persona departs from a plain, cooperative caller."""
    b = persona.behaviours
    sentences = []
    if b.corrections > 0:
        sentences.append(
            "You sometimes give a value, then correct yourself after the agent reads it back."
        )
    if b.out_of_order > 0:
        sentences.append("You sometimes volunteer information before you are asked for it.")
    if b.off_topic > 0:
        sentences.append("You sometimes ask an unrelated question before answering.")
    if b.interrupts:
        sentences.append("You sometimes start talking before the other person has finished.")
    if b.verbosity == "terse":
        sentences.append("Keep every answer as short as possible.")
    if b.verbosity == "rambling":
        sentences.append("Add a sentence or two of unrelated detail to most answers.")
    if b.clarity == "mumbled":
        sentences.append("Hedge, restart sentences, and say um and uh often.")
    return sentences


def _seconds(value: float) -> str:
    """Render a number of seconds without a trailing ``.0`` when it is whole."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _section(title: str, lines: list[str]) -> str:
    """Render a Markdown section: the header, its lines, and a closing blank line."""
    return "\n".join([f"# {title}", *lines, ""]) + "\n"


def build_brief(scenario: Scenario, persona: Persona, plan: VariationPlan) -> CallerBrief:
    """Render the caller's system prompt from the scenario, persona and variation plan."""
    facts = [
        f"- {spec.name}: {spoken_value(spec)}{_SPELL_OUT_NOTE if spec.spell_out else ''}"
        for spec in scenario.fields
    ]
    corrections = [
        f"When asked for your {c.field}, say '{c.wrong_value}'. Then, after the agent reads "
        f"it back, say that was wrong and give '{c.final_value}'."
        for c in plan.corrections
    ] or ["None."]
    values = {spec.name: spoken_value(spec) for spec in scenario.fields}
    early = [
        f"Before the agent asks for your {name}, bring it up yourself: {values[name]}."
        for name in plan.volunteer_early
    ] or ["None."]
    off_topic = [
        f"After you answer your {q.after_field}, ask: '{q.question}'" for q in plan.off_topic
    ] or ["None."]
    timeout = _seconds(scenario.call.greeting_timeout_s)
    sections = [
        _section(
            "Who you are",
            [scenario.caller_goal.strip(), "You are the caller, never the agent."],
        ),
        _section(
            "How you speak",
            [persona.style.strip(), *_behaviour_sentences(persona), _NO_UNSCRIPTED_CORRECTIONS],
        ),
        _section("Your facts", facts),
        _section(
            "What to do first",
            [
                "Wait for the other person to speak first. "
                f"If you hear nothing for {timeout} seconds, say 'Hello?'"
            ],
        ),
        _section("Corrections to make", corrections),
        _section("Things to mention early", early),
        _section("Questions to ask", off_topic),
        _section(
            "Ending the call",
            [
                "When the agent confirms the form is complete, say goodbye and call the "
                "end_call tool. Never call it before then."
            ],
        ),
    ]
    return CallerBrief(
        persona_id=persona.id,
        system_prompt="".join(sections).rstrip("\n") + "\n",
        voice=persona.voice,
        greeting_timeout_s=scenario.call.greeting_timeout_s,
    )


def preflight_brief() -> CallerBrief:
    """Return the fixed brief that makes the caller say one test phrase and nothing else."""
    return CallerBrief(
        persona_id="preflight", system_prompt=_PREFLIGHT_PROMPT, voice={}, greeting_timeout_s=8.0
    )
