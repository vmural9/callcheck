"""The reference intake agent: a clinic intake assistant with three planted bugs.

It is the demo target for the harness, defined as a system prompt and two tools so any
speech-to-speech adapter can host it. ``load_reference_agent`` builds it; ``BUGS`` names the
planted bugs; ``FormStore`` is the form its tools write to.
"""

from callcheck.reference_agent.loader import load_reference_agent
from callcheck.reference_agent.tools import BUGS, FormStore

__all__ = ["BUGS", "FormStore", "load_reference_agent"]
