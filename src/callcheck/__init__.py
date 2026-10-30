"""callcheck: an evaluation harness for voice agents.

It places simulated calls against a voice agent, records and transcribes them, and judges the
transcript against a written rubric. The provider-neutral engine lives in ``callcheck.core`` and
the provider integrations in ``callcheck.adapters``.
"""

__version__ = "0.1.0"
