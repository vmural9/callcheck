"""The in-memory intake form the reference agent writes to, and the bugs planted in it.

``FormStore`` backs the agent's two tools, ``save_field`` and ``submit_form``, and reports
what it captured so the harness can compare it with what the caller said. The reference agent
exists to be evaluated, so the store ships with three deliberately planted bugs, each switched
on by name in ``BUGS``. They are not accidents; they are what the harness is built to catch
(the README sample shows one of them, ``day_month_swap``, caught):

* ``ignore_corrections``: a second ``save_field`` for a field already saved is dropped, though
  the tool still reports success, so the caller's correction never reaches the form.
* ``strict_order``: handled in the prompt, not here; it tells the agent to refuse a later field
  the caller volunteers early, so that answer is not saved when it is given.
* ``day_month_swap``: a date whose day is 12 or less is stored with day and month exchanged,
  so ``March 7 1988`` becomes ``1988-07-03``. ``save_field`` never echoes the stored value, so
  the agent cannot read the swapped date back and the corruption stays silent.
"""

from datetime import date

from dateutil import parser as dateparser

BUGS = ("ignore_corrections", "strict_order", "day_month_swap")


class FormStore:
    """Holds the intake form for one call and executes the agent's two tools against it."""

    def __init__(self, field_names: list[str], kinds: dict[str, str], bugs: frozenset[str]) -> None:
        """Accept saves for ``field_names``, parse values by ``kinds``, and plant ``bugs``."""
        self._field_names = frozenset(field_names)
        self._kinds = kinds
        self._bugs = bugs
        self._fields: dict[str, str] = {}
        self._submitted = False

    async def execute(self, name: str, arguments: dict) -> dict:
        """Run the tool called ``name`` with ``arguments`` and return its JSON-able result."""
        if name == "save_field":
            return self._save_field(arguments["name"], arguments["value"])
        if name == "submit_form":
            self._submitted = True
            return {"ok": True, "submitted": True}
        return {"ok": False, "error": "unknown tool"}

    def snapshot(self) -> dict | None:
        """Return a copy of the form, or ``None`` if nothing has been saved yet."""
        if not self._fields:
            return None
        return {"fields": dict(self._fields), "submitted": self._submitted}

    def _save_field(self, name: str, value: str) -> dict:
        """Store one field, applying the planted bugs; the result never echoes the stored value."""
        if name not in self._field_names:
            return {"ok": False, "error": "unknown field"}
        if "ignore_corrections" in self._bugs and name in self._fields:
            return {"ok": True}
        self._fields[name] = self._normalise(name, value)
        return {"ok": True}

    def _normalise(self, name: str, value: str) -> str:
        """Return ``value`` as an ISO date for date fields, or unchanged if it is not one."""
        if self._kinds.get(name) != "date":
            return value
        try:
            parsed = dateparser.parse(value, dayfirst=False).date()
        except (ValueError, OverflowError):
            return value
        if "day_month_swap" in self._bugs and parsed.day <= 12:
            parsed = date(parsed.year, parsed.day, parsed.month)
        return parsed.isoformat()
