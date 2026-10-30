"""Small formatting helpers shared by the single-run and batch reports."""


def cell(value: object) -> str:
    """Format a value for a table cell, escaping pipes and flattening line breaks."""
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def ms(value: float | None) -> str:
    """Format milliseconds with no decimals, or ``n/a`` when there is no value."""
    return "n/a" if value is None else f"{round(value)} ms"


def usd(value: float | None, partial: bool = False) -> str:
    """Format dollars to the cent; a partial sum reads ``≥ $x.xx``, no value reads ``n/a``."""
    if value is None:
        return "n/a"
    return f"{'≥ ' if partial else ''}${value:.2f}"


UNVERIFIED_NOTE = "Prices are unverified: some rates in the pricing table have not been checked."
PARTIAL_NOTE = "≥ marks a partial sum: some usage had no price and is left out."
