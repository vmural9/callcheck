"""Typed field comparators: decide, without a model, whether a captured value matches.

``compare_field`` dispatches on the field's type to a comparator that normalises both sides
and returns a verdict with a short note saying what was normalised or what differed. Types
that need a model (free text) return ``undetermined`` so the judge can defer them.
"""

import re
from collections.abc import Callable
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Any

from dateutil import parser as date_parser

from callcheck.core.judge.schema import FieldVerdict, Verdict
from callcheck.core.scenario import FieldSpec, FieldType

_MATCH_RATIO = 0.95
_NEAR_MISS_RATIO = 0.75
_BAND_WIDTH = 0.20
_PHONE_TAIL = 10
_NUMBER_TOLERANCE = 1e-9
_DEFAULT_A = datetime(2000, 1, 1)
_DEFAULT_B = datetime(2001, 2, 2)
_TRUE_WORDS = frozenset({"true", "yes", "y", "1"})
_FALSE_WORDS = frozenset({"false", "no", "n", "0"})
_LIST_SPLIT = re.compile(r",|\band\b", re.IGNORECASE)

_HONORIFICS = {"doctor": "dr", "mister": "mr", "missus": "mrs"}
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_NEGATIONS = frozenset({"not", "no", "never", "isn"})
_NEGATION_WINDOW = 3
_DIGIT_RUN = re.compile(rf"\b(?:{'|'.join(_DIGIT_WORDS)})(?: (?:{'|'.join(_DIGIT_WORDS)}))+\b")

Outcome = tuple[Verdict, str]
Comparator = Callable[[FieldSpec, Any], Outcome]


def normalize_text(s: str) -> str:
    """Casefold ``s``, turn every non-alphanumeric run into one space, and strip.

    Then spell honorifics one way (``doctor`` becomes ``dr``) and turn a run of two or more
    spoken digits (``two one eight``) into digits (``218``). A lone number word is untouched.
    """
    text = re.sub(r"[\W_]+", " ", s.casefold()).strip()
    text = " ".join(_HONORIFICS.get(word, word) for word in text.split())
    return _DIGIT_RUN.sub(lambda m: "".join(_DIGIT_WORDS[w] for w in m.group().split()), text)


def normalize_phone(s: str) -> str:
    """Return only the digits of ``s``."""
    return re.sub(r"\D", "", s)


def parse_date(s: Any) -> date | None:
    """Return ``s`` as a date, or ``None`` when it does not parse or lacks a year, month or day.

    A string is parsed against two different default dates; if the results differ, some
    component came from a default rather than from the text, so the date is partial.
    """
    if isinstance(s, datetime):
        return s.date()
    if isinstance(s, date):
        return s
    try:
        first = date_parser.parse(s, dayfirst=False, default=_DEFAULT_A).date()
        second = date_parser.parse(s, dayfirst=False, default=_DEFAULT_B).date()
    except (ValueError, OverflowError, TypeError):
        return None
    return first if first == second else None


def similarity(a: str, b: str) -> float:
    """Return the difflib similarity ratio of the normalised forms of ``a`` and ``b``."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _is_missing(captured: Any) -> bool:
    """Report whether a captured value is absent: ``None``, an empty string or list."""
    if captured is None:
        return True
    if isinstance(captured, str):
        return not captured.strip()
    return isinstance(captured, list) and not captured


def _equal(same: bool, note: str) -> Outcome:
    """Turn an equality test into a match or mismatch outcome with ``note``."""
    return ("match" if same else "mismatch"), note


def _digit_tokens(s: str) -> list[str]:
    """Return the sorted digit runs of ``s``, counting a lone spoken digit word as its digit."""
    tokens = []
    for word in normalize_text(s).split():
        if word in _DIGIT_WORDS and word != "oh":
            tokens.append(_DIGIT_WORDS[word])
        else:
            tokens.extend(re.findall(r"\d+", word))
    return sorted(tokens)


def _compare_similar(spec: FieldSpec, captured: Any) -> Outcome:
    """Grade a name or text field by similarity bands, shifted by ``spec.tolerance``."""
    match_at = spec.tolerance if spec.tolerance is not None else _MATCH_RATIO
    near_at = match_at - _BAND_WIDTH if spec.tolerance is not None else _NEAR_MISS_RATIO
    ratio = similarity(str(spec.expected), str(captured))
    note = f"similarity {ratio:.2f} (match at {match_at:.2f}, near miss at {near_at:.2f})"
    want_digits = _digit_tokens(str(spec.expected))
    got_digits = _digit_tokens(str(captured))
    if want_digits != got_digits:
        note = f"digits differ: {want_digits} vs {got_digits}; {note}"
        return ("near_miss" if ratio >= near_at else "mismatch"), note
    if ratio >= match_at:
        return "match", note
    return ("near_miss" if ratio >= near_at else "mismatch"), note


def _compare_phone(spec: FieldSpec, captured: Any) -> Outcome:
    """Compare phone numbers by last ten digits, or in full when both have 11 or more."""
    want, got = normalize_phone(str(spec.expected)), normalize_phone(str(captured))
    if min(len(want), len(got)) < _PHONE_TAIL:
        return "undetermined", f"fewer than {_PHONE_TAIL} digits: {want} vs {got}"
    if len(want) > _PHONE_TAIL and len(got) > _PHONE_TAIL:
        return _equal(want == got, f"full digits {want} vs {got}")
    return _equal(
        want[-_PHONE_TAIL:] == got[-_PHONE_TAIL:],
        f"last {_PHONE_TAIL} digits {want[-_PHONE_TAIL:]} vs {got[-_PHONE_TAIL:]}",
    )


def _compare_date(spec: FieldSpec, captured: Any) -> Outcome:
    """Compare parsed dates; a captured value that does not parse is undetermined."""
    got = parse_date(captured)
    if got is None:
        return "undetermined", f"partial or unparseable date: {captured!r}"
    want = parse_date(spec.expected)
    return _equal(want == got, f"dates {want} vs {got}")


def _compare_email(spec: FieldSpec, captured: Any) -> Outcome:
    """Compare email addresses case-insensitively."""
    want, got = str(spec.expected).strip().casefold(), str(captured).strip().casefold()
    return _equal(want == got, f"casefolded {want} vs {got}")


def _compare_alnum(spec: FieldSpec, captured: Any) -> Outcome:
    """Compare codes after casefolding and dropping every non-alphanumeric character."""
    want = re.sub(r"[\W_]+", "", str(spec.expected).casefold())
    got = re.sub(r"[\W_]+", "", str(captured).casefold())
    return _equal(want == got, f"alphanumerics {want} vs {got}")


def _compare_number(spec: FieldSpec, captured: Any) -> Outcome:
    """Compare numbers to within 1e-9; a captured value that does not parse is undetermined."""
    try:
        got = float(captured)
    except (TypeError, ValueError):
        return "undetermined", f"could not parse {captured!r} as a number"
    want = float(spec.expected)
    return _equal(abs(want - got) <= _NUMBER_TOLERANCE, f"numbers {want:g} vs {got:g}")


def _compare_boolean(spec: FieldSpec, captured: Any) -> Outcome:
    """Read the captured value as yes or no and compare it with the expected boolean."""
    word = str(captured).strip().casefold()
    if word in _TRUE_WORDS:
        got = True
    elif word in _FALSE_WORDS:
        got = False
    else:
        return "undetermined", f"{captured!r} is not a recognisable yes or no"
    return _equal(got == bool(spec.expected), f"boolean {bool(spec.expected)} vs {got}")


def _phrase_in(phrase: str, text: str) -> bool:
    """Report whether ``phrase`` appears in ``text`` as a whole-word phrase."""
    return bool(phrase) and f" {phrase} " in f" {text} "


def _negated(phrase: str, text: str) -> bool:
    """Report whether a negation word sits within three tokens before ``phrase`` in ``text``."""
    padded = f" {text} "
    head = padded[: padded.index(f" {phrase} ")].split()
    return any(word in _NEGATIONS for word in head[-_NEGATION_WINDOW:])


def _compare_enum(spec: FieldSpec, captured: Any) -> Outcome:
    """Match the captured value against the expected key or any of its synonyms.

    Failing an exact match, accept a captured value that contains exactly one option's
    phrase (a synonym, or the key with underscores as spaces) as whole words. Phrases from
    two different options stay a mismatch.
    """
    key = str(spec.expected)
    synonyms = spec.synonyms or {}
    accepted = {normalize_text(key)}
    accepted.update(normalize_text(s) for s in synonyms.get(key, []))
    got = normalize_text(str(captured))
    if got in accepted:
        return "match", f"{got!r} against {sorted(accepted)}"
    found = []
    for option in dict.fromkeys([key, *synonyms]):
        phrases = {normalize_text(option)}
        phrases.update(normalize_text(s) for s in synonyms.get(option, []))
        hits = [p for p in phrases if _phrase_in(p, got)]
        if any(_negated(p, got) for p in hits):
            return "undetermined", f"{got!r} negates the phrase for {option!r}"
        if hits:
            found.append(option)
    if len(found) == 1:
        verdict = "match" if found[0] == key else "mismatch"
        return verdict, f"{got!r} contains the phrase for {found[0]!r}"
    if len(found) > 1:
        return "mismatch", f"{got!r} contains phrases for both {found[0]!r} and {found[1]!r}"
    return "mismatch", f"{got!r} against {sorted(accepted)}"


def _compare_list(spec: FieldSpec, captured: Any) -> Outcome:
    """Compare lists as sets of normalised items; a string is split on commas and 'and'."""
    items = captured if isinstance(captured, list) else _LIST_SPLIT.split(str(captured))
    want = {normalize_text(str(i)) for i in spec.expected} - {""}
    got = {normalize_text(str(i)) for i in items} - {""}
    if want == got:
        return "match", f"same {len(want)} items"
    return "mismatch", f"missing {sorted(want - got)}, extra {sorted(got - want)}"


def _defer_free_text(spec: FieldSpec, captured: Any) -> Outcome:
    """Leave free text to the judge model."""
    return "undetermined", "needs the judge model"


_COMPARATORS: dict[FieldType, Comparator] = {
    "name": _compare_similar,
    "text": _compare_similar,
    "phone": _compare_phone,
    "date": _compare_date,
    "email": _compare_email,
    "alnum": _compare_alnum,
    "number": _compare_number,
    "boolean": _compare_boolean,
    "enum": _compare_enum,
    "list": _compare_list,
    "free_text": _defer_free_text,
}


def compare_field(spec: FieldSpec, captured: Any | None, method: str) -> FieldVerdict:
    """Compare ``captured`` with ``spec.expected`` using the comparator for the field's type."""
    if _is_missing(captured):
        verdict, note = "missing", "nothing captured"
    else:
        verdict, note = _COMPARATORS[spec.type](spec, captured)
    return FieldVerdict.model_validate(
        {
            "name": spec.name,
            "expected": spec.expected,
            "captured": captured,
            "verdict": verdict,
            "method": method,
            "note": note,
        }
    )
