"""Deterministic, offline semantic experts.

Each expert inspects a column's *distinct* values (never row-by-row over the
whole frame) and returns scored :class:`SemanticProposal` objects. No expert
makes a network call, needs an API key, or imports a heavy dependency.

The module also exposes the low-level value parsers (number words, currency,
units, booleans) and identifier guards, which :mod:`freshdata.semantic.context`
reuses to classify columns.

TODO(extension): future private-LLM candidate-generation experts can implement
the same ``applies`` / ``propose`` interface and be registered in
:mod:`freshdata.semantic.router`. Retrieval-backed corrections from
:class:`~freshdata.CleaningMemory` are implemented separately, as a merge step
in :mod:`freshdata.semantic.apply` (see :mod:`freshdata.semantic.memory`), since
they must be re-checked against the *current* data rather than routed like a
stateless expert.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .formats import EmailExpert, PhoneExpert, ReferenceExpert
from .scoring import make_proposal
from .types import SemanticColumnInfo, SemanticEvidence, SemanticProposal


class SemanticExpert(Protocol):
    """The interface every semantic expert implements (value or veto)."""

    name: str
    issue_type: str

    def applies(self, info: SemanticColumnInfo) -> bool: ...

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]: ...

# --------------------------------------------------------------------------- #
# Value parsers (pure, deterministic)
# --------------------------------------------------------------------------- #

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}
_BIG = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def parse_number_words(text: str) -> int | None:
    """Parse an English number phrase to an int, or ``None`` if not a number.

    Handles ``"zero"``, ``"thirty five"``, ``"one hundred"``,
    ``"two thousand five hundred"``. Any unknown token (``"room"``, ``"oneplus"``)
    makes the whole phrase a non-number, so embedded words are never converted.
    """
    tokens = [t for t in re.split(r"[\s\-]+", text.strip().lower()) if t]
    if not tokens:
        return None
    total = 0
    current = 0
    seen = False
    for tok in tokens:
        if tok in _ONES:
            current += _ONES[tok]
        elif tok in _TENS:
            current += _TENS[tok]
        elif tok == "hundred":
            current = (current or 1) * 100
        elif tok in _BIG:
            total += (current or 1) * _BIG[tok]
            current = 0
        elif tok == "and":
            continue
        else:
            return None
        seen = True
    if not seen:
        return None
    return total + current


_CURRENCY_SYMBOLS = "$€£¥₹"
_CURRENCY_CODES = frozenset(
    {"usd", "eur", "gbp", "inr", "jpy", "rs", "cad", "aud", "chf", "cny"}
)


def parse_currency(text: str) -> float | None:
    """Parse a currency-formatted string to a float, or ``None``.

    Requires an explicit currency marker (symbol or ISO-ish code) so that a bare
    ``"1,200"`` is left to ordinary dtype repair, not treated as money.
    """
    s = text.strip()
    has_symbol = any(c in s for c in _CURRENCY_SYMBOLS)
    codes = {t.lower() for t in re.findall(r"[A-Za-z]+", s)}
    has_code = bool(codes & _CURRENCY_CODES)
    if not (has_symbol or has_code):
        return None
    cleaned = re.sub(r"[A-Za-z$€£¥₹,\s]", "", s)
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


_SYMBOL_TO_CODE = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}


def detect_currency(text: str) -> str | None:
    """Best-effort ISO code for the currency marker in *text*, or ``None``."""
    s = text.strip()
    for symbol, code in _SYMBOL_TO_CODE.items():
        if symbol in s:
            return code
    codes = {t.lower() for t in re.findall(r"[A-Za-z]+", s)}
    matched = codes & _CURRENCY_CODES
    if matched:
        code = sorted(matched)[0]
        return "INR" if code == "rs" else code.upper()
    return None


_UNIT_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z%/°]+)\s*$")


def parse_unit(text: str) -> tuple[float, str] | None:
    """Parse a ``"<number> <unit>"`` string to ``(value, unit)`` or ``None``."""
    m = _UNIT_RE.match(text)
    if not m:
        return None
    return float(m.group(1)), m.group(2).lower()


_TRUE_TOKENS = frozenset({"yes", "y", "true", "t", "1"})
_FALSE_TOKENS = frozenset({"no", "n", "false", "f", "0"})


def parse_boolean(text: str) -> bool | None:
    """Parse a boolean-like token to ``True``/``False``, or ``None``."""
    key = text.strip().lower()
    if key in _TRUE_TOKENS:
        return True
    if key in _FALSE_TOKENS:
        return False
    return None


def is_plain_number(value: object) -> bool:
    """True if *value* is (or stringifies to) an ordinary number, excluding bool."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value))
    if isinstance(value, str):
        s = value.strip().replace(",", "")
        if not s:
            return False
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


_RELATIVE_PHRASES = {"today": 0, "yesterday": -1, "tomorrow": 1}
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_ISO_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_ISO_SLASH_DATE_RE = re.compile(r"^(\d{4})/(\d{2})/(\d{2})$")
#: Dash-form ISO is the canonical date representation: such values are only
#: re-encoded when the column has a genuine repair, never rewritten alone.
_CANONICAL_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")
#: A partial ISO date (year and month, no day).
_PARTIAL_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})$")
_NUMERIC_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")
_MONTH_NAME_DATE_RE = re.compile(
    r"^(?:([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+(\d{4})"
    r"|(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4}))$"
)


def looks_like_date_value(text: str) -> bool:
    """Cheap shape probe: does *text* look like a date phrase or date string?

    Used only to score column *eligibility* (:mod:`freshdata.semantic.context`);
    it never resolves day/month ambiguity, so it is safe to call without a
    ``reference_date`` or ``dayfirst`` hint.
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    if s.casefold() in _RELATIVE_PHRASES:
        return True
    if _ISO_DATE_RE.match(s) or _ISO_SLASH_DATE_RE.match(s) or _NUMERIC_DATE_RE.match(s):
        return True
    return bool(_MONTH_NAME_DATE_RE.match(s))


def _safe_timestamp(year: int, month: int, day: int) -> pd.Timestamp | None:
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class _DateResolution:
    """What :func:`_resolve_date` learned about one raw date-phrase value."""

    value: object  # pd.Timestamp | None
    confidence: float
    risk: str
    detail: str


def _resolve_date(
    raw: str, *, dayfirst: bool | None, reference_date: str | None
) -> _DateResolution | None:
    """Resolve one raw string to a date, or ``None`` if it isn't a date phrase.

    Returns a value even for the phrase-without-``reference_date`` and
    ambiguous-numeric-without-``dayfirst`` cases (so the decision is auditable),
    but always paired with a confidence/risk that keeps the policy gate from
    auto-applying it (see the module-level rationale in the class docstring).
    """
    s = raw.strip()
    key = s.casefold()

    if key in _RELATIVE_PHRASES:
        offset = _RELATIVE_PHRASES[key]
        if not reference_date:
            return _DateResolution(
                None, 0.50, "high",
                f"{raw!r} is a relative date phrase but no reference_date was supplied",
            )
        try:
            ref = pd.Timestamp(reference_date)
        except (ValueError, TypeError):
            return _DateResolution(
                None, 0.50, "high",
                f"reference_date {reference_date!r} could not be parsed",
            )
        # Positional value + explicit unit (not days=<int>): the keyword form
        # routes through a numpy "generic unit" timedelta64 that newer numpy
        # (2.5+) deprecates for bare-int input.
        value = ref + pd.Timedelta(offset, unit="D")
        return _DateResolution(
            value, 0.95, "low", f"{raw!r} resolved against reference_date {reference_date}"
        )

    m = _ISO_DATE_RE.match(s) or _ISO_SLASH_DATE_RE.match(s)
    if m:
        year, month, day = (int(g) for g in m.groups())
        value = _safe_timestamp(year, month, day)
        if value is None:
            return None
        return _DateResolution(value, 0.98, "low", f"{raw!r} is an ISO-ordered date")

    m = _PARTIAL_ISO_RE.match(s)
    if m:
        # A month with no day: any resolution fabricates data, so surface a
        # (never-applied) first-of-month guess for the human reviewer.
        value = _safe_timestamp(int(m.group(1)), int(m.group(2)), 1)
        return _DateResolution(
            value, 0.60, "high",
            f"{raw!r} is a partial date (no day); needs human review",
        )

    m = _MONTH_NAME_DATE_RE.match(s)
    if m:
        groups = m.groups()
        if groups[0] is not None:
            month_name, day_s, year_s = groups[0], groups[1], groups[2]
        else:
            day_s, month_name, year_s = groups[3], groups[4], groups[5]
        month_num = _MONTHS.get(month_name.strip().casefold())
        if month_num is None:
            return None
        value = _safe_timestamp(int(year_s), month_num, int(day_s))
        if value is None:
            return None
        return _DateResolution(value, 0.96, "low", f"{raw!r} is a month-name date")

    m = _NUMERIC_DATE_RE.match(s)
    if m:
        a_s, b_s, year_s = m.groups()
        a, b, year = int(a_s), int(b_s), int(year_s)
        if a > 12 and b <= 12:
            day, month = a, b
        elif b > 12 and a <= 12:
            month, day = a, b
        elif a > 12 and b > 12:
            return None  # neither token can be a month; not a valid date
        elif dayfirst is True:
            day, month = a, b
        elif dayfirst is False:
            month, day = a, b
        else:
            # Ambiguous: both tokens <= 12 and no explicit convention. Still
            # surface a (never-applied) guess for audit purposes.
            value = _safe_timestamp(year, a, b)
            return _DateResolution(
                value, 0.75, "high",
                f"{raw!r} is ambiguous (day/month both <= 12) with no dayfirst hint",
            )
        value = _safe_timestamp(year, month, day)
        if value is None:
            return None
        return _DateResolution(
            value, 0.95, "low", f"{raw!r} resolved unambiguously (day/month order determined)"
        )

    return None


_MIXED_ALNUM = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")
_SPECIAL = re.compile(r"[-_/@.]")


def looks_like_identifier_value(value: object) -> bool:
    """Heuristic veto: does this individual value look like a code/identifier?

    Catches zero-padded numbers (``"007"``), mixed alphanumerics (``"105A"``),
    and punctuated codes/handles (``"A-100"``, ``"SKU-30"``, ``"D@vid"``). Used
    by the spelled-number and category experts to leave codes untouched.
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    if len(s) > 1 and s.isdigit() and s[0] == "0":
        return True
    if _MIXED_ALNUM.search(s) and _HAS_DIGIT.search(s):
        return True
    return bool(_SPECIAL.search(s) and re.search(r"[A-Za-z]", s))


_ID_NAME_TOKENS = re.compile(
    r"(?:^|[_\s])(?:id|uuid|guid|key|code|zip|pin|sku|account|phone)s?$"
    r"|^(?:id|index|pk|pin|zip|sku|code|account|phone|uuid|guid)$",
    re.I,
)


def column_name_is_identifier(name: str) -> bool:
    """True if a column name signals an identifier (``*_id``, ``sku``, ...)."""
    return bool(_ID_NAME_TOKENS.search(str(name)))


# --------------------------------------------------------------------------- #
# Experts
# --------------------------------------------------------------------------- #


def _value_counts(series: pd.Series) -> pd.Series:
    # Native-engine distinct paths (freshdata.semantic.native) precompute the
    # true value->count table over the *full* column and attach it here, so the
    # experts never rescan a materialized frame. Everything else falls through
    # to a direct pandas count, byte-for-byte as before.
    precomputed = series.attrs.get("fd_value_counts")
    if precomputed is not None:
        return precomputed
    try:
        return series.value_counts(dropna=True)
    except TypeError:  # unhashable payloads
        return pd.Series(dtype="int64")


class SpelledNumberExpert:
    """Convert English number words to numbers in numeric-like columns."""

    name = "spelled_number"
    issue_type = "spelled_number"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.numeric_like and not info.free_text

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str) or looks_like_identifier_value(raw):
                continue
            value = parse_number_words(raw)
            if value is None:
                continue
            evidence = (
                SemanticEvidence("pattern", f"{raw!r} is an English number word", 0.0),
                SemanticEvidence(
                    "column_role", "column reads as numeric", 0.02 if info.numeric_like else 0.0
                ),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=value,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.95,
                    evidence=evidence,
                    count=int(count),
                    rationale=f"{raw!r} spells the number {value}",
                    info=info,
                )
            )
        return out


class BooleanSynonymExpert:
    """Normalize boolean-like tokens (yes/no/true/false/1/0) to bools."""

    name = "boolean_synonym"
    issue_type = "boolean_synonym"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.boolean_like and not info.free_text

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if isinstance(raw, bool):
                continue
            text = raw if isinstance(raw, str) else str(raw)
            value = parse_boolean(text)
            if value is None:
                continue
            evidence = (
                SemanticEvidence("pattern", f"{raw!r} is a boolean synonym", 0.0),
                SemanticEvidence("column_role", "column reads as boolean", 0.02),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=value,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.96,
                    evidence=evidence,
                    count=int(count),
                    rationale=f"{raw!r} normalizes to {value}",
                    info=info,
                )
            )
        return out


class CurrencyStringExpert:
    """Strip currency formatting to numeric values in money-like columns."""

    name = "currency_string"
    issue_type = "currency_string"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.money_like and not info.free_text

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str):
                continue
            value = parse_currency(raw)
            if value is None:
                continue
            code = detect_currency(raw)
            if (
                info.allowed_currencies
                and code is not None
                and code not in info.allowed_currencies
            ):
                # A currency outside the declared policy set: parsing the
                # number silently drops the denomination, so surface it as a
                # high-risk suggestion for the human reviewer.
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=value,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=0.60,
                        evidence=(
                            SemanticEvidence(
                                "pattern", f"{raw!r} is a currency string", 0.0
                            ),
                            SemanticEvidence(
                                "context_hint",
                                f"currency {code} is not in the declared set "
                                f"{list(info.allowed_currencies)}; needs human review",
                                0.0,
                            ),
                        ),
                        count=int(count),
                        rationale=(
                            f"{raw!r} is denominated in {code}, outside the "
                            "declared currency policy; needs human review"
                        ),
                        info=info,
                        risk_override="high",
                    )
                )
                continue
            evidence = (
                SemanticEvidence("pattern", f"{raw!r} is a currency string", 0.0),
                SemanticEvidence("column_role", "column reads as monetary", 0.02),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=value,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.96,
                    evidence=evidence,
                    count=int(count),
                    rationale=f"{raw!r} is the monetary amount {value}",
                    info=info,
                )
            )
        return out


class UnitSuffixExpert:
    """Strip a consistent unit suffix to numeric values (``"10 kg"`` -> 10)."""

    name = "unit_suffix"
    issue_type = "unit_suffix"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.unit_like and not info.free_text

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        expected = info.unit or info.dominant_unit
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str):
                continue
            parsed = parse_unit(raw)
            if parsed is None:
                continue
            value, unit = parsed
            if expected is not None and unit != expected:
                # A value denominated differently from the column's unit
                # ("5000 mcg" in a mg column, "10 lb" among kg): stripping or
                # converting silently changes meaning, so surface it as a
                # high-risk suggestion for the human reviewer instead of
                # skipping it silently.
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=value,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=0.60,
                        evidence=(
                            SemanticEvidence(
                                "pattern",
                                f"{raw!r} is a number with unit {unit!r}",
                                0.0,
                            ),
                            SemanticEvidence(
                                "value_share",
                                f"unit {unit!r} conflicts with the column unit "
                                f"{expected!r}; needs human review",
                                0.0,
                            ),
                        ),
                        count=int(count),
                        rationale=(
                            f"{raw!r} uses unit {unit!r} but the column is "
                            f"{expected!r}-denominated; needs human review"
                        ),
                        info=info,
                        risk_override="high",
                    )
                )
                continue
            # Only an explicitly declared column unit makes stripping
            # automatic: an inferred dominant unit — however common — is
            # still a guess about meaning, so those strips are suggested.
            if info.unit:
                support = SemanticEvidence(
                    "context_hint", f"unit {unit!r} is declared for the column", 0.02
                )
                base = 0.95
            else:
                support = SemanticEvidence(
                    "value_share",
                    f"unit {unit!r} is inferred, not declared; strip suggested "
                    "for review",
                    0.0,
                )
                base = 0.90
            evidence = (
                SemanticEvidence("pattern", f"{raw!r} is a number with unit {unit!r}", 0.0),
                support,
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=value,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=base,
                    evidence=evidence,
                    count=int(count),
                    rationale=f"{raw!r} is {value} with unit {unit!r}",
                    info=info,
                )
            )
        return out


_GENDER_SYNONYMS = {
    "m": "male", "male": "male",
    "f": "female", "female": "female",
}


class CategorySynonymExpert:
    """Conservatively normalize categorical variants using column-local evidence.

    v1 handles three safe cases: (1) case/whitespace variants that fold to a
    dominant column-local canonical, (2) a small built-in gender synonym set in
    gender-like columns, and (3) mapping to ``allowed_values`` supplied via
    ``semantic_context``. It never performs global synonym replacement.
    """

    name = "category_synonym"
    issue_type = "category_synonym"

    def applies(self, info: SemanticColumnInfo) -> bool:
        if info.free_text or info.identifier_like or info.boolean_like:
            return False
        if info.allowed_values:
            # Phase 2: explicit reference lists are handled (with fuzzy and
            # ambiguity handling) by ReferenceExpert; avoid double proposals.
            return False
        return info.role == "categorical"

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        counts = _value_counts(series)
        strings = {v: int(c) for v, c in counts.items() if isinstance(v, str)}
        if not strings:
            return []
        out: list[SemanticProposal] = []

        allowed = {str(a).strip().casefold(): a for a in info.allowed_values}
        # Dominant canonical per casefold/strip key (for trivial normalization).
        canon: dict[str, tuple[str, int]] = {}
        for value, count in strings.items():
            key = value.strip().casefold()
            best = canon.get(key)
            if best is None or count > best[1]:
                canon[key] = (value.strip(), count)

        gender_like = info.semantic_type in (None, "category", "gender") and all(
            v.strip().casefold() in _GENDER_SYNONYMS for v in strings
        )

        for value, count in strings.items():
            if looks_like_identifier_value(value):
                continue
            key = value.strip().casefold()
            proposed: object | None = None
            base = 0.0
            why = ""
            ekind = "value_share"
            if key in allowed and allowed[key] != value:
                proposed = allowed[key]
                base, why, ekind = 0.96, "matches an allowed value", "context_hint"
            elif gender_like:
                target = _GENDER_SYNONYMS[key]
                if target != value:
                    proposed = target
                    base, why = 0.95, "is a known gender synonym"
            else:
                target = canon[key][0]
                if target != value and len(canon[key][0]) and "".join(
                    value.split()
                ).casefold() == "".join(target.split()).casefold():
                    proposed = target
                    base, why = 0.93, "case/whitespace variant of a dominant value"
            if proposed is None:
                continue
            evidence = (
                SemanticEvidence(ekind, f"{value!r} {why}", 0.0),
                SemanticEvidence("value_share", "column is low-cardinality categorical", 0.0),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=value,
                    proposed_value=proposed,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=base,
                    evidence=evidence,
                    count=count,
                    rationale=f"{value!r} {why} -> {proposed!r}",
                    info=info,
                )
            )
        return out


class DatePhraseExpert:
    """Normalize obvious date phrases/strings in date-like columns.

    Only ``today``/``yesterday``/``tomorrow`` (with an explicit
    ``reference_date`` in ``semantic_context``), unambiguous ISO/slash/dash and
    month-name dates, and numeric dates disambiguated by an explicit
    ``dayfirst`` hint are ever eligible to auto-apply. Ambiguous numeric dates
    (both day and month <= 12, no ``dayfirst``) and relative phrases without a
    ``reference_date`` are always recorded as high-risk suggestions/skips, never
    silently mutated — see :func:`_resolve_date`.
    """

    name = "date_phrase"
    issue_type = "date_phrase"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.date_like and not info.free_text and not info.identifier_like

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        candidates: list[tuple[str, int, _DateResolution]] = []
        unresolved = False
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str):
                continue
            resolution = _resolve_date(
                raw, dayfirst=info.dayfirst, reference_date=info.reference_date
            )
            # A high-risk resolution (ambiguous order, partial date) is a
            # review item, not a conversion: the column cannot uniformly
            # convert around it, so it counts as unresolved here.
            if resolution is None or resolution.value is None or resolution.risk == "high":
                unresolved = True
            if resolution is None:
                continue
            candidates.append((raw, int(count), resolution))
        # Values already in canonical ISO form are re-encoded in two cases:
        # the column has a genuine phrase/format repair (so it comes out
        # uniform), or *every* string value resolves (the proposals amount to
        # a coherent whole-column datetime conversion).  When unparseable
        # values keep the column as text and nothing needs repairing,
        # rewriting valid ISO dates to timestamps is pure representation
        # churn — leave them untouched.
        genuine = any(
            resolution.value is not None
            and resolution.risk == "low"
            and not _CANONICAL_ISO.fullmatch(raw.strip())
            for raw, _, resolution in candidates
        )
        out: list[SemanticProposal] = []
        for raw, count, resolution in candidates:
            if unresolved and not genuine and _CANONICAL_ISO.fullmatch(raw.strip()):
                continue
            evidence = (
                SemanticEvidence("pattern", resolution.detail, 0.0),
                SemanticEvidence("column_role", f"{info.name!r} reads as a date", 0.0),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=resolution.value,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=resolution.confidence,
                    evidence=evidence,
                    count=count,
                    rationale=resolution.detail,
                    info=info,
                    risk_override=resolution.risk,
                )
            )
        return out


class IdentifierProtectionExpert:
    """A veto expert: flags identifier-like columns and records protective skips.

    It produces an ``identifier_like`` proposal (a no-op on the data) only when
    an identifier-like column also contains values that a value-expert might
    otherwise have tried to transform, so the audit log shows the protection.
    """

    name = "identifier_protection"
    issue_type = "identifier_like"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.identifier_like

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        # Emit a protective record only when a value-expert would otherwise have
        # tried to transform a value here (identifier-likeness suppresses those
        # experts, so we check the raw values directly).
        def transformable(value: object) -> bool:
            if looks_like_identifier_value(value):
                return True
            if isinstance(value, str):
                return (
                    parse_number_words(value) is not None
                    or parse_boolean(value) is not None
                    or parse_currency(value) is not None
                    or parse_unit(value) is not None
                )
            return False

        if not any(transformable(v) for v in series.dropna().head(200)):
            return []
        affected = int(series.notna().sum())
        evidence = (
            SemanticEvidence("column_role", f"{info.name!r} reads as an identifier", -0.5),
        )
        return [
            make_proposal(
                column=info.name,
                raw_value=None,
                proposed_value=None,
                issue_type=self.issue_type,
                expert=self.name,
                base_confidence=0.30,
                evidence=evidence,
                count=affected,
                rationale=(
                    f"{info.name!r} looks like an identifier; semantic repair is "
                    "vetoed to protect codes/keys"
                ),
                info=info,
            )
        ]


#: All value-mutating experts, in priority order.
VALUE_EXPERTS: tuple[SemanticExpert, ...] = (
    SpelledNumberExpert(),
    BooleanSynonymExpert(),
    CurrencyStringExpert(),
    UnitSuffixExpert(),
    CategorySynonymExpert(),
    ReferenceExpert(),
    DatePhraseExpert(),
    EmailExpert(),
    PhoneExpert(),
)

#: The protective veto expert.
IDENTIFIER_EXPERT: SemanticExpert = IdentifierProtectionExpert()
