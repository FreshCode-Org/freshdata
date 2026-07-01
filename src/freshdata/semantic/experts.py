"""Deterministic, offline semantic experts.

Each expert inspects a column's *distinct* values (never row-by-row over the
whole frame) and returns scored :class:`SemanticProposal` objects. No expert
makes a network call, needs an API key, or imports a heavy dependency.

The module also exposes the low-level value parsers (number words, currency,
units, booleans) and identifier guards, which :mod:`freshdata.semantic.context`
reuses to classify columns.

TODO(extension): future experts (private-LLM candidate generation,
retrieval-backed corrections) can implement the same ``applies`` / ``propose``
interface and be registered in :mod:`freshdata.semantic.router`.
"""

from __future__ import annotations

import math
import re
from typing import Protocol

import pandas as pd

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
                continue
            evidence = (
                SemanticEvidence("pattern", f"{raw!r} is a number with unit {unit!r}", 0.0),
                SemanticEvidence(
                    "context_hint" if info.unit else "value_share",
                    f"unit {unit!r} is consistent for the column",
                    0.02,
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
        return info.role == "categorical" or bool(info.allowed_values)

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
)

#: The protective veto expert.
IDENTIFIER_EXPERT: SemanticExpert = IdentifierProtectionExpert()
