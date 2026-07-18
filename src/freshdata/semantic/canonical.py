"""Deterministic canonical-form experts: NFC, mojibake, and format alignment.

These experts repair values whose *content* is already correct but whose
representation deviates from the column's canonical form: decomposed Unicode
(NFD ``"José"``), double-encoded mojibake (``"CafÃ©"``), separator drift
against a dominant column template (``"555 0101"`` among ``"555-0101"``),
numeric-formatting stragglers (``"95%"``, ``"1.234,56"``), and ISO-8601
oddities (``"24:00"``, an instant in a date-only column).  Two review-only
experts route values that *look* wrong but have no safe rewrite — a composite
of two valid category values (``"US/CA"``), and a rare near-variant of a
dominant value (``"2025/26"`` among ``"2025-2026"``) — to a human instead.

Like every expert in :mod:`freshdata.semantic.experts` they operate on a
column's distinct values, are pure and deterministic, and only ever propose —
the policy gate decides.  Declared-sensitive columns are never touched here:
their anomalies are routed to review by the cross-field consistency checks
(:mod:`freshdata.semantic.consistency`) without disclosing any value.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from .formats import _edit_distance, _ref_key
from .scoring import make_proposal
from .types import SemanticColumnInfo, SemanticEvidence, SemanticProposal


def _value_counts(series: pd.Series) -> pd.Series:
    # Mirrors freshdata.semantic.experts._value_counts (kept separate to avoid
    # an import cycle): native distinct paths attach the true value->count
    # table in ``series.attrs`` so experts never rescan a materialized frame.
    precomputed = series.attrs.get("fd_value_counts")
    if precomputed is not None:
        return precomputed
    try:
        return series.value_counts(dropna=True)
    except TypeError:  # unhashable payloads
        return pd.Series(dtype="int64")


# --------------------------------------------------------------------------- #
# Unicode canonical composition (NFC)
# --------------------------------------------------------------------------- #


class UnicodeNormalizationExpert:
    """Normalize decomposed Unicode text to canonical NFC form.

    NFC composition rewrites only the byte representation: the rendered text is
    canonically equivalent, so the repair is payload-preserving and reversible
    (the raw form survives in the action metadata).  Already-composed values
    are untouched.
    """

    name = "unicode_nfc"
    issue_type = "format_alignment"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return not info.sensitive and not info.numeric_like and not info.boolean_like

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str):
                continue
            value = unicodedata.normalize("NFC", raw)
            if value == raw:
                continue
            evidence = (
                SemanticEvidence(
                    "pattern", "value is not in Unicode NFC canonical form", 0.0
                ),
                SemanticEvidence(
                    "pattern",
                    "NFC composition preserves canonically equivalent content",
                    0.0,
                ),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=value,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.97,
                    evidence=evidence,
                    count=int(count),
                    rationale=(
                        "decomposed Unicode sequence composes to the canonical "
                        "NFC form with identical rendered content"
                    ),
                    info=info,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Mojibake (UTF-8 mis-decoded as Latin-1)
# --------------------------------------------------------------------------- #

_MOJIBAKE_MARKER = re.compile(r"[ÃÂ]|â€")
_HTML_ENTITY = re.compile(r"&(?:#\d+|#x[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")


class MojibakeExpert:
    """Repair UTF-8 text that was mis-decoded as Latin-1 (``"CafÃ©"``).

    The repair is the exact inverse of the corruption (re-encode as Latin-1,
    decode as UTF-8), so it is bijective and reversible.  A value that mixes
    mojibake with HTML entities is compound corruption — the repair order is
    ambiguous — so it is held for human review instead of rewritten.
    """

    name = "mojibake"
    issue_type = "encoding_repair"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return not info.sensitive and not info.numeric_like and not info.boolean_like

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str) or not _MOJIBAKE_MARKER.search(raw):
                continue
            try:
                repaired = raw.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
            if repaired == raw or _MOJIBAKE_MARKER.search(repaired):
                continue
            if _HTML_ENTITY.search(raw):
                out.append(
                    make_proposal(
                        column=info.name,
                        raw_value=raw,
                        proposed_value=None,
                        issue_type=self.issue_type,
                        expert=self.name,
                        base_confidence=0.75,
                        evidence=(
                            SemanticEvidence(
                                "pattern",
                                "value mixes a double-encoded (mojibake) sequence "
                                "with HTML entities",
                                -0.2,
                            ),
                        ),
                        count=int(count),
                        rationale=(
                            "value stacks two corruptions (Latin-1 mojibake and "
                            "HTML entities); the repair order is ambiguous, so it "
                            "is held for human review"
                        ),
                        info=info,
                        risk_override="high",
                    )
                )
                continue
            evidence = (
                SemanticEvidence(
                    "pattern",
                    "UTF-8 bytes were mis-decoded as Latin-1; the inverse "
                    "transcoding restores the original characters",
                    0.0,
                ),
                SemanticEvidence(
                    "pattern", "the repair round-trips (bijective, reversible)", 0.0
                ),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=repaired,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.96,
                    evidence=evidence,
                    count=int(count),
                    rationale=(
                        "mis-decoded UTF-8 (mojibake) transcodes exactly back to "
                        "the original text"
                    ),
                    info=info,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Dominant-shape separator alignment (payload-preserving)
# --------------------------------------------------------------------------- #

_SAFE_SEPARATORS = frozenset(" -()/_:+.")


def _shape(value: str) -> str | None:
    """Character-class template: digits -> ``9``, letters -> ``A``, safe
    separators kept literally.  ``None`` for values with exotic characters."""
    out: list[str] = []
    for ch in value:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A")
        elif ch in _SAFE_SEPARATORS:
            out.append(ch)
        else:
            return None
    return "".join(out)


def _payload(value: str) -> str:
    """The alphanumeric content of *value* in NFC form (separators removed)."""
    return "".join(
        ch for ch in unicodedata.normalize("NFC", value) if ch.isalnum()
    )


def _render(payload: str, shape: str) -> str | None:
    """Render *payload* into *shape*, or ``None`` if slots do not fit exactly."""
    out: list[str] = []
    index = 0
    for ch in shape:
        if ch in ("9", "A"):
            if index >= len(payload):
                return None
            item = payload[index]
            if ch == "9" and not item.isdigit():
                return None
            if ch == "A" and not item.isalpha():
                return None
            out.append(item)
            index += 1
        else:
            out.append(ch)
    if index != len(payload):
        return None
    return "".join(out)


class ShapeAlignmentExpert:
    """Align separator drift to a column's dominant value template.

    When one character-class template (``999-9999``) dominates a column and a
    value's alphanumeric payload fits that template exactly, the value is
    re-rendered into the dominant template (``"555 0101"`` -> ``"555-0101"``).
    The payload is untouched by construction, which is what lets the policy
    gate admit these repairs even in identifier-like columns.
    """

    name = "shape_alignment"
    issue_type = "format_alignment"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return (
            not info.sensitive
            and not info.free_text
            and not info.numeric_like
            and not info.money_like
            and not info.boolean_like
            and not info.unit_like
            and not info.date_like
            and (info.nunique or 0) >= 2
        )

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        counts = _value_counts(series)
        shapes: dict[str, int] = {}
        total = 0
        for raw, count in counts.items():
            if not isinstance(raw, str):
                return []  # mixed-type column: no meaningful template
            total += int(count)
            shape = _shape(raw)
            if shape is not None:
                shapes[shape] = shapes.get(shape, 0) + int(count)
        if not shapes or total < 4:
            return []
        # Deterministic dominant selection: highest count, ties lexicographic.
        dominant, dominant_count = sorted(
            shapes.items(), key=lambda item: (-item[1], item[0])
        )[0]
        if dominant_count / total < 0.75:
            return []
        out: list[SemanticProposal] = []
        for raw, count in counts.items():
            if _shape(raw) == dominant:
                continue
            rendered = _render(_payload(raw), dominant)
            if rendered is None or rendered == raw:
                continue
            evidence = (
                SemanticEvidence(
                    "value_share",
                    f"one template covers {dominant_count}/{total} of the "
                    "column's values",
                    0.0,
                ),
                SemanticEvidence(
                    "pattern",
                    "the value's alphanumeric payload fits the dominant "
                    "template exactly; only separators change",
                    0.0,
                ),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=rendered,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.96,
                    evidence=evidence,
                    count=int(count),
                    rationale=(
                        "separators realigned to the column's dominant format; "
                        "the alphanumeric payload is unchanged"
                    ),
                    info=info,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Numeric formatting stragglers (percent suffix, European decimal)
# --------------------------------------------------------------------------- #

_PERCENT_VALUE = re.compile(r"^\s*[+-]?\d+(?:\.\d+)?\s*%\s*$")
_PERCENT_NAME = re.compile(r"percent|pct|rate|ratio", re.I)
_EURO_GROUPED = re.compile(r"^\s*[+-]?\d{1,3}(?:\.\d{3})+,\d{1,2}\s*$")


class NumericFormatExpert:
    """Parse unambiguous formatted-number stragglers in numeric columns.

    Two exact cases only: a ``"95%"`` value in a percent-denominated column
    (the suffix is redundant with the column's meaning), and a European-grouped
    number (``"1.234,56"`` — dot thousands *and* comma decimal present, so the
    reading is unambiguous).  Anything else is left to dtype repair.
    """

    name = "numeric_format"
    issue_type = "numeric_format"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return (
            not info.sensitive
            and not info.free_text
            and (info.numeric_like or info.money_like)
        )

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        out: list[SemanticProposal] = []
        percent_column = bool(_PERCENT_NAME.search(info.name))
        for raw, count in _value_counts(series).items():
            if not isinstance(raw, str):
                continue
            if percent_column and _PERCENT_VALUE.match(raw):
                value = float(raw.strip().rstrip("%").strip())
                rationale = (
                    "the '%' suffix is redundant in a percent-denominated "
                    f"column; parses exactly to {value}"
                )
                detail = "column name declares percent denomination"
            elif _EURO_GROUPED.match(raw):
                value = float(raw.strip().replace(".", "").replace(",", "."))
                rationale = (
                    "European-formatted number (dot thousands, comma decimal) "
                    f"parses unambiguously to {value}"
                )
                detail = "both separators present, so the locale is unambiguous"
            else:
                continue
            evidence = (
                SemanticEvidence("pattern", detail, 0.0),
                SemanticEvidence("column_role", "column reads as numeric", 0.0),
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
                    rationale=rationale,
                    info=info,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# ISO-8601 canonicalization: 24:00 midnight, instants in date-only columns
# --------------------------------------------------------------------------- #

_TIME_VALUE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?$")
_MIDNIGHT_24 = re.compile(r"^24:00(?::00)?$")


class TimeCanonicalExpert:
    """Canonicalize ISO-8601 end-of-day ``24:00`` to ``00:00`` in time columns."""

    name = "time_canonical"
    issue_type = "format_alignment"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return not info.sensitive and not info.free_text and not info.numeric_like

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        counts = _value_counts(series)
        total = 0
        time_shaped = 0
        for raw, count in counts.items():
            if not isinstance(raw, str):
                return []
            total += int(count)
            if _TIME_VALUE.match(raw) or _MIDNIGHT_24.match(raw):
                time_shaped += int(count)
        if total < 4 or time_shaped / total < 0.6:
            return []
        out: list[SemanticProposal] = []
        for raw, count in counts.items():
            if not _MIDNIGHT_24.match(raw):
                continue
            value = "00:00:00" if raw.count(":") == 2 else "00:00"
            evidence = (
                SemanticEvidence(
                    "pattern",
                    "ISO 8601 defines 24:00 as midnight, canonically 00:00",
                    0.0,
                ),
                SemanticEvidence(
                    "value_share", "column values are clock times", 0.0
                ),
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
                    rationale=(
                        "ISO 8601 end-of-day 24:00 canonicalizes to midnight "
                        "00:00; the instant is unchanged"
                    ),
                    info=info,
                )
            )
        return out


_ISO_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_INSTANT = re.compile(
    r"^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?$"
)


class IsoInstantDateExpert:
    """Truncate an ISO instant to its written date in a date-only column.

    When a column's values are overwhelmingly plain ISO dates, a lone
    timestamp carries spurious precision at the column's granularity (the FHIR
    ``date`` normalization).  The written calendar date is preserved exactly;
    the time-of-day the column cannot represent is dropped.
    """

    name = "iso_instant_date"
    issue_type = "format_alignment"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return info.date_like and not info.sensitive and not info.free_text

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        counts = _value_counts(series)
        total = 0
        date_only = 0
        for raw, count in counts.items():
            if not isinstance(raw, str):
                continue
            total += int(count)
            if _ISO_DATE_ONLY.match(raw):
                date_only += int(count)
        if total < 4 or date_only / total < 0.6:
            return []
        out: list[SemanticProposal] = []
        for raw, count in counts.items():
            if not isinstance(raw, str):
                continue
            match = _ISO_INSTANT.match(raw)
            if match is None:
                continue
            value = match.group(1)
            try:
                pd.Timestamp(value)
            except (ValueError, TypeError):
                continue
            evidence = (
                SemanticEvidence(
                    "value_share",
                    f"{date_only}/{total} of the column's values are plain "
                    "ISO dates",
                    0.0,
                ),
                SemanticEvidence(
                    "pattern",
                    "the instant's written calendar date is kept exactly; only "
                    "sub-day precision the column cannot represent is dropped",
                    0.0,
                ),
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
                    rationale=(
                        "timestamp truncated to its written calendar date to "
                        "match the column's date-only granularity"
                    ),
                    info=info,
                )
            )
        return out


# --------------------------------------------------------------------------- #
# Review-only detectors: composite categories, dominant-value variants
# --------------------------------------------------------------------------- #

_COMPOSITE_SEPARATORS = ("/", "|")


class CompositeCategoryExpert:
    """Route a composite of two valid category values to human review.

    ``"US/CA"`` in a country column whose values include both ``"US"`` and
    ``"CA"`` is a contradiction or an unmade choice — no rewrite is safe, so
    the value is surfaced for review and never changed.
    """

    name = "composite_category"
    issue_type = "unsafe_ambiguous"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return (
            not info.sensitive
            and not info.free_text
            and not info.identifier_like
            and not info.numeric_like
            and not info.boolean_like
            and not info.date_like
            and 2 <= (info.nunique or 0) <= 24
        )

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        counts = _value_counts(series)
        values = {value for value in counts.index if isinstance(value, str)}
        out: list[SemanticProposal] = []
        for raw, count in counts.items():
            if not isinstance(raw, str):
                continue
            for separator in _COMPOSITE_SEPARATORS:
                if separator not in raw:
                    continue
                parts = [part.strip() for part in raw.split(separator)]
                if (
                    len(parts) == 2
                    and parts[0] != parts[1]
                    and all(part and part != raw and part in values for part in parts)
                ):
                    evidence = (
                        SemanticEvidence(
                            "pattern",
                            "the value is two distinct valid values of this "
                            f"column joined by {separator!r}",
                            -0.2,
                        ),
                        SemanticEvidence(
                            "value_share",
                            "both component values occur on their own in the "
                            "column",
                            0.0,
                        ),
                    )
                    out.append(
                        make_proposal(
                            column=info.name,
                            raw_value=raw,
                            proposed_value=None,
                            issue_type=self.issue_type,
                            expert=self.name,
                            base_confidence=0.75,
                            evidence=evidence,
                            count=int(count),
                            rationale=(
                                "value combines two distinct valid values of "
                                "this column; the intended single value is "
                                "ambiguous and needs human review"
                            ),
                            info=info,
                            risk_override="high",
                        )
                    )
                    break
        return out


class DominantVariantExpert:
    """Route a rare, confusably-close variant of a dominant value to review.

    In a column dominated by one value (``"2025-2026"`` share >= 75%), a rare
    value whose normalized form is within a small edit distance of the
    dominant one (``"2025/26"``) looks like a variant or entry error — but no
    rewrite is provably content-preserving, so it goes to a human.
    """

    name = "dominant_variant"
    issue_type = "unsafe_ambiguous"

    def applies(self, info: SemanticColumnInfo) -> bool:
        return (
            not info.sensitive
            and not info.free_text
            and not info.identifier_like
            and not info.numeric_like
            and not info.boolean_like
            and (info.nunique or 0) >= 2
        )

    def propose(self, series: pd.Series, info: SemanticColumnInfo) -> list[SemanticProposal]:
        counts = _value_counts(series)
        strings = [
            (value, int(count))
            for value, count in counts.items()
            if isinstance(value, str)
        ]
        total = sum(count for _, count in strings)
        if total < 8:
            return []
        dominant, dominant_count = sorted(
            strings, key=lambda item: (-item[1], item[0])
        )[0]
        if dominant_count / total < 0.75:
            return []
        dominant_key = _ref_key(dominant)
        rare_cap = max(3, int(0.1 * total))
        out: list[SemanticProposal] = []
        for raw, count in strings:
            if raw == dominant or count > rare_cap:
                continue
            key = _ref_key(raw)
            if key == dominant_key:
                continue  # pure separator/case drift is format-alignable
            if _edit_distance(key, dominant_key, cap=4) > 3:
                continue
            evidence = (
                SemanticEvidence(
                    "value_share",
                    f"one value covers {dominant_count}/{total} of the column",
                    0.0,
                ),
                SemanticEvidence(
                    "pattern",
                    "the rare value is confusably close to the dominant value "
                    "but not provably equivalent",
                    -0.2,
                ),
            )
            out.append(
                make_proposal(
                    column=info.name,
                    raw_value=raw,
                    proposed_value=None,
                    issue_type=self.issue_type,
                    expert=self.name,
                    base_confidence=0.75,
                    evidence=evidence,
                    count=count,
                    rationale=(
                        "rare value is confusably close to the column's "
                        "dominant value; it may be a variant or an entry error "
                        "and needs human review"
                    ),
                    info=info,
                    risk_override="high",
                )
            )
        return out


__all__ = [
    "CompositeCategoryExpert",
    "DominantVariantExpert",
    "IsoInstantDateExpert",
    "MojibakeExpert",
    "NumericFormatExpert",
    "ShapeAlignmentExpert",
    "TimeCanonicalExpert",
    "UnicodeNormalizationExpert",
]
