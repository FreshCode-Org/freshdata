"""Shared, generic custom-check functions for the newer domain packs.

These are config-driven checks with the standard custom-check signature
``(df, mapping, rule) -> list[row_labels]`` that the healthcare/education/agriculture/
media packs register by name in ``register_extensions`` and reference from their
``rules.yaml`` via ``params.func``. They never mutate the frame.

Date parsing suppresses pandas' format-inference ``UserWarning`` (freshdata treats its
own-namespace warnings as errors). All checks evaluate only rows where the target value
is present — a wholly-absent column is reported by the schema layer, not here.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd

from .base import ColumnMapping, Rule

__all__ = [
    "PHI_MASK",
    "check_at_least_one",
    "check_both_present",
    "check_fhir_date",
    "check_ge_date",
    "check_iso_date",
    "check_iso_datetime",
    "check_nonneg",
    "check_nonneg_number",
    "check_not_future",
    "check_numeric",
    "check_partial_date",
    "check_positive",
    "check_positive_integer",
    "check_requires_field",
    "check_requires_when_value",
    "parse_iso_date",
    "redact_phi_actions",
    "to_datetime_safe",
]

#: Placeholder shown in the audit trail in place of a protected PHI value.
PHI_MASK = "[PHI]"

_FULL_ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PARTIAL_ISO_RE = re.compile(r"\d{4}(-\d{2})?")  # YYYY or YYYY-MM
_MONTHS = range(1, 13)
# pandas >= 2.0 infers a single datetime format for a whole Series, coercing rows in a
# different (but still valid) ISO shape to NaT — e.g. a date-only value mixed with
# datetimes. format="ISO8601" parses each ISO value on its own merits; it only exists
# on pandas >= 2.0, so older pandas (which parses element-wise anyway) falls back.
_PANDAS_GE_2 = tuple(int(part) for part in pd.__version__.split(".")[:2]) >= (2, 0)


def _sort_key(value: Any) -> tuple[int, Any]:
    """Order row labels of mixed type deterministically (numbers first, then strings)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))


def to_datetime_safe(values: Any, **kwargs: Any) -> pd.Series:
    """``pd.to_datetime(errors="coerce")`` that parses mixed ISO date/datetime shapes.

    Warnings are silenced (freshdata treats own-namespace warnings as errors). On
    pandas >= 2.0 each ISO value is parsed independently via ``format="ISO8601"`` so a
    date-only value mixed with datetimes is not spuriously coerced to ``NaT``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        already_dt = isinstance(values, pd.Series) and pd.api.types.is_datetime64_any_dtype(values)
        if _PANDAS_GE_2 and "format" not in kwargs and not already_dt:
            try:
                return pd.to_datetime(values, errors="coerce", format="ISO8601", **kwargs)
            except (ValueError, TypeError):
                pass
        return pd.to_datetime(values, errors="coerce", **kwargs)


def parse_iso_date(value: Any) -> pd.Timestamp | None:
    """Strict ``YYYY-MM-DD`` parse; ``None`` for any other shape or an impossible date."""
    if isinstance(value, pd.Timestamp):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _FULL_ISO_RE.fullmatch(text):
        return None
    try:
        ts = pd.to_datetime(text, format="%Y-%m-%d")
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts


def _valid_partial_date(text: str) -> bool:
    """True for a FHIR partial date: ``YYYY`` or ``YYYY-MM`` (month 01-12)."""
    parts = text.split("-")
    if not (parts[0].isdigit() and len(parts[0]) == 4):
        return False
    if len(parts) == 1:
        return True
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 2:
        return int(parts[1]) in _MONTHS
    return False


def _present(df: pd.DataFrame, mapping: ColumnMapping, field: str) -> pd.Series | None:
    col = mapping.actual(field)
    return df[col].notna() if col is not None else None


def check_iso_date(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present values across ``rule.fields`` that are not a valid full ISO date."""
    rows: set[Any] = set()
    for field in rule.fields:
        col = mapping.actual(field)
        if col is None:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        present = series.notna()
        well_formed = series.astype("string").str.fullmatch(_FULL_ISO_RE.pattern).fillna(False)
        bad = present & ~well_formed
        for idx in df.index[present & well_formed]:
            if parse_iso_date(series.at[idx]) is None:
                bad.at[idx] = True
        rows.update(df.index[bad].tolist())
    return sorted(rows, key=_sort_key)


def check_iso_datetime(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present values across ``rule.fields`` not parseable as an ISO date/datetime."""
    rows: set[Any] = set()
    for field in rule.fields:
        col = mapping.actual(field)
        if col is None:
            continue
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            continue
        present = series.notna()
        bad = present & to_datetime_safe(series).isna()
        rows.update(df.index[bad].tolist())
    return sorted(rows, key=_sort_key)


def check_fhir_date(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag malformed dates; accept full *and* FHIR partial dates (``YYYY``/``YYYY-MM``)."""
    col = mapping.actual(rule.fields[0])
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return []
    bad: list[Any] = []
    for idx in df.index[series.notna()]:
        text = str(series.at[idx]).strip()
        if _FULL_ISO_RE.fullmatch(text):
            if parse_iso_date(text) is None:
                bad.append(idx)
        elif not (_PARTIAL_ISO_RE.fullmatch(text) and _valid_partial_date(text)):
            bad.append(idx)
    return bad


def check_partial_date(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag (info) present values that are FHIR *partial* dates (reduced precision)."""
    col = mapping.actual(rule.fields[0])
    series = df[col]
    if pd.api.types.is_datetime64_any_dtype(series):
        return []
    rows: list[Any] = []
    for idx in df.index[series.notna()]:
        text = str(series.at[idx]).strip()
        if (
            not _FULL_ISO_RE.fullmatch(text)
            and _PARTIAL_ISO_RE.fullmatch(text)
            and _valid_partial_date(text)
        ):
            rows.append(idx)
    return rows


def check_not_future(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present dates later than the run date (UTC)."""
    col = mapping.actual(rule.fields[0])
    parsed = to_datetime_safe(df[col], utc=True)
    today = pd.Timestamp.now(tz="UTC").normalize()
    future = parsed.notna() & (parsed.dt.normalize() > today)
    return df.index[future].tolist()


def check_numeric(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present values that are not numeric."""
    col = mapping.actual(rule.fields[0])
    series = df[col]
    bad = series.notna() & pd.to_numeric(series, errors="coerce").isna()
    return df.index[bad].tolist()


def check_nonneg(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present numeric values that are negative (non-numeric is a numeric rule's job)."""
    col = mapping.actual(rule.fields[0])
    numeric = pd.to_numeric(df[col], errors="coerce")
    return df.index[numeric.notna() & (numeric < 0)].tolist()


def check_nonneg_number(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present values that are non-numeric *or* negative (must be a number >= 0)."""
    col = mapping.actual(rule.fields[0])
    series = df[col]
    present = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    bad = (present & numeric.isna()) | (numeric.notna() & (numeric < 0))
    return df.index[bad].tolist()


def check_positive(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present values that are not a positive number (non-numeric or ``<= 0``)."""
    col = mapping.actual(rule.fields[0])
    series = df[col]
    present = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    bad = (present & numeric.isna()) | (numeric.notna() & (numeric <= 0))
    return df.index[bad].tolist()


def check_positive_integer(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag present values that are not a positive whole number."""
    col = mapping.actual(rule.fields[0])
    series = df[col]
    present = series.notna()
    numeric = pd.to_numeric(series, errors="coerce")
    is_pos_int = numeric.notna() & (numeric > 0) & (numeric == numeric.round())
    return df.index[present & ~is_pos_int].tolist()


def check_both_present(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag rows where exactly one of two fields is present (both-or-neither)."""
    a = _present(df, mapping, rule.fields[0])
    b = _present(df, mapping, rule.fields[1])
    if a is None or b is None:
        return []
    return df.index[a ^ b].tolist()


def check_requires_field(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag rows where ``fields[0]`` is present but the ``params.requires`` field is not."""
    primary = _present(df, mapping, rule.fields[0])
    if primary is None:
        return []
    dep = _present(df, mapping, str(rule.params["requires"]))
    if dep is None:
        return df.index[primary].tolist()
    return df.index[primary & ~dep].tolist()


def check_requires_when_value(
    df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
) -> list[Any]:
    """Flag rows where ``fields[0]`` equals a trigger value but ``params.requires`` is absent."""
    trigger_col = mapping.actual(rule.fields[0])
    values = rule.params.get("equals", [])
    if isinstance(values, str):
        values = [values]
    series = df[trigger_col]
    if rule.params.get("case_insensitive"):
        wanted = {str(v).casefold() for v in values}
        matched = series.astype("string").str.casefold().isin(wanted).fillna(False)
    else:
        matched = series.isin(list(values))
    dep = _present(df, mapping, str(rule.params["requires"]))
    if dep is None:
        return df.index[matched].tolist()
    return df.index[matched & ~dep].tolist()


def check_at_least_one(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag rows where every one of the named fields is null (need at least one value)."""
    fields = [*rule.fields, *rule.params.get("others", [])]
    cols = [mapping.actual(f) for f in fields]
    cols = [c for c in cols if c is not None]
    if not cols:
        return []
    all_null = df[cols[0]].isna()
    for col in cols[1:]:
        all_null = all_null & df[col].isna()
    return df.index[all_null].tolist()


def check_ge_date(df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
    """Flag rows where ``fields[0]`` is earlier than ``fields[1]`` (``fields[0] >= fields[1]``)."""
    # utc=True: one side may carry a UTC offset (e.g. a FHIR dateTime) while the other
    # is naive or uses a different offset; comparing in UTC (naive read as UTC) never
    # raises and does not change the ordering of two naive values.
    later = to_datetime_safe(df[mapping.actual(rule.fields[0])], utc=True)
    earlier = to_datetime_safe(df[mapping.actual(rule.fields[1])], utc=True)
    both = later.notna() & earlier.notna()
    return df.index[both & (later < earlier)].tolist()


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None


def redact_phi_actions(
    df: pd.DataFrame,
    log: Any,
    mapping: ColumnMapping,
    phi_fields: tuple[str, ...],
    include_phi: bool,
) -> None:
    """Make a repair log PHI-safe in place (shared by the healthcare/education packs).

    For every logged action that targets a PHI column: a *flagged* action is first
    enriched with the offending cell value (so the audit explains *what* was wrong),
    then — unless ``include_phi`` is set — its ``from``/``to`` values are masked to
    :data:`PHI_MASK`. Non-PHI actions and null values are left untouched.
    """
    phi_cols = {mapping.actual(f) for f in phi_fields if mapping.is_mapped(f)}
    phi_cols.discard(None)
    if not phi_cols:
        return
    for action in log.actions:
        if action.column not in phi_cols:
            continue
        if action.status == "flagged" and action.row in df.index:
            value = df.at[action.row, action.column]
            if not _is_missing(value):
                action.from_value = value
        if not include_phi:
            if action.from_value is not None:
                action.from_value = PHI_MASK
            if action.to_value is not None:
                action.to_value = PHI_MASK
