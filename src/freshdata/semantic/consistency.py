"""Deterministic cross-field consistency checks with human-review routing.

Value experts see one column at a time; some contradictions only exist across
columns or against dataset context — a completion date before its enrollment
date, a measurement whose sibling unit column disagrees with the column's
norm, a retention policy contradicted by a repair window, a time window that
depends on a timezone declared as a *transition*.  These checks never mutate
data: each finding is routed to a human through a report warning that names
the column and the affected rows.

Privacy rule: warnings emitted here never contain cell values — only column
names and row labels — so declared-sensitive columns can be routed to review
without disclosing anything.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping

import pandas as pd

from ..config import CleanConfig
from ..report import CleanReport
from .types import SemanticContext

_START_NAME = re.compile(
    r"(?:^|_)(?:enroll(?:ment)?|incident|signup|start|begin|admission|admit|hire)",
    re.I,
)
_END_NAME = re.compile(
    r"(?:^|_)(?:complet(?:ion|ed)?|report(?:ed)?|closed?|end|discharged?|"
    r"resolved?|finish(?:ed)?)",
    re.I,
)
_RETENTION_NAME = re.compile(r"retention|retain", re.I)
_REPAIR_NAME = re.compile(r"repair|purge|delete|erase|dispos", re.I)
_DURATION = re.compile(r"(\d+)\s*(year|month|week|day)s?", re.I)
_DURATION_DAYS = {"year": 365, "month": 30, "week": 7, "day": 1}
_TEMP_NAME = re.compile(r"temp", re.I)
_TZ_NAME = re.compile(r"time_?zone|(?:^|_)tz(?:$|_)", re.I)
_TZ_TRANSITION = re.compile(r"\S\s*(?:→|->)\s*\S")
_WINDOW_VALUE = re.compile(
    r"\d{1,2}:\d{2}\s*[-–—]\s*(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*)?\d{1,2}:\d{2}"
)
_MONEY_NAME = re.compile(
    r"price|amount|cost|fee|balance|payment|revenue|salary|premium|reserve|"
    r"total|budget|charge",
    re.I,
)

#: Warnings list at most this many row labels; a finding wider than the cap is
#: reported in aggregate so warnings stay bounded on megaframes.
_MAX_NAMED_ROWS = 20


def _rows_text(rows: Iterable[object]) -> str:
    named = [f"(row {row})" for row in list(rows)[:_MAX_NAMED_ROWS]]
    return " ".join(named)


def _as_datetime(series: pd.Series) -> pd.Series | None:
    """Parse a column to datetimes, or ``None`` when it clearly is not one."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    if series.dtype != object:
        return None
    nonnull = series.dropna()
    if len(nonnull) < 4 or not all(isinstance(v, str) for v in nonnull.head(20)):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except (TypeError, ValueError, OverflowError):
            return None
    if parsed.notna().sum() / len(nonnull) < 0.6:
        return None
    return parsed


def _modal_value(series: pd.Series) -> object:
    try:
        counts = series.value_counts(dropna=True)
    except TypeError:
        return None
    if counts.empty:
        return None
    return counts.index[0]


def _check_date_pair_ordering(df: pd.DataFrame, report: CleanReport) -> None:
    """A (start, end) date-column pair whose dominant ordering breaks in a few
    rows: route the deviating side to review."""
    starts = [c for c in df.columns if _START_NAME.search(str(c))]
    ends = [c for c in df.columns if _END_NAME.search(str(c))]
    for start_col in starts:
        start_parsed = _as_datetime(df[start_col])
        if start_parsed is None:
            continue
        for end_col in ends:
            end_parsed = _as_datetime(df[end_col])
            if end_parsed is None or str(end_col) == str(start_col):
                continue
            both = start_parsed.notna() & end_parsed.notna()
            n = int(both.sum())
            if n < 8:
                continue
            ordered = (end_parsed > start_parsed) & both
            share = int(ordered.sum()) / n
            if share < 0.75:
                continue
            # A reversal (end before start) contradicts any dominant ordering.
            # Mere equality (a same-day completion) is only suspicious when
            # the pair's strict ordering is otherwise near-invariant.
            violations = both & (end_parsed < start_parsed)
            if share >= 0.9:
                violations = violations | (both & (end_parsed == start_parsed))
            count = int(violations.sum())
            if not 0 < count <= max(3, int(0.1 * n)):
                continue
            start_modal = _modal_value(df[start_col][both])
            end_modal = _modal_value(df[end_col][both])
            per_column: dict[str, list[object]] = {}
            for row in df.index[violations]:
                start_deviates = df.at[row, start_col] != start_modal
                end_deviates = df.at[row, end_col] != end_modal
                if end_deviates and not start_deviates:
                    per_column.setdefault(str(end_col), []).append(row)
                elif start_deviates and not end_deviates:
                    per_column.setdefault(str(start_col), []).append(row)
                else:
                    per_column.setdefault(str(end_col), []).append(row)
                    per_column.setdefault(str(start_col), []).append(row)
            for column, rows in per_column.items():
                other = str(start_col) if column == str(end_col) else str(end_col)
                report.add_warning(
                    f"column '{column}': {len(rows)} row(s) break the date "
                    f"ordering the column pair with '{other}' otherwise "
                    f"follows {_rows_text(rows)}. Review these rows; values "
                    "were NOT changed."
                )


def _check_future_start_dates(
    df: pd.DataFrame, semantic_context: object, report: CleanReport
) -> None:
    """A start-of-record date after the declared reference date cannot have
    happened yet: route it to review."""
    if not isinstance(semantic_context, Mapping):
        return
    reference = semantic_context.get("reference_date")
    if not reference:
        return
    try:
        reference_ts = pd.Timestamp(str(reference))
    except (ValueError, TypeError):
        return
    for col in df.columns:
        if not _START_NAME.search(str(col)):
            continue
        parsed = _as_datetime(df[col])
        if parsed is None:
            continue
        n = int(parsed.notna().sum())
        future = parsed.notna() & (parsed > reference_ts)
        count = int(future.sum())
        if n < 8 or not 0 < count <= max(3, int(0.1 * n)):
            continue
        rows = list(df.index[future])
        report.add_warning(
            f"column '{col}': {count} date(s) lie after the declared "
            f"reference date {reference_ts.date().isoformat()} "
            f"{_rows_text(rows)}. Review these rows; values were NOT changed."
        )


def _duration_days(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    match = _DURATION.search(value)
    if match is None:
        return None
    return int(match.group(1)) * _DURATION_DAYS[match.group(2).lower()]


def _check_policy_durations(df: pd.DataFrame, report: CleanReport) -> None:
    """A repair/purge window shorter than the declared retention duration is a
    governance contradiction: route both policy cells to review."""
    retention_cols = [c for c in df.columns if _RETENTION_NAME.search(str(c))]
    repair_cols = [c for c in df.columns if _REPAIR_NAME.search(str(c))]
    for retention_col in retention_cols:
        for repair_col in repair_cols:
            if str(repair_col) == str(retention_col):
                continue
            rows: list[object] = []
            for row in df.index:
                retention = _duration_days(df.at[row, retention_col])
                repair = _duration_days(df.at[row, repair_col])
                if retention is not None and repair is not None and repair < retention:
                    rows.append(row)
            if not rows:
                continue
            report.add_warning(
                f"column '{retention_col}': the retention duration is "
                f"contradicted by the shorter repair window in column "
                f"'{repair_col}' for {len(rows)} row(s) {_rows_text(rows)}. "
                "Review these rows; policies were NOT changed."
            )
            report.add_warning(
                f"column '{repair_col}': the repair window is shorter than "
                f"the retention duration in column '{retention_col}' for "
                f"{len(rows)} row(s) {_rows_text(rows)}. Review these rows; "
                "policies were NOT changed."
            )


def _check_unit_column_conflicts(df: pd.DataFrame, report: CleanReport) -> None:
    """A measurement whose sibling ``<col>_unit`` deviates from the dominant
    unit is denominated differently from the rest of the column."""
    for col in df.columns:
        unit_col = f"{col}_unit"
        if unit_col not in df.columns:
            continue
        units = df[unit_col]
        nonnull = units.dropna()
        n = len(nonnull)
        if n < 8:
            continue
        dominant = _modal_value(units)
        if dominant is None:
            continue
        share = int((nonnull == dominant).sum()) / n
        if share < 0.75:
            continue
        deviating = units.notna() & (units != dominant)
        count = int(deviating.sum())
        if not 0 < count <= max(3, int(0.1 * n)):
            continue
        rows = list(df.index[deviating])
        report.add_warning(
            f"column '{col}': {count} value(s) are denominated in a different "
            f"unit than the column's dominant unit (see column '{unit_col}') "
            f"{_rows_text(rows)}. Review these rows; values were NOT converted."
        )


def _check_fahrenheit_in_celsius(df: pd.DataFrame, report: CleanReport) -> None:
    """A temperature far outside the column's range whose Fahrenheit-to-Celsius
    conversion fits the range reads as a unit mix-up: route it to review."""
    for col in df.columns:
        if not _TEMP_NAME.search(str(col)):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        nonnull = numeric.dropna()
        if len(nonnull) < 8:
            continue
        flagged: list[object] = []
        for row in df.index[numeric.notna()]:
            value = float(numeric.at[row])
            others = nonnull[nonnull.index != row]
            if len(others) < 4 or value <= float(others.max()) + 10:
                continue
            converted = (value - 32.0) * 5.0 / 9.0
            if float(others.min()) - 0.5 <= converted <= float(others.max()) + 0.5:
                flagged.append(row)
        if not 0 < len(flagged) <= max(2, int(0.1 * len(nonnull))):
            continue
        report.add_warning(
            f"column '{col}': {len(flagged)} value(s) read as Fahrenheit in a "
            f"Celsius-ranged column (the converted value fits the column's "
            f"range) {_rows_text(flagged)}. Review these rows; values were "
            "NOT converted."
        )


def _check_timezone_transitions(df: pd.DataFrame, report: CleanReport) -> None:
    """A timezone cell declaring a transition (``A→B``) makes the row's time
    windows uninterpretable: route the zone and the dependent windows."""
    for tz_col in df.columns:
        if not _TZ_NAME.search(str(tz_col)):
            continue
        values = df[tz_col]
        transitions = [
            row
            for row in df.index[values.notna()]
            if isinstance(values.at[row], str)
            and _TZ_TRANSITION.search(values.at[row])
        ]
        n = int(values.notna().sum())
        if n < 8 or not 0 < len(transitions) <= max(3, int(0.1 * n)):
            continue
        report.add_warning(
            f"column '{tz_col}': {len(transitions)} value(s) declare a "
            f"timezone transition rather than a single zone "
            f"{_rows_text(transitions)}. Review these rows; values were NOT "
            "changed."
        )
        for other in df.columns:
            if str(other) == str(tz_col):
                continue
            window_rows = [
                row
                for row in transitions
                if isinstance(df.at[row, other], str)
                and _WINDOW_VALUE.search(df.at[row, other])
            ]
            if not window_rows:
                continue
            report.add_warning(
                f"column '{other}': {len(window_rows)} time window(s) cannot "
                f"be interpreted because the row's timezone (column "
                f"'{tz_col}') declares a transition {_rows_text(window_rows)}. "
                "Review these rows; values were NOT changed."
            )


def _check_negative_amounts(df: pd.DataFrame, report: CleanReport) -> None:
    """A rare negative amount in a predominantly non-negative monetary column
    is an anomaly worth human eyes."""
    for col in df.columns:
        if not _MONEY_NAME.search(str(col)):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        nonnull = numeric.dropna()
        n = len(nonnull)
        if n < 8:
            continue
        negative = numeric.notna() & (numeric < 0)
        count = int(negative.sum())
        if not 0 < count <= max(2, int(0.1 * n)):
            continue
        if (n - count) / n < 0.9:
            continue
        rows = list(df.index[negative])
        report.add_warning(
            f"column '{col}': {count} negative amount(s) in a predominantly "
            f"non-negative monetary column {_rows_text(rows)}. Review these "
            "rows; values were NOT changed."
        )


def _check_sensitive_anomalies(
    df: pd.DataFrame, config: CleanConfig, report: CleanReport
) -> None:
    """In a declared-sensitive column dominated by one repeating value, a rare
    deviation is an anomaly that must reach a human — with the value withheld."""
    for col in config.sensitive_columns:
        if col not in df.columns:
            continue
        series = df[col]
        nonnull = series.dropna()
        n = len(nonnull)
        if n < 8:
            continue
        dominant = _modal_value(series)
        if dominant is None:
            continue
        share = int((nonnull == dominant).sum()) / n
        if share < 0.75:
            continue
        deviating = series.notna() & (series != dominant)
        count = int(deviating.sum())
        if not 0 < count <= max(3, int(0.1 * n)):
            continue
        rows = list(df.index[deviating])
        report.add_warning(
            f"column '{col}': {count} value(s) deviate from the column's "
            f"dominant pattern {_rows_text(rows)}. Review these rows; the "
            "values are withheld because the column is declared sensitive."
        )


def run_consistency_checks(
    df: pd.DataFrame,
    config: CleanConfig,
    ctx: SemanticContext,
    report: CleanReport,
) -> None:
    """Run every cross-field check; only warnings are ever emitted."""
    del ctx  # column metadata is not needed yet; kept for interface stability
    _check_date_pair_ordering(df, report)
    _check_future_start_dates(df, config.semantic_context, report)
    _check_policy_durations(df, report)
    _check_unit_column_conflicts(df, report)
    _check_fahrenheit_in_celsius(df, report)
    _check_timezone_transitions(df, report)
    _check_negative_amounts(df, report)
    _check_sensitive_anomalies(df, config, report)


__all__ = ["run_consistency_checks"]
