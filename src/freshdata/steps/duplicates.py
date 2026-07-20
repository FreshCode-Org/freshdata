"""Duplicate-row handling: exact duplicates, subset duplicates, aggregation.

Duplicate rows are *detected and reported* by default but never removed;
removal is opt-in via ``drop_duplicates=True`` (keeping the first occurrence).
With ``duplicate_subset`` set, rows are compared on those columns only and
``duplicate_keep`` chooses the resolution: keep ``"first"``/``"last"``,
``"drop"`` every member of a duplicated group, or ``"aggregate"`` groups into
one row (numeric mean, first non-missing otherwise).

Safety rules:

- Time-indexed frames (``DatetimeIndex``) never lose rows unless
  ``allow_timeseries_duplicates=True`` — repeated readings are often real.
- A duplicate ratio above ``duplicate_threshold`` raises a warning in the
  report (or :class:`DuplicateRatioError` with
  ``duplicate_ratio_action="error"``): that much duplication usually means an
  upstream join or export bug.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import CleanConfig
from ..report import CleanReport


class DuplicateRatioError(ValueError):
    """Raised when the duplicate ratio exceeds ``duplicate_threshold`` and
    ``duplicate_ratio_action="error"`` asked for a hard stop."""


def check_duplicate_ratio(n_dup: int, n_before: int, config: CleanConfig) -> bool:
    """True when the duplicate ratio exceeds ``duplicate_threshold``.

    Raises :class:`DuplicateRatioError` instead under
    ``duplicate_ratio_action="error"``. Shared by the pandas step and the
    native execution backends so the escalation knob behaves identically on
    every engine.
    """
    if n_before <= 0:
        return False
    if n_dup / n_before <= config.duplicate_threshold:
        return False
    if config.duplicate_ratio_action == "error":
        raise DuplicateRatioError(
            f"duplicate ratio {100.0 * n_dup / n_before:.1f}% exceeds "
            f"duplicate_threshold ({100 * config.duplicate_threshold:.0f}%) and "
            'duplicate_ratio_action="error" is set; check for an upstream '
            "join or export problem"
        )
    return True


def report_detected_duplicates(
    n_dup: int,
    n_before: int,
    config: CleanConfig,
    report: CleanReport,
    subset: list | None = None,
) -> None:
    """Detection-only reporting for the ``drop_duplicates=False`` default.

    Shared by the pandas step and the native execution backends so reports
    stay backend-identical: an action records the detection, a strong warning
    fires above ``duplicate_threshold``, and ``duplicate_ratio_action="error"``
    escalates to :class:`DuplicateRatioError`.
    """
    if n_dup <= 0 or n_before <= 0:
        return
    high_ratio = check_duplicate_ratio(n_dup, n_before, config)
    pct = 100.0 * n_dup / n_before
    where = f" (compared on {subset})" if subset else ""
    report.add(
        "drop_duplicates",
        f"detected {n_dup} duplicate row(s) ({pct:.1f}%){where}, none removed",
        rationale="drop_duplicates=False (default): duplicate rows are "
                  "reported, never removed; pass drop_duplicates=True to "
                  "remove them",
        risk="low",
    )
    if high_ratio:
        report.add_warning(
            f"duplicate ratio {pct:.1f}% exceeds duplicate_threshold "
            f"({100 * config.duplicate_threshold:.0f}%); duplicates were "
            "NOT removed — pass drop_duplicates=True to remove them, and "
            "check for an upstream join or export problem"
        )
        report.add_recommendation(
            "review why so many rows were duplicated before trusting "
            "downstream stats"
        )


def _validated_subset(df: pd.DataFrame, config: CleanConfig) -> list | None:
    if config.duplicate_subset is None:
        return None
    subset = list(config.duplicate_subset)
    missing = [c for c in subset if c not in df.columns]
    if missing:
        raise ValueError(
            f"duplicate_subset column(s) not found: {missing}. "
            f"Available columns: {list(df.columns)}. "
            "Note: names refer to columns *after* renaming when column_names=True."
        )
    return subset


def _aggregate_duplicates(
    df: pd.DataFrame, subset: list, protected: tuple[str, ...] = ()
) -> pd.DataFrame:
    """Collapse each duplicated group into one row (mean / first non-null).

    Context-protected columns always aggregate as ``"first"`` (their values
    must survive byte-identical), and when any exist the surviving rows keep
    their original index labels so the hard guard can align them.
    """
    agg = {
        c: "mean"
        if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c]) and str(c) not in protected
        else "first"
        for c in df.columns if c not in subset
    }
    if not protected:
        grouped = df.groupby(subset, sort=False, dropna=False, as_index=False).agg(agg)
        return grouped[list(df.columns)]
    marker = "__fd_orig_index__"
    while marker in df.columns:
        marker = "_" + marker
    tmp = df.copy(deep=False)
    tmp[marker] = df.index
    agg[marker] = "first"
    grouped = tmp.groupby(subset, sort=False, dropna=False, as_index=False).agg(agg)
    grouped.index = pd.Index(grouped.pop(marker).to_numpy())
    grouped.index.name = df.index.name
    return grouped[list(df.columns)]


def _filter_rows(df: pd.DataFrame, keep_mask: pd.Series) -> pd.DataFrame:
    """Filter rows without pandas boolean take, which can crash on some wheels."""
    mask = keep_mask.to_numpy(dtype=bool, copy=True)
    out = pd.DataFrame(
        {col: df[col].to_numpy(copy=True)[mask] for col in df.columns},
        columns=df.columns,
    )
    for col in df.columns:
        out[col] = out[col].astype(df[col].dtype)
    out.index = df.index.to_numpy(copy=True)[mask]
    return out


def drop_duplicate_rows(df: pd.DataFrame, config: CleanConfig,
                        report: CleanReport) -> pd.DataFrame:
    """Detect duplicate rows; resolve them per ``duplicate_keep`` only when
    ``drop_duplicates=True`` (detection-and-report otherwise).

    Columns holding unhashable values (lists, dicts) make duplicate detection
    impossible; the step is then skipped and noted in the report rather than
    guessing.
    """
    if df.empty:
        return df
    subset = _validated_subset(df, config)
    try:
        dup_any = df.duplicated(subset=subset, keep="first")
    except TypeError:
        report.add("drop_duplicates",
                   "skipped: column(s) contain unhashable values (e.g. lists)")
        return df
    n_dup = int(dup_any.sum())
    if n_dup == 0:
        return df

    n_before = len(df)
    pct = 100.0 * n_dup / n_before
    where = f" (compared on {subset})" if subset else ""
    high_ratio = check_duplicate_ratio(n_dup, n_before, config)

    if not config.drop_duplicates:
        report_detected_duplicates(n_dup, n_before, config, report, subset=subset)
        return df

    if isinstance(df.index, pd.DatetimeIndex) and not config.allow_timeseries_duplicates:
        report.add(
            "drop_duplicates",
            f"preserved {n_dup} duplicate row(s) ({pct:.1f}%){where}",
            rationale="time-indexed data: repeated observations may be real "
                      "readings, so they are never auto-removed",
            risk="medium",
        )
        report.add_warning(
            f"{n_dup} duplicate row(s) preserved in time-indexed data; pass "
            "allow_timeseries_duplicates=True to remove them"
        )
        return df

    keep = config.duplicate_keep
    if keep == "aggregate":
        if subset is None:
            # Exact duplicates are identical in every column, so aggregation
            # degenerates to keeping the first occurrence.
            keep = "first"
        else:
            from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

            df = _aggregate_duplicates(
                df, subset, protected=hard_protected_columns(config, df.columns)
            )
    if keep in ("first", "last"):
        df = _filter_rows(df, ~df.duplicated(subset=subset, keep=keep))
    elif keep == "drop":
        df = _filter_rows(df, ~df.duplicated(subset=subset, keep=False))

    n_removed = n_before - len(df)
    verb = {"first": "dropped", "last": "dropped", "drop": "dropped",
            "aggregate": "aggregated away"}[config.duplicate_keep]
    report.add(
        "drop_duplicates",
        f"{verb} {n_removed} duplicate row(s) ({pct:.1f}% of rows, "
        f"keep={config.duplicate_keep!r}){where}",
        count=n_removed,
        risk="medium" if high_ratio else "low",
    )
    report.duplicates_removed += n_removed
    if high_ratio:
        report.add_warning(
            f"duplicate ratio {pct:.1f}% exceeds duplicate_threshold "
            f"({100 * config.duplicate_threshold:.0f}%); check for an upstream "
            "join or export problem"
        )
        report.add_recommendation(
            "review why so many rows were duplicated before trusting downstream stats"
        )
    return df
