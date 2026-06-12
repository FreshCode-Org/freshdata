"""Duplicate-row handling: exact duplicates, subset duplicates, aggregation.

Exact duplicate rows are removed by default (keeping the first occurrence).
With ``duplicate_subset`` set, rows are compared on those columns only and
``duplicate_keep`` chooses the resolution: keep ``"first"``/``"last"``,
``"drop"`` every member of a duplicated group, or ``"aggregate"`` groups into
one row (numeric mean, first non-missing otherwise).

Safety rules:

- Time-indexed frames (``DatetimeIndex``) never lose rows unless
  ``allow_timeseries_duplicates=True`` — repeated readings are often real.
- A duplicate ratio above ``duplicate_threshold`` raises a warning in the
  report: that much duplication usually means an upstream join or export bug.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import CleanConfig
from ..report import CleanReport


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


def _aggregate_duplicates(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    """Collapse each duplicated group into one row (mean / first non-null)."""
    agg = {
        c: "mean" if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c]) else "first"
        for c in df.columns if c not in subset
    }
    grouped = df.groupby(subset, sort=False, dropna=False, as_index=False).agg(agg)
    return grouped[list(df.columns)]


def drop_duplicate_rows(df: pd.DataFrame, config: CleanConfig,
                        report: CleanReport) -> pd.DataFrame:
    """Resolve duplicate rows according to ``duplicate_keep``.

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
            df = _aggregate_duplicates(df, subset)
    if keep in ("first", "last"):
        df = df.loc[~df.duplicated(subset=subset, keep=keep)]
    elif keep == "drop":
        df = df.loc[~df.duplicated(subset=subset, keep=False)]

    n_removed = n_before - len(df)
    verb = {"first": "dropped", "last": "dropped", "drop": "dropped",
            "aggregate": "aggregated away"}[config.duplicate_keep]
    report.add(
        "drop_duplicates",
        f"{verb} {n_removed} duplicate row(s) ({pct:.1f}% of rows, "
        f"keep={config.duplicate_keep!r}){where}",
        count=n_removed,
        risk="medium" if n_dup / n_before > config.duplicate_threshold else "low",
    )
    report.duplicates_removed += n_removed
    if n_dup / n_before > config.duplicate_threshold:
        report.add_warning(
            f"duplicate ratio {pct:.1f}% exceeds duplicate_threshold "
            f"({100 * config.duplicate_threshold:.0f}%); check for an upstream "
            "join or export problem"
        )
        report.add_recommendation(
            "review why so many rows were duplicated before trusting downstream stats"
        )
    return df
