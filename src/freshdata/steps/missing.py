"""Opt-in missing-value imputation.

Off by default: filling values changes the statistics of the data, so the user
must ask for it (``impute="auto" | "mean" | "median" | "mode"``). The "auto"
strategy uses the median for numeric columns and the mode for everything else.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import CleanConfig
from ..report import CleanReport


def _mode_value(s: pd.Series) -> Any | None:
    """Most frequent non-missing value; deterministic for ties when sortable."""
    try:
        modes = s.mode(dropna=True)
        if len(modes):
            return modes.iloc[0]
    except TypeError:
        pass  # mixed un-comparable types; fall back to frequency order
    counts = s.value_counts(dropna=True)
    return counts.index[0] if len(counts) else None


def _fill_value(s: pd.Series, strategy: str) -> Any | None:
    numeric = is_numeric_dtype(s) and not is_bool_dtype(s)
    if strategy == "auto":
        strategy = "median" if numeric else "mode"
    if strategy in ("mean", "median"):
        if not numeric:
            return None  # not defined for this dtype; caller reports the skip
        return s.mean() if strategy == "mean" else s.median()
    return _mode_value(s)


def _strategy_for_column(col: object, config: CleanConfig) -> str | None:
    if config.impute_strategy and str(col) in config.impute_strategy:
        return config.impute_strategy[str(col)]
    return config.impute


def impute_missing(df: pd.DataFrame, config: CleanConfig,
                   report: CleanReport) -> pd.DataFrame:
    """Fill missing values per column according to explicit impute config."""
    if config.impute is None and not config.impute_strategy:
        return df
    from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

    protected = hard_protected_columns(config, df.columns)
    missforest_columns = [
        col for col in df.columns
        if str(col) not in protected
        and _strategy_for_column(col, config) == "missforest"
        and int(df[col].isna().sum()) > 0
    ]
    if missforest_columns:
        from ..engine.context import build_contexts  # noqa: PLC0415
        from ..imputation.missforest import MissForestImputer  # noqa: PLC0415

        df = MissForestImputer(config, report).impute(
            df,
            missforest_columns,
            build_contexts(df, config),
        )

    for col in df.columns:
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        strategy = _strategy_for_column(col, config)
        if strategy is None or strategy == "missforest":
            continue
        s = df[col]
        n_missing = int(s.isna().sum())
        if n_missing == 0 or s.notna().sum() == 0:
            continue  # nothing to fill, or nothing to learn a fill value from
        value = _fill_value(s, strategy)
        if value is None or pd.isna(value):
            if strategy in ("mean", "median"):
                report.add("impute",
                           f"skipped ({strategy} is not defined for dtype {s.dtype})",
                           column=str(col))
            continue
        cast_note = ""
        try:
            filled = s.fillna(value)
        except (TypeError, ValueError):
            if is_numeric_dtype(s) and isinstance(value, float):
                # e.g. fractional median into an integer column
                filled = s.astype("float64").fillna(value)
                cast_note = ", column cast to float64"
            else:
                # e.g. value not representable in this dtype
                report.add("impute", f"skipped (could not fill dtype {s.dtype})",
                           column=str(col))
                continue
        df[col] = filled
        shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
        report.add("impute",
                   f"filled {n_missing} missing value(s) with {strategy} ({shown}{cast_note})",
                   column=str(col), count=n_missing)
    return df
