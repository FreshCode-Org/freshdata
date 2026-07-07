"""Minimal, function-first Core surface for freshdata.

This module offers six plain functions -- :func:`fill_missing`,
:func:`detect_outliers`, :func:`remove_outliers`, :func:`resolve_duplicates`,
:func:`group_aggregate` and :func:`pipeline_clean` -- that each take a DataFrame
and a handful of simply-named arguments. It is a deliberately small,
self-contained alternative to the full :func:`freshdata.clean` pipeline: no
config objects, no report objects, just DataFrames in and DataFrames out.

Every transformer copies its input by default and mutates in place only when
``inplace=True`` is passed, matching the library's "never mutates your data"
promise. Defaults are deterministic -- the same input always yields the same
output -- and ``seed`` is accepted (but unused) by the outlier functions purely
for forward compatibility with future stochastic methods.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

__all__ = [
    "detect_outliers",
    "fill_missing",
    "group_aggregate",
    "pipeline_clean",
    "remove_outliers",
    "resolve_duplicates",
]

# Tukey fence multiplier for IQR; standard-deviation multiplier for z-score.
_DEFAULT_FACTOR = {"iqr": 1.5, "zscore": 3.0}

_FILL_METHODS = ("auto", "mean", "median", "mode", "constant", "ffill", "bfill")
_OUTLIER_METHODS = ("iqr", "zscore")
_DUPLICATE_METHODS = ("first", "last", "drop")


def _as_column_list(columns: str | Sequence[str] | None) -> list[str] | None:
    """Normalize a ``columns`` argument to a concrete list, or None for 'all'."""
    if columns is None:
        return None
    if isinstance(columns, str):
        return [columns]
    return list(columns)


def _resolve_columns(
    df: pd.DataFrame,
    columns: str | Sequence[str] | None,
    *,
    numeric_only: bool = False,
) -> list[str]:
    """Return the columns to operate on, validating that any named ones exist."""
    cols = _as_column_list(columns)
    if cols is None:
        cols = list(df.columns)
        if numeric_only:
            cols = [c for c in cols if _is_outlier_numeric(df[c])]
        return cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    return cols


def _is_outlier_numeric(s: pd.Series) -> bool:
    """True for numeric columns eligible for outlier math (excludes booleans).

    ``is_numeric_dtype`` reports True for bool columns, but IQR/z-score math on
    booleans is meaningless and raises on the boolean ``-`` operator, so bool
    columns are treated as non-numeric here.
    """
    return is_numeric_dtype(s) and not is_bool_dtype(s)


def _mode_value(s: pd.Series) -> object | None:
    """Most-frequent non-null value, or None when the column is all-null."""
    modes = s.dropna().mode()
    return modes.iloc[0] if not modes.empty else None


def _outlier_mask(
    df: pd.DataFrame,
    cols: Sequence[str],
    method: str,
    threshold: float | None,
) -> pd.Series:
    """Boolean Series: True where a row is an outlier in at least one column.

    Shared by :func:`detect_outliers` and :func:`remove_outliers` so the two stay
    perfectly consistent. Constant or undefined columns contribute no outliers.
    """
    factor = _DEFAULT_FACTOR[method] if threshold is None else threshold
    mask = pd.Series(False, index=df.index)
    for col in cols:
        s = df[col]
        if not _is_outlier_numeric(s):
            continue
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - factor * iqr, q3 + factor * iqr
        else:  # zscore
            mean, std = s.mean(), s.std()
            lo, hi = mean - factor * std, mean + factor * std
        if pd.isna(lo) or pd.isna(hi) or lo == hi:
            continue  # all-null / constant column -- nothing to flag
        mask |= (s < lo) | (s > hi)
    return mask


def fill_missing(
    df: pd.DataFrame,
    columns: str | Sequence[str] | None = None,
    method: str = "auto",
    value: object = None,
    inplace: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fill missing values in ``columns`` (all columns by default).

    ``method`` is one of ``"auto"`` (median for numeric columns, mode for the
    rest), ``"mean"``, ``"median"``, ``"mode"``, ``"constant"`` (fills with
    ``value``), ``"ffill"`` or ``"bfill"``. Returns the filled frame -- a copy
    unless ``inplace=True``.
    """
    if method not in _FILL_METHODS:
        raise ValueError(f"method must be one of {_FILL_METHODS}, got {method!r}")
    if method == "constant" and value is None:
        raise ValueError("method='constant' requires a value")
    if not inplace:
        df = df.copy()
    cols = _resolve_columns(df, columns)
    filled = 0
    for col in cols:
        s = df[col]
        before = int(s.isna().sum())
        if before == 0:
            continue
        if method == "ffill":
            df[col] = s.ffill()
        elif method == "bfill":
            df[col] = s.bfill()
        elif method == "constant":
            df[col] = s.fillna(value)
        else:
            strategy = method
            if strategy == "auto":
                strategy = "median" if is_numeric_dtype(s) else "mode"
            if strategy == "mode":
                fill = _mode_value(s)
            elif is_numeric_dtype(s):
                fill = s.mean() if strategy == "mean" else s.median()
            else:
                fill = None  # mean/median are undefined for non-numeric columns
            if fill is None:
                continue
            df[col] = s.fillna(fill)
        filled += before - int(df[col].isna().sum())
    if verbose:
        print(
            f"fill_missing: filled {filled} cell(s) across "
            f"{len(cols)} column(s) using method={method!r}"
        )
    return df


def detect_outliers(
    df: pd.DataFrame,
    columns: str | Sequence[str] | None = None,
    method: str = "iqr",
    threshold: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> pd.Series:
    """Flag outlier rows via ``"iqr"`` (Tukey fences) or ``"zscore"``.

    Returns a boolean Series aligned to ``df.index``: True marks a row that is an
    outlier in at least one scanned numeric column. ``threshold`` overrides the
    per-method default (1.5 for ``"iqr"``, 3.0 for ``"zscore"``). ``seed`` is
    accepted for forward compatibility and unused by these deterministic methods.
    """
    if method not in _OUTLIER_METHODS:
        raise ValueError(f"method must be one of {_OUTLIER_METHODS}, got {method!r}")
    cols = _resolve_columns(df, columns, numeric_only=True)
    mask = _outlier_mask(df, cols, method, threshold)
    if verbose:
        print(
            f"detect_outliers: {int(mask.sum())} outlier row(s) across "
            f"{len(cols)} column(s) using method={method!r}"
        )
    return mask


def remove_outliers(
    df: pd.DataFrame,
    columns: str | Sequence[str] | None = None,
    method: str = "iqr",
    threshold: float | None = None,
    inplace: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Drop rows flagged by :func:`detect_outliers` (same method/threshold).

    The original index is preserved (not reset). Returns the filtered frame; with
    ``inplace=True`` the flagged rows are dropped from the input, which is returned.
    """
    if method not in _OUTLIER_METHODS:
        raise ValueError(f"method must be one of {_OUTLIER_METHODS}, got {method!r}")
    cols = _resolve_columns(df, columns, numeric_only=True)
    mask = _outlier_mask(df, cols, method, threshold)
    if inplace:
        df.drop(index=df.index[mask], inplace=True)
        result = df
    else:
        result = df.loc[~mask]
    if verbose:
        print(f"remove_outliers: removed {int(mask.sum())} row(s) using method={method!r}")
    return result


def resolve_duplicates(
    df: pd.DataFrame,
    columns: str | Sequence[str] | None = None,
    method: str = "first",
    inplace: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Resolve duplicate rows.

    ``columns`` is the subset that defines a duplicate (the whole row by
    default). ``method`` ``"first"``/``"last"`` keeps that occurrence of each
    duplicate group; ``"drop"`` removes every row that has any duplicate. Row
    order is preserved. Returns the de-duplicated frame -- a copy unless
    ``inplace=True``.
    """
    if method not in _DUPLICATE_METHODS:
        raise ValueError(f"method must be one of {_DUPLICATE_METHODS}, got {method!r}")
    subset = _resolve_columns(df, columns) if columns is not None else None
    keep: str | bool = False if method == "drop" else method
    drop_mask = df.duplicated(subset=subset, keep=keep)
    if inplace:
        df.drop(index=df.index[drop_mask], inplace=True)
        result = df
    else:
        result = df.loc[~drop_mask]
    if verbose:
        print(f"resolve_duplicates: removed {int(drop_mask.sum())} row(s) using method={method!r}")
    return result


def group_aggregate(
    df: pd.DataFrame,
    by: str | Sequence[str],
    agg: str | Sequence[str] | Mapping[str, object] = "mean",
    columns: str | Sequence[str] | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Group rows by ``by`` and aggregate.

    ``agg`` may be a function name (``"mean"``), a list applied to every
    aggregated column, or a ``{column: function}`` mapping. ``columns`` selects
    which columns to aggregate (all non-``by`` numeric columns by default, and
    ignored when ``agg`` is a mapping). Group keys are returned as ordinary
    columns with a reset index.
    """
    by_list = _as_column_list(by)
    if not by_list:
        raise ValueError("by is required")
    missing = [c for c in by_list if c not in df.columns]
    if missing:
        raise KeyError(f"group columns not found: {missing}")
    grouped = df.groupby(by_list, sort=False, dropna=False, as_index=False)
    if isinstance(agg, Mapping):
        result = grouped.agg(agg)
    else:
        if columns is None:
            cols = [
                c for c in df.columns if c not in by_list and is_numeric_dtype(df[c])
            ]
        else:
            cols = _resolve_columns(df, columns)
        result = grouped[cols].agg(agg)
    result = result.reset_index(drop=True)
    if verbose:
        print(f"group_aggregate: produced {len(result)} group(s) over {by_list}")
    return result


def pipeline_clean(
    df: pd.DataFrame,
    columns: str | Sequence[str] | None = None,
    method: str = "auto",
    threshold: float | None = None,
    inplace: bool = False,
    seed: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run the standard clean in a fixed, deterministic order.

    1. :func:`fill_missing` (using ``method``)
    2. :func:`remove_outliers` (IQR, using ``threshold``)
    3. :func:`resolve_duplicates` (keep first)

    ``columns`` scopes every step. Returns the cleaned frame -- a copy unless
    ``inplace=True``.
    """
    if not inplace:
        df = df.copy()
    df = fill_missing(df, columns=columns, method=method, inplace=True, verbose=verbose)
    df = remove_outliers(
        df, columns=columns, method="iqr", threshold=threshold, inplace=True,
        seed=seed, verbose=verbose,
    )
    return resolve_duplicates(df, columns=columns, method="first", inplace=True, verbose=verbose)
