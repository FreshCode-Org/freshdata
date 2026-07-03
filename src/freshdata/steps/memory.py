"""Opt-in memory optimization: numeric downcasting and category conversion.

Off by default because it changes dtypes callers may rely on, and float
downcasting (float64 -> float32) trades precision for space. Enabled with
``optimize_memory=True``; every change and the bytes saved are reported.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_float_dtype,
    is_integer_dtype,
)

from .._util import format_bytes, stringlike_columns
from ..config import CleanConfig
from ..engine.context import _ID_NAME
from ..report import CleanReport


def _is_identifier_column(col: object, config: CleanConfig) -> bool:
    """Identifiers/keys must not be narrow-downcast — overflow would corrupt
    them (e.g. an int8 key wraps at 128). Detected by explicit config or name."""
    label = str(col)
    if label in config.id_columns or label == config.target_column:
        return True
    return bool(_ID_NAME.search(label.casefold()))

_NULLABLE_INT_LADDER: list[tuple[str, int, int]] = [
    ("Int8", -(2**7), 2**7 - 1),
    ("Int16", -(2**15), 2**15 - 1),
    ("Int32", -(2**31), 2**31 - 1),
]


def _downcast_numeric(s: pd.Series) -> pd.Series:
    if is_bool_dtype(s):
        return s
    if is_integer_dtype(s):
        if pd.api.types.is_extension_array_dtype(s.dtype):  # nullable IntXX
            lo, hi = s.min(), s.max()
            if pd.isna(lo):
                return s
            for name, dmin, dmax in _NULLABLE_INT_LADDER:
                if dmin <= lo and hi <= dmax:
                    return s.astype(name)
            return s
        return pd.to_numeric(s, downcast="integer")
    if is_float_dtype(s) and s.dtype == "float64":
        return pd.to_numeric(s, downcast="float")
    return s


def optimize_memory(df: pd.DataFrame, config: CleanConfig,
                    report: CleanReport) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality text to category."""
    if not config.optimize_memory or df.empty:
        return df
    from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

    protected = hard_protected_columns(config, df.columns)

    n_downcast, downcast_saved = 0, 0
    for col in df.columns:
        s = df[col]
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if _is_identifier_column(col, config) and not is_float_dtype(s):
            continue  # never narrow an integer key/id — overflow corrupts it
        smaller = _downcast_numeric(s)
        if smaller.dtype != s.dtype:
            downcast_saved += int(s.memory_usage(deep=True) - smaller.memory_usage(deep=True))
            df[col] = smaller
            n_downcast += 1
    if n_downcast:
        report.add("optimize_memory",
                   f"downcast {n_downcast} numeric column(s), saved "
                   f"{format_bytes(downcast_saved)}",
                   count=n_downcast)

    n_cat, cat_saved = 0, 0
    for col in stringlike_columns(df):
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        s = df[col]
        try:
            n_unique = s.nunique(dropna=True)
        except TypeError:  # unhashable values cannot become categories
            continue
        if n_unique == 0 or n_unique / len(s) > config.category_threshold:
            continue
        as_cat = s.astype("category")
        saved = int(s.memory_usage(deep=True) - as_cat.memory_usage(deep=True))
        if saved <= 0:
            continue  # categories only help when they actually shrink the column
        df[col] = as_cat
        cat_saved += saved
        n_cat += 1
    if n_cat:
        report.add("optimize_memory",
                   f"converted {n_cat} text column(s) to category, saved "
                   f"{format_bytes(cat_saved)}",
                   count=n_cat)
    return df
