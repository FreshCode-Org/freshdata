"""Structural pruning: fully-empty rows/columns and constant columns."""

from __future__ import annotations

import pandas as pd

from ..config import CleanConfig
from ..report import CleanReport


def drop_empty_columns(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """Drop columns in which every value is missing."""
    if df.empty:
        return df
    empty = df.isna().all()
    if empty.any():
        dropped = [str(c) for c in df.columns[empty]]
        df = df.loc[:, ~empty]
        report.columns_dropped.extend(dropped)
        report.add("drop_empty_columns",
                   f"dropped {len(dropped)} all-missing column(s): {', '.join(dropped[:6])}"
                   + (" …" if len(dropped) > 6 else ""),
                   count=len(dropped))
    return df


def drop_empty_rows(df: pd.DataFrame, report: CleanReport) -> pd.DataFrame:
    """Drop rows in which every value is missing."""
    if df.empty:
        return df
    empty = df.isna().all(axis=1)
    n = int(empty.sum())
    if n:
        df = df.loc[~empty]
        report.add("drop_empty_rows", f"dropped {n} all-missing row(s)", count=n)
    return df


def _is_constant(s: pd.Series) -> bool:
    try:
        return s.nunique(dropna=False) <= 1
    except TypeError:  # unhashable cells (lists/dicts) — treat as not constant
        return False


def drop_constant_columns(df: pd.DataFrame, config: CleanConfig,
                          report: CleanReport) -> pd.DataFrame:
    """Drop columns holding a single distinct value (including all-missing)."""
    if df.empty or len(df) < 2:
        return df
    constant = [c for c in df.columns if _is_constant(df[c])]
    if constant:
        names = [str(c) for c in constant]
        df = df.drop(columns=constant)
        report.columns_dropped.extend(names)
        report.add("drop_constant_columns",
                   f"dropped {len(names)} constant column(s): {', '.join(names[:6])}"
                   + (" …" if len(names) > 6 else ""),
                   count=len(names))
    return df
