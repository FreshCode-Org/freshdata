"""Polars adapter — accept pl.DataFrame at the API boundary."""

from __future__ import annotations

from typing import Any

import pandas as pd

_POLARS: Any = None


def _polars_module():
    global _POLARS
    if _POLARS is None:
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars support requires polars. "
                "Install with: pip install freshdata-cleaner[polars]"
            ) from exc
        _POLARS = pl
    return _POLARS


def is_polars_frame(obj: object) -> bool:
    try:
        pl = _polars_module()
    except ImportError:
        return False
    return isinstance(obj, pl.DataFrame)


def to_pandas(df: object) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if is_polars_frame(df):
        pl_df: Any = df
        return pl_df.to_pandas()
    raise TypeError(f"expected pandas or polars DataFrame, got {type(df).__name__}")


def from_pandas(df: pd.DataFrame, original: object | None = None) -> object:
    if original is None or isinstance(original, pd.DataFrame):
        return df
    if is_polars_frame(original):
        pl = _polars_module()
        return pl.from_pandas(df)
    return df
