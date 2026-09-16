"""Polars adapter — accept pl.DataFrame at the API boundary."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

_POLARS: Any = None

#: Polars integer dtypes that map to a pandas nullable integer dtype of the same width.
_NULLABLE_INT_DTYPES = frozenset(
    {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"}
)


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


def is_polars_lazy(obj: object) -> bool:
    """True for an uncollected ``pl.LazyFrame``."""
    try:
        pl = _polars_module()
    except ImportError:
        return False
    return isinstance(obj, pl.LazyFrame)


def to_pandas(df: object) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    if is_polars_lazy(df):
        lazy: Any = df
        df = lazy.collect()  # the pandas pipeline needs the rows in memory
    if is_polars_frame(df):
        pl_df: Any = df
        try:
            converted = pl_df.to_pandas()
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "converting a Polars frame to pandas requires pyarrow; "
                'install it with `pip install "freshdata-cleaner[polars]"` '
                "(or `pip install pyarrow`)"
            ) from exc
        return restore_nullable_integers(pl_df, converted)
    raise TypeError(f"expected pandas or polars DataFrame, got {type(df).__name__}")


def restore_nullable_integers(pl_df: Any, converted: pd.DataFrame) -> pd.DataFrame:
    """Rebuild integer columns that hold nulls as pandas nullable integers.

    ``pl.DataFrame.to_pandas()`` renders an integer column that has nulls as
    ``float64``, because NumPy integers cannot hold one. That silently rounds
    every value a float64 cannot represent — ``2**53 + 1`` comes back as
    ``2**53``, and a ``UInt64`` beyond ``2**63`` loses far more. The integers
    are exact in polars, so rebuild those columns from the raw values plus a
    null mask, which is both lossless and what the same data would be in pandas
    to begin with. Columns without nulls already round-trip exactly and are
    left alone.
    """
    for name, dtype in zip(pl_df.columns, pl_df.dtypes):
        if str(dtype) not in _NULLABLE_INT_DTYPES:
            continue
        column = pl_df[name]
        if column.null_count() == 0:
            continue  # to_pandas() already gave us the exact NumPy integers
        values = np.asarray(column.fill_null(0).to_numpy())
        mask = np.asarray(column.is_null().to_numpy(), dtype=bool)
        converted[name] = pd.arrays.IntegerArray(values, mask)
    return converted


def from_pandas(df: pd.DataFrame, original: object | None = None) -> object:
    if original is None or isinstance(original, pd.DataFrame):
        return df
    if is_polars_frame(original):
        pl = _polars_module()
        return pl.from_pandas(df)
    if is_polars_lazy(original):
        pl = _polars_module()
        return pl.from_pandas(df).lazy()  # LazyFrame in, LazyFrame out
    return df
