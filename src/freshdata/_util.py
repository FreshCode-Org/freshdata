"""Small shared helpers. Internal — no stability guarantees."""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.errors import PerformanceWarning


def add_column(df: pd.DataFrame, name: object, values: object) -> None:
    """Insert a new column in place, suppressing pandas' fragmentation notice.

    The engine deliberately appends per-column indicator/flag columns in a
    loop; on wide frames this trips ``PerformanceWarning: DataFrame is highly
    fragmented``. The advice (concat all at once) does not apply here because
    later columns can depend on rows removed by earlier ones, so we accept the
    fragmentation and quiet the noise rather than misreport it to users.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        df[name] = values

#: Major version of the installed pandas, for the few places behavior differs.
PANDAS_MAJOR: int = int(pd.__version__.split(".")[0])


#: Above this many rows, object payloads are estimated from a sample instead
#: of measured cell by cell, keeping report bookkeeping ~free on tall frames.
_MEMORY_SAMPLE_THRESHOLD = 200_000
_MEMORY_SAMPLE_SIZE = 20_000


def memory_bytes(df: pd.DataFrame) -> int:
    """Total memory footprint of *df* in bytes, including object payloads.

    Exact for frames up to ~200k rows; for taller frames the per-row payload
    of object/string columns is estimated from a 20k-row random sample (other
    dtypes are always exact — their size does not depend on values).
    """
    n = len(df)
    if n <= _MEMORY_SAMPLE_THRESHOLD:
        return int(df.memory_usage(deep=True).sum())
    total = int(df.memory_usage(deep=False).sum())
    for i, dtype in enumerate(df.dtypes):
        if not _is_stringlike_dtype(dtype):
            continue
        sample = df.iloc[:, i].sample(_MEMORY_SAMPLE_SIZE, random_state=0)
        payload = sample.memory_usage(deep=True, index=False) - sample.memory_usage(
            deep=False, index=False
        )
        total += int(payload / len(sample) * n)
    if _is_stringlike_dtype(df.index.dtype):
        idx = df.index.to_series().sample(_MEMORY_SAMPLE_SIZE, random_state=0)
        payload = idx.memory_usage(deep=True, index=False) - idx.memory_usage(
            deep=False, index=False
        )
        total += int(payload / len(idx) * n)
    return total


def format_bytes(n: float) -> str:
    """Render a byte count for humans: ``format_bytes(2048) == '2.0 KB'``."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def sample_series(s: pd.Series, size: int, random_state: int) -> pd.Series:
    """Return *s* itself if small, else a reproducible random sample of *size*."""
    if len(s) <= size:
        return s
    return s.sample(size, random_state=random_state)


def stringlike_columns(df: pd.DataFrame) -> list:
    """Column labels whose dtype can hold free-form text (object or string)."""
    return list(df.columns[[_is_stringlike_dtype(dt) for dt in df.dtypes]])


def _is_stringlike_dtype(dtype: object) -> bool:
    return pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype)
