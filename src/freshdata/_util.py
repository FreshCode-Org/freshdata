"""Small shared helpers. Internal — no stability guarantees."""

from __future__ import annotations

import hashlib
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


#: Leading characters Excel/Sheets/LibreOffice interpret as a formula
#: (OWASP CSV-injection guidance).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _formula_guard(value: object) -> object:
    # lstrip: spreadsheets ignore leading whitespace when deciding whether a
    # cell is a formula, so " =1+1" is just as live as "=1+1".
    if isinstance(value, str) and value.lstrip(" \t").startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def sanitize_csv_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of *df* safe to open in a spreadsheet: string cells (and column
    labels) starting with ``= + - @ <tab> <cr>`` — including after leading
    whitespace — are prefixed with ``'`` so they render as text instead of
    executing as formulas. Non-string cells (including negative numbers) are
    untouched.
    """
    out = df.copy()
    for i, dtype in enumerate(out.dtypes):
        if _is_stringlike_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            column = out.iloc[:, i]
            guarded = column.astype(object).map(_formula_guard)
            if not guarded.equals(column.astype(object)):
                out.isetitem(i, guarded)
    out.columns = pd.Index([_formula_guard(c) for c in out.columns])
    return out


def mask_sensitive_value(value: object) -> str:
    """Deterministic stand-in for a sensitive value in report text.

    The short digest lets two mentions of the same value be correlated
    without disclosing it; the token never round-trips to the original.
    """
    digest = hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:8]
    return f"[SENSITIVE:{digest}]"
