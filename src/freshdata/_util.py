"""Small shared helpers. Internal — no stability guarantees."""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import threading
import warnings
from fractions import Fraction
from typing import Any

import pandas as pd
from pandas.errors import PerformanceWarning


def safe_median(s: pd.Series) -> Any:
    """``s.median()`` that also works on nullable integer columns.

    pandas computes a masked-integer median by filling masked slots with
    ``iNaT``, which overflows narrow dtypes (Int8/16/32, UInt*) on newer numpy.
    The ``Float64`` view yields the same value (pandas takes the median in
    float64 anyway); every other dtype goes through ``s.median()`` unchanged.
    """
    if isinstance(s.array, pd.arrays.IntegerArray):
        return s.astype("Float64").median()
    return s.median()


#: float64 represents every integer up to this magnitude exactly.
FLOAT64_EXACT_INT = 2**53


def exceeds_float64_exact(s: pd.Series) -> bool:
    """True for a nullable integer column holding a value float64 cannot represent.

    Casting such a column to float64 silently changes *present* values
    (``2**53 + 1`` becomes ``2**53``), so fill paths must keep its integer dtype.
    """
    if not isinstance(s.array, pd.arrays.IntegerArray):
        return False
    present = s.dropna()
    if present.empty:
        return False
    return int(present.max()) > FLOAT64_EXACT_INT or int(present.min()) < -FLOAT64_EXACT_INT


def exact_int_stat(s: pd.Series, strategy: str) -> int:
    """Mean or median of an integer column in exact integer arithmetic.

    Rounded half-to-even to the nearest integer so the result fits the column's
    dtype. Only used for columns where :func:`exceeds_float64_exact` holds.
    """
    values = sorted(int(v) for v in s.dropna())
    n = len(values)
    if strategy == "mean":
        return round(Fraction(sum(values), n))
    mid = n // 2
    if n % 2:
        return values[mid]
    return round(Fraction(values[mid - 1] + values[mid], 2))


def fill_na_exact(s: pd.Series, value: Any) -> tuple[pd.Series, str]:
    """``s.fillna(value)`` that never silently corrupts large nullable integers.

    Returns the filled series and a note for the report. A column holding values
    beyond 2**53 keeps its integer dtype and receives an integer fill value.
    Any other numeric column that cannot hold a fractional *value* is cast to
    float64, which is exact for it. Raises ``TypeError``/``ValueError`` when
    *value* cannot be stored.
    """
    if exceeds_float64_exact(s):
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"cannot fill {s.dtype} with {value!r}")
            value = int(round(value))
        return s.fillna(value), f", kept {s.dtype} so values beyond 2**53 stay exact"
    try:
        return s.fillna(value), ""
    except (TypeError, ValueError):
        if pd.api.types.is_numeric_dtype(s) and isinstance(value, float):
            # e.g. a fractional median into an integer column
            return s.astype("float64").fillna(value), ", column cast to float64"
        raise


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
    return (
        pd.api.types.is_object_dtype(dtype)
        or isinstance(dtype, pd.StringDtype)
        or is_arrow_string_dtype(dtype)
    )


def is_text_dtype(dtype: object) -> bool:
    """True when *dtype* holds text, identically on pandas 1.5 and 2.x.

    Covers ``object``, ``StringDtype`` (python/pyarrow), ``pd.ArrowDtype`` of
    ``string``/``large_string``/``string_view`` or a dictionary of those, and a
    ``CategoricalDtype`` whose categories are text. ``is_string_dtype`` is not
    used because it answers differently for categoricals across pandas lines.
    """
    if isinstance(dtype, pd.CategoricalDtype):
        return is_text_dtype(dtype.categories.dtype)
    if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.StringDtype):
        return True
    arrow_dtype_cls = getattr(pd, "ArrowDtype", None)
    if arrow_dtype_cls is None or not isinstance(dtype, arrow_dtype_cls):
        return False
    import pyarrow as pa  # noqa: PLC0415 - an ArrowDtype implies pyarrow is installed

    arrow_type = getattr(dtype, "pyarrow_dtype", None)
    if arrow_type is None:
        return False
    if pa.types.is_dictionary(arrow_type):
        arrow_type = arrow_type.value_type
    is_string_view = getattr(pa.types, "is_string_view", None)
    return bool(
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or (is_string_view is not None and is_string_view(arrow_type))
    )


def is_arrow_string_dtype(dtype: object) -> bool:
    """True for a ``pd.ArrowDtype`` holding strings (pandas >= 2 only).

    ``pd.ArrowDtype(pa.string())`` carries the same text as ``string[pyarrow]``
    but is a different dtype class, so it needs its own check. pandas 1.5's
    experimental ``ArrowDtype`` is left alone.
    """
    arrow_dtype_cls = getattr(pd, "ArrowDtype", None)
    if PANDAS_MAJOR < 2 or arrow_dtype_cls is None or not isinstance(dtype, arrow_dtype_cls):
        return False
    import pyarrow as pa  # noqa: PLC0415 - an ArrowDtype implies pyarrow is installed

    arrow_type = getattr(dtype, "pyarrow_dtype", None)
    is_string_view = getattr(pa.types, "is_string_view", None)
    return bool(
        pa.types.is_string(arrow_type)
        or pa.types.is_large_string(arrow_type)
        or (is_string_view is not None and is_string_view(arrow_type))
    )


def as_string_view(s: pd.Series) -> pd.Series:
    """``string[pyarrow]`` copy of an Arrow-string column; any other column as-is.

    Type inference parses text into numbers/dates and then does arithmetic on the
    result; Arrow-backed results do not implement all of it (e.g. ``%``), while
    the ``string[pyarrow]`` path produces regular pandas dtypes.
    """
    if is_arrow_string_dtype(s.dtype):
        return s.astype(pd.StringDtype("pyarrow"))
    return s


#: Leading characters Excel/Sheets/LibreOffice interpret as a formula
#: (OWASP CSV-injection guidance).
_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _formula_guard(value: object) -> object:
    # lstrip: spreadsheets ignore leading whitespace when deciding whether a
    # cell is a formula, so " =1+1" is just as live as "=1+1".
    if isinstance(value, str) and value.lstrip(" \t").startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


def _guard_axis(axis: pd.Index) -> pd.Index:
    """Formula-guard every label (every level of a MultiIndex) and name of *axis*.

    The axis object is returned unchanged when nothing needs guarding, so
    numeric, datetime and categorical axes keep their type.
    """
    names = [_formula_guard(n) for n in axis.names]
    names_changed = any(g is not n for g, n in zip(names, axis.names))
    if isinstance(axis, pd.MultiIndex):
        if any(_formula_guard(v) is not v for level in axis.levels for v in level):
            return pd.MultiIndex.from_tuples(
                [tuple(_formula_guard(v) for v in label) for label in axis], names=names
            )
    elif _is_stringlike_dtype(axis.dtype) or isinstance(axis.dtype, pd.CategoricalDtype):
        changed = False
        labels: list[object] = []
        for value in axis:
            guarded = _formula_guard(value)
            changed = changed or guarded is not value
            labels.append(guarded)
        if changed:
            return pd.Index(labels, dtype=object, name=names[0], tupleize_cols=False)
    return axis.set_names(names) if names_changed else axis


def sanitize_csv_formulas(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of *df* safe to open in a spreadsheet: string cells, column
    labels (every level of a multi-row header), index labels (every level),
    and column/index names starting with ``= + - @ <tab> <cr>`` — including
    after leading whitespace — are prefixed with ``'`` so they render as text
    instead of executing as formulas. Non-string cells and labels (including
    negative numbers) are untouched. Header aliases a caller passes to the
    writer (``to_csv(header=[...])``) are not part of *df* and are not guarded.
    """
    out = df.copy()
    for i, dtype in enumerate(out.dtypes):
        if _is_stringlike_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            column = out.iloc[:, i]
            guarded = column.astype(object).map(_formula_guard)
            if not guarded.equals(column.astype(object)):
                out.isetitem(i, guarded)
    out.columns = _guard_axis(out.columns)
    out.index = _guard_axis(out.index)
    return out


_SENSITIVE_TOKEN_KEY: bytes | None = None
_SENSITIVE_TOKEN_KEY_LOCK = threading.Lock()


def _sensitive_token_key() -> bytes:
    """The per-process secret for :func:`mask_sensitive_value`, made on first use.

    It comes from :func:`secrets.token_bytes` and is never written anywhere.
    The lock makes sure concurrent first calls share one key, so one report
    never mixes tokens made under two keys.
    """
    global _SENSITIVE_TOKEN_KEY  # noqa: PLW0603 - lazy process-wide secret
    key = _SENSITIVE_TOKEN_KEY
    if key is None:
        with _SENSITIVE_TOKEN_KEY_LOCK:
            if _SENSITIVE_TOKEN_KEY is None:
                _SENSITIVE_TOKEN_KEY = secrets.token_bytes(32)
            key = _SENSITIVE_TOKEN_KEY
    return key


def mask_sensitive_value(value: object) -> str:
    """Stand-in token for a sensitive value in report text: ``[SENSITIVE:xxxxxxxx]``.

    The 8 hex characters are a truncated HMAC-SHA256 of ``repr(value)``. The key
    is a random secret made once per process. Within a process the same value
    always gives the same token, so mentions in one report can still be matched
    up. Tokens change between processes and never map back to the value.

    Why the key matters: an unkeyed digest of a low-entropy value (an SSN, a
    phone number, a date of birth, a small category) can be reversed by hashing
    a list of guesses. Without the per-process key, a guess list cannot be
    checked against the tokens. No caller needs tokens to match across runs,
    so there is no stable-key option and no constant fallback.
    """
    digest = hmac.new(
        _sensitive_token_key(), repr(value).encode("utf-8"), hashlib.sha256
    ).hexdigest()[:8]
    return f"[SENSITIVE:{digest}]"


#: Every character ``str.isspace`` accepts, i.e. what ``str.strip()`` removes.
#: Native engines strip exactly this set so they match the pandas reference
#: (RE2's ``\s`` is ASCII-only; Rust's whitespace excludes ``\x1c``-``\x1f``).
PY_WHITESPACE = (
    "\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a"
    "\u2028\u2029\u202f\u205f\u3000"
)
