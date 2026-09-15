"""Crash-safe :func:`pandas.to_numeric`.

pandas < 3 reads the exponent digits of a numeric string into a C int with no
overflow check, and it does so *before* rejecting trailing text
(pandas-dev/pandas#62617, #63089, #63167; fixed in pandas 3.0 by
pandas-dev/pandas#62741). A cell that merely starts with such a token -- e.g.
the hash-like value ``"81e3104049863b72"`` -- can segfault ``to_numeric`` and
take the whole process down, whatever ``errors=`` says.

:func:`safe_to_numeric` keeps those cells away from the parser. Every
``to_numeric`` call on data that may hold text must go through it
(``tests/test_numeric.py`` enforces this for ``src/freshdata``).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray

# The leading scientific-notation token of a cell, as pandas' C float parser
# (precise_xstrtod) reads it. Match the prefix, not the whole cell: trailing
# text is only rejected after the exponent has been accumulated.
_SCIENTIFIC_PREFIX = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE]([+-]?\d+)")
# precise_xstrtod accumulates the exponent digits in a C int (n = n * 10 + d)
# and adds n to a mantissa adjustment, with no overflow check. Leading zeros
# keep n at 0, so only significant digits matter: nine (below 10**9) cannot
# overflow even after the adjustment, ten might. Every shorter exponent --
# including subnormal and out-of-range values such as "4.9e-324" or "1e400" --
# is left to pandas, which parses or rejects it safely.
_MIN_UNSAFE_EXPONENT = 10**9
# Column-level screen: an unsafe cell contains an exponent marker followed by
# at least ten digits. A false positive (e.g. leading zeros) merely routes the
# column to the exact per-cell check.
_RISKY_EXPONENT = re.compile(r"[eE][+-]?\d{10,}")
# dtype kinds pandas converts without its string parser: bool, integer,
# unsigned, float, complex, datetime and timedelta.
_PARSER_FREE_KINDS = frozenset("biufcmM")
# Stand-in for masked cells under errors="ignore": text pandas cannot parse.
_UNPARSEABLE = "x"


def _has_unsafe_scientific_exponent(value: object) -> bool:
    """True when *value* starts with scientific notation whose exponent can
    overflow the C int in pandas' parser (ten or more significant digits)."""
    if isinstance(value, bytes):  # pandas parses bytes cells with the same C code
        value = value.decode("latin-1")
    if not isinstance(value, str):
        return False
    match = _SCIENTIFIC_PREFIX.match(value.lstrip())
    if match is None:
        return False
    try:
        return abs(int(match.group(1))) >= _MIN_UNSAFE_EXPONENT
    except ValueError:  # e.g. more digits than int() accepts from a string
        return True


def _text_blob(cells: np.ndarray) -> str:
    """All text cells joined into one string, using C-level joins when possible."""
    try:
        return "\x1f".join(cells)
    except TypeError:  # missing values or non-text cells
        pass
    try:
        return "\x1f".join(cells[pd.notna(cells)])
    except TypeError:  # non-text objects (numbers, bytes, lists)
        pass
    return "\x1f".join(
        v.decode("latin-1") if isinstance(v, bytes) else v
        for v in cells
        if isinstance(v, (str, bytes))
    )


def _unsafe_cells(cells: np.ndarray) -> np.ndarray | None:
    """Positional mask of unsafe cells, or ``None`` when there are none."""
    blob = _text_blob(cells)
    if ("e" not in blob and "E" not in blob) or _RISKY_EXPONENT.search(blob) is None:
        return None
    unsafe = np.fromiter(
        map(_has_unsafe_scientific_exponent, cells), dtype=bool, count=len(cells)
    )
    return unsafe if unsafe.any() else None


def _parsed_cells(values: Any) -> np.ndarray | None:
    """The 1-D cells pandas would run through its string parser, else ``None``."""
    if isinstance(values, (pd.Series, pd.Index, np.ndarray, ExtensionArray)):
        if values.ndim != 1 or values.dtype.kind in _PARSER_FREE_KINDS:
            return None
        if isinstance(values.dtype, pd.CategoricalDtype) and _unsafe_cells(
            np.asarray(values.dtype.categories, dtype=object)
        ) is None:
            return None
        return values if isinstance(values, np.ndarray) else values.to_numpy()
    if isinstance(values, (list, tuple)):
        cells = np.array(values, dtype=object)
        return cells if cells.ndim == 1 else None
    if isinstance(values, (str, bytes)):
        return np.array([values], dtype=object)
    return None


def _categories(values: Any) -> Any:
    """The ``.cat``-style accessor of a categorical Series or Categorical."""
    return values.cat if isinstance(values, pd.Series) else values


def _is_categorical(values: Any) -> bool:
    return isinstance(getattr(values, "dtype", None), pd.CategoricalDtype)


def _substitute(values: Any, unsafe: np.ndarray, fill: object) -> Any:
    """A copy of *values* with the unsafe cells replaced by *fill*
    (``None`` means missing), keeping the container pandas dispatches on."""
    if isinstance(values, pd.Index):
        return values.where(~unsafe) if fill is None else values.where(~unsafe, fill)
    if isinstance(values, (pd.Series, ExtensionArray)) and fill is None:
        return (
            values.mask(unsafe)
            if isinstance(values, pd.Series)
            else pd.Series(values, copy=False).mask(unsafe).array
        )
    positions = np.flatnonzero(unsafe)
    if isinstance(values, (list, tuple)):
        out = np.array(values, dtype=object)
    elif isinstance(values, np.ndarray):
        # str/bytes arrays cannot hold None; pandas parses them as object anyway.
        out = values.astype(object) if fill is None else values.copy()
    elif isinstance(values, (pd.Series, ExtensionArray)):
        if _is_categorical(values) and fill not in values.dtype.categories:
            values = _categories(values).add_categories([fill])
        out = values.copy()
    else:
        return fill  # scalar
    if isinstance(out, pd.Series):
        out.iloc[positions] = fill
    else:
        out[positions] = fill
    return out


def _head(values: Any, stop: int) -> Any:
    """The cells before position *stop*, in the same container."""
    if isinstance(values, pd.Series):
        return values.iloc[:stop]
    if isinstance(values, (str, bytes)):
        return np.array([], dtype=object)
    return values[:stop]


def _restore(result: Any, values: Any, cells: np.ndarray, unsafe: np.ndarray) -> Any:
    """Put the original cells back into an ``errors="ignore"`` result."""
    if isinstance(values, (str, bytes)):
        return values
    positions = np.flatnonzero(unsafe)
    if isinstance(result, pd.Index):
        arr = result.to_numpy(dtype=object, copy=True)
        arr[positions] = cells[positions]
        return pd.Index(arr, dtype=result.dtype, name=result.name)
    out = result.copy()
    if isinstance(out, pd.Series):
        out.iloc[positions] = cells[positions]
    else:
        out[positions] = cells[positions]
    if (
        _is_categorical(out)
        and _is_categorical(values)
        and _UNPARSEABLE not in values.dtype.categories
    ):
        out = _categories(out).remove_categories([_UNPARSEABLE])
    return out


def safe_to_numeric(values: Any, **kwargs: Any) -> Any:
    """:func:`pandas.to_numeric` that cannot crash on exponent-overflow text.

    Accepts everything ``pd.to_numeric`` does (scalar, list, tuple, 1-D array,
    ``Index`` or ``Series``) and forwards every keyword unchanged
    (``errors=``, ``downcast=``, ``dtype_backend=``). Input with no unsafe cell
    -- including every numeric, boolean or datetime dtype -- goes to pandas
    untouched, so the result is exactly pandas' result (index, name, dtype).

    A cell that starts with scientific notation whose exponent has ten or more
    significant digits (enough to overflow pandas' C int) is treated as
    unparseable text: it becomes missing with ``errors="coerce"``, raises
    pandas' ``Unable to parse string`` ``ValueError`` with ``errors="raise"``,
    and is returned as-is with ``errors="ignore"``. pandas rejects almost all
    such cells too; the exception is a whole-cell negative exponent small
    enough not to overflow, which pandas would underflow to 0.0. Shorter
    exponents, including subnormal and out-of-range ones, are left to pandas.

    The check is cheap: text columns are screened as one joined string, and
    only a column that contains an exponent marker followed by ten digits
    pays for the per-cell check.
    """
    cells = _parsed_cells(values)
    unsafe = None if cells is None else _unsafe_cells(cells)
    if cells is None or unsafe is None:
        return pd.to_numeric(values, **kwargs)
    errors = kwargs.get("errors", "raise")
    if errors == "coerce":
        return pd.to_numeric(_substitute(values, unsafe, None), **kwargs)
    if errors == "raise":
        first = int(np.argmax(unsafe))
        # An earlier unparseable cell raises first, exactly as pandas would.
        pd.to_numeric(_head(values, first), **kwargs)
        cell = cells[first]
        text = cell.decode("latin-1") if isinstance(cell, bytes) else cell
        raise ValueError(f'Unable to parse string "{text}" at position {first}')
    # errors="ignore" (pandas itself rejects any other value).
    result = pd.to_numeric(_substitute(values, unsafe, _UNPARSEABLE), **kwargs)
    return _restore(result, values, cells, unsafe)


__all__ = ["safe_to_numeric"]
