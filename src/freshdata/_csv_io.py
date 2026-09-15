"""CSV read helpers shared by the CLI and :func:`freshdata.clean_csv`. Internal.

``pandas.read_csv`` infers ``"02134"`` as the integer ``2134`` before any cleaning
step runs, so ``CleanConfig.preserve_leading_zeros`` never gets a chance to keep
the padding. :func:`leading_zero_dtypes` pre-scans a bounded sample of the file as
text and returns a ``dtype`` mapping that reads only the zero-padded numeric
columns as ``str``; every other column keeps pandas' normal type inference.
"""

from __future__ import annotations

import os
from collections.abc import Hashable, Mapping
from typing import Any

import pandas as pd

from ._numeric import safe_to_numeric
from .steps.dtypes import _has_leading_zero_ids

#: Rows read by the pre-scan. Zero padding that first appears after this many rows
#: is not detected (the column is then read with pandas' usual inference).
LEADING_ZERO_SCAN_ROWS = 10_000

# Options that already decide column types, or that turn the read into an iterator.
_TYPE_OPTIONS = ("dtype", "converters")
_ITERATOR_OPTIONS = ("chunksize", "iterator")


def leading_zero_dtypes(
    path: object,
    *,
    read_csv_kwargs: Mapping[str, Any] | None = None,
    nrows: int = LEADING_ZERO_SCAN_ROWS,
) -> dict[Hashable, type[str]]:
    """Return ``{column: str}`` for numeric-looking CSV columns with zero-padded values.

    The first *nrows* rows are read with ``dtype=str``. A column is included when
    all of its non-missing sample values parse as numbers (so pandas would infer a
    numeric dtype) and at least one of them is zero-padded (``"007"``, ``"02134"``),
    as judged by the same detector the dtype-inference step uses.

    Returns ``{}`` (read with plain inference) when *read_csv_kwargs* already sets
    ``dtype`` or ``converters``, when *path* is not a filesystem path (a buffer
    cannot be read twice), or when the sample cannot be read — the real read then
    reports that error itself.
    """
    kwargs = dict(read_csv_kwargs or {})
    if any(kwargs.get(key) is not None for key in _TYPE_OPTIONS):
        return {}
    if not isinstance(path, (str, os.PathLike)):
        return {}
    for key in _ITERATOR_OPTIONS:
        kwargs.pop(key, None)
    limit = kwargs.get("nrows")
    kwargs["nrows"] = nrows if limit is None else min(int(limit), nrows)
    try:
        sample = pd.read_csv(path, dtype=str, **kwargs)
    except (OSError, ValueError):
        return {}

    padded: dict[Hashable, type[str]] = {}
    for position, column in enumerate(sample.columns):
        values = sample.iloc[:, position].dropna()
        if values.empty or not _has_leading_zero_ids(values):
            continue
        if safe_to_numeric(values, errors="coerce").notna().all():
            padded[column] = str
    return padded
