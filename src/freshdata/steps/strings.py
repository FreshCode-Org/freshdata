"""Text cell repair: trim whitespace and convert sentinel strings to missing.

Safety property: mixed-type object columns (e.g. numbers and strings in the
same column) are handled element-wise so non-string values are never touched —
a naive ``.str.strip()`` would silently turn every number into NaN.

:func:`normalize_text` is the single implementation, shared by the cleaning
pipeline and by :func:`freshdata.profile`, so the profile's preview always
matches what cleaning actually does.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import infer_dtype

from .._sentinels import DEFAULT_SENTINELS
from .._util import stringlike_columns
from ..config import CleanConfig
from ..report import CleanReport

#: infer_dtype kinds that can contain str values worth repairing.
_TEXTUAL_KINDS = ("string", "mixed", "mixed-integer")


def active_sentinels(config: CleanConfig) -> frozenset[str]:
    return frozenset(DEFAULT_SENTINELS | set(config.extra_sentinels))


def _strip_series(s: pd.Series, kind: str) -> pd.Series:
    """Whitespace-strip string values of *s*, preserving non-string values."""
    if kind == "string":
        return s.str.strip()
    # Mixed column: operate only on positions that actually hold a str.
    mask = s.map(lambda v: isinstance(v, str))
    if not mask.any():
        return s
    out = s.copy()
    out[mask] = s[mask].str.strip()
    return out


def normalize_text(
    s: pd.Series, config: CleanConfig, sentinels: frozenset[str]
) -> tuple[pd.Series, int, int]:
    """Strip whitespace and null out sentinels in one text-capable column.

    Returns ``(normalized, n_stripped, n_sentinels)``. Returns the input
    series unchanged (``normalized is s``) when there is nothing to do.
    """
    kind = infer_dtype(s, skipna=True)
    if kind not in _TEXTUAL_KINDS:
        return s, 0, 0

    n_stripped = 0
    if config.strip_whitespace:
        stripped = _strip_series(s, kind)
        n_stripped = int((stripped.ne(s) & s.notna()).sum())
        if n_stripped:
            s = stripped

    n_sentinels = 0
    if config.normalize_sentinels:
        # .str.casefold() yields NaN for non-string values, which simply
        # fail the isin() membership test — exactly what we want.
        hits = s.str.casefold().isin(sentinels) & s.notna()
        n_sentinels = int(hits.sum())
        if n_sentinels:
            s = s.mask(hits)

    return s, n_stripped, n_sentinels


def clean_strings(df: pd.DataFrame, config: CleanConfig, report: CleanReport) -> pd.DataFrame:
    """Apply whitespace stripping and sentinel→missing to text-capable columns."""
    if not (config.strip_whitespace or config.normalize_sentinels):
        return df
    sentinels = active_sentinels(config)
    from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

    protected = hard_protected_columns(config, df.columns)
    for col in stringlike_columns(df):
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        normalized, n_stripped, n_sentinels = normalize_text(df[col], config, sentinels)
        if n_stripped:
            report.add("strip_whitespace", "trimmed surrounding whitespace",
                       column=str(col), count=n_stripped)
        if n_sentinels:
            report.add("normalize_sentinels",
                       'replaced sentinel strings ("N/A", "-", "", …) with missing',
                       column=str(col), count=n_sentinels)
        if n_stripped or n_sentinels:
            df[col] = normalized
    return df
