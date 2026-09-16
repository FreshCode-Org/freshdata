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
from .._util import _is_stringlike_dtype
from ..config import CleanConfig
from ..report import CleanReport

#: infer_dtype kinds that can contain str values worth repairing.
_TEXTUAL_KINDS = ("string", "mixed", "mixed-integer")


def active_sentinels(config: CleanConfig) -> frozenset[str]:
    return frozenset(DEFAULT_SENTINELS | set(config.extra_sentinels))


def _str_positions(s: pd.Series, kind: str) -> pd.Series:
    """Boolean mask of the positions of *s* that hold a real ``str``.

    These are the only cells text repair ever rewrites, so the mask drives both
    the repair and the count of repaired cells.
    """
    if kind == "string":
        # infer_dtype("string") guarantees every non-missing value is a str.
        return s.notna()
    return s.map(lambda v: isinstance(v, str)).astype(bool)


def _n_repaired(new: pd.Series, old: pd.Series, mask: pd.Series) -> int:
    """Count the masked cells *new* actually changed.

    Comparing the whole column instead (``new.ne(old)``) aborts the pipeline on
    ordinary data: the flex comparison hands object columns straight to NumPy,
    which calls ``bool()`` on ``pd.NA != pd.NA`` and raises "boolean value of NA
    is ambiguous". Any non-scalar cell (a list or dict from JSON) keeps the
    column object-dtype, so that path is easy to hit. Only ``str`` cells can
    differ here — every other cell is returned untouched by construction — so
    restrict the comparison to them and never look at a cell we did not repair.
    """
    if not mask.any():
        return 0
    positions = mask.to_numpy(dtype=bool)
    left = new.to_numpy(dtype=object)[positions]
    right = old.to_numpy(dtype=object)[positions]
    return int((left != right).sum())


def _strip_series(s: pd.Series, kind: str, mask: pd.Series) -> pd.Series:
    """Whitespace-strip string values of *s*, preserving non-string values."""
    if kind == "string":
        return s.str.strip()
    # Mixed column: operate only on positions that actually hold a str.
    if not mask.any():
        return s
    out = s.copy()
    out[mask] = s[mask].str.strip()
    return out


def _case_series(s: pd.Series, kind: str, string_case: str, mask: pd.Series) -> pd.Series:
    """Case-normalize string values of *s*, preserving non-string values."""
    if kind == "string":
        return s.str.lower() if string_case == "lower" else s.str.upper()
    if not mask.any():
        return s
    out = s.copy()
    out[mask] = s[mask].str.lower() if string_case == "lower" else s[mask].str.upper()
    return out


def normalize_text(
    s: pd.Series, config: CleanConfig, sentinels: frozenset[str]
) -> tuple[pd.Series, int, int, int]:
    """Strip whitespace and null out sentinels in one text-capable column.

    Returns ``(normalized, n_stripped, n_sentinels)``. Returns the input
    series unchanged (``normalized is s``) when there is nothing to do.
    """
    kind = infer_dtype(s, skipna=True)
    if kind not in _TEXTUAL_KINDS:
        return s, 0, 0, 0

    n_stripped = 0
    if config.strip_whitespace:
        mask = _str_positions(s, kind)
        stripped = _strip_series(s, kind, mask)
        n_stripped = _n_repaired(stripped, s, mask)
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

    n_case = 0
    if config.string_case is not None:
        before = s
        # Recomputed: the sentinel pass above may have nulled some str cells.
        mask = _str_positions(before, kind)
        cased = _case_series(before, kind, config.string_case, mask)
        n_case = _n_repaired(cased, before, mask)
        if n_case:
            s = cased

    return s, n_stripped, n_sentinels, n_case


def is_text_categorical_dtype(dtype: object) -> bool:
    """True for a categorical dtype whose categories hold text worth repairing."""
    return (
        isinstance(dtype, pd.CategoricalDtype)
        and infer_dtype(dtype.categories, skipna=True) in _TEXTUAL_KINDS
    )


def normalize_categorical(
    s: pd.Series, config: CleanConfig, sentinels: frozenset[str]
) -> tuple[pd.Series, int, int, int]:
    """:func:`normalize_text` for a text categorical, keeping the categorical dtype.

    Values are repaired exactly as the equivalent object column would be (so
    counts match), then rebuilt as a categorical with the same ``ordered`` flag
    whose categories are the repaired originals: ``" a "`` and ``"a"`` merge,
    and sentinel categories disappear.
    """
    normalized, n_stripped, n_sentinels, n_case = normalize_text(
        s.astype(object), config, sentinels
    )
    if not (n_stripped or n_sentinels or n_case):
        return s, 0, 0, 0
    categories, *_ = normalize_text(pd.Series(s.cat.categories, dtype=object), config, sentinels)
    rebuilt = pd.Categorical(
        normalized, categories=pd.unique(categories.dropna()), ordered=s.cat.ordered
    )
    return pd.Series(rebuilt, index=s.index, name=s.name), n_stripped, n_sentinels, n_case


def _text_columns(df: pd.DataFrame) -> list:
    """Object/string columns plus text categoricals, in frame order."""
    return [
        col
        for col, dtype in zip(df.columns, df.dtypes)
        if _is_stringlike_dtype(dtype) or is_text_categorical_dtype(dtype)
    ]


def declared_vocabulary(config: CleanConfig, column: object) -> frozenset[str]:
    """Casefolded ``allowed_values`` the caller declared for *column*.

    An explicit vocabulary outranks a generic null marker: ``"NA"`` is Namibia
    in an ISO-3166 column and ``"None"`` is a real brand. ``fieldcheck`` has
    applied that rule since the TestAllowedValuesBeatNullMarkers regression;
    this is the same rule for the cleaning path.
    """
    context = getattr(config, "semantic_context", None)
    if not isinstance(context, dict):
        return frozenset()
    columns = context.get("columns")
    if not isinstance(columns, dict):
        return frozenset()
    hints = columns.get(str(column))
    if not isinstance(hints, dict):
        return frozenset()
    allowed = hints.get("allowed_values")
    if not allowed or isinstance(allowed, (str, bytes)):
        return frozenset()
    try:
        return frozenset(str(v).casefold().strip() for v in allowed)
    except TypeError:
        return frozenset()


def clean_strings(df: pd.DataFrame, config: CleanConfig, report: CleanReport) -> pd.DataFrame:
    """Apply whitespace stripping and sentinel→missing to text-capable columns.

    Categorical columns with text categories are repaired too; they keep their
    categorical dtype (see :func:`normalize_categorical`).
    """
    if not (
        config.strip_whitespace or config.normalize_sentinels or config.string_case is not None
    ):
        return df
    sentinels = active_sentinels(config)
    from ..guard import hard_protected_columns  # noqa: PLC0415 — cycle-safe lazy import

    protected = hard_protected_columns(config, df.columns)
    for col in _text_columns(df):
        if str(col) in protected:
            continue  # context-protected columns must stay byte-identical
        s = df[col]
        # A token the caller declared as a permitted value is a value, not a
        # missing marker, so it is removed from this column's sentinel set.
        column_sentinels = sentinels - declared_vocabulary(config, col)
        normalize = (
            normalize_categorical if isinstance(s.dtype, pd.CategoricalDtype) else normalize_text
        )
        normalized, n_stripped, n_sentinels, n_case = normalize(s, config, column_sentinels)
        if n_stripped:
            report.add(
                "strip_whitespace",
                "trimmed surrounding whitespace",
                column=str(col),
                count=n_stripped,
            )
        if n_sentinels:
            report.add(
                "normalize_sentinels",
                'replaced sentinel strings ("N/A", "-", "", …) with missing',
                column=str(col),
                count=n_sentinels,
            )
        if n_case:
            report.add(
                "normalize_case",
                f"converted text to {config.string_case}",
                column=str(col),
                count=n_case,
            )
        if n_stripped or n_sentinels or n_case:
            df[col] = normalized
    return df
