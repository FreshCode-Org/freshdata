"""Column profiling for the decision engine: roles, stats, missingness context.

A column's *role* decides which rules may touch it:

- ``id`` — identifier-like; never imputed, outliers ignored.
- ``target`` — label/target; never modified.
- ``datetime`` — time-aware logic (ordered fills only).
- ``text`` — free text; never forced into numeric/sentinel imputation.
- ``boolean`` / ``numeric`` / ``categorical`` — standard rules apply.

Roles come from names (``customer_id``, ``target``) *and* data shape
(all-unique key columns, long high-cardinality strings), so misnamed columns
are still usually classified correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .._util import _is_stringlike_dtype
from ..config import CleanConfig
from ..steps.outliers import safe_skew

#: Below this many rows, ratios are too noisy for drops or non-trivial
#: imputation; the engine preserves and warns instead.
MIN_ROWS_FOR_ENGINE = 30

_ID_NAME = re.compile(r"(?:^|[_\s])(?:id|uuid|guid|key)s?$|^(?:id|index|pk)$", re.I)
_TARGET_NAMES = frozenset({"target", "label", "y", "outcome", "class", "response"})
_TARGET_EXACT = frozenset({
    "aqi", "score", "rating", "churn", "default", "conversion", "label",
})
_TARGET_SUFFIX = re.compile(
    r"(?:_bucket|_class|_label|_category|_grade|_tier|_rating|_score)$", re.I
)
#: Domains where extreme values are usually the signal, not noise.
_DOMAIN_SENSITIVE = re.compile(
    r"fraud|anomal|outlier|spike|alert|rare|risk|aqi|pm\d*|pollut|temp|humid|"
    r"pressure|concentration|level|amount|price|revenue|latency",
    re.I,
)

#: Standardized mean difference above which missingness counts as informative.
_INFORMATIVE_SMD = 0.5


@dataclass
class ColumnContext:
    """Everything the rule engine needs to know about one column."""

    name: str
    role: str
    n_rows: int
    n_missing: int
    missing_ratio: float
    nunique: int | None  # None when values are unhashable
    unique_ratio: float
    skew: float | None
    mode_ratio: float | None  # share of the most frequent non-missing value
    time_ordered: bool
    informative_missing: bool
    preserve: bool
    domain_sensitive: bool
    high_cardinality: bool


def _safe_nunique(s: pd.Series) -> int | None:
    try:
        return int(s.nunique(dropna=True))
    except TypeError:
        return None


def _mode_ratio(s: pd.Series) -> float | None:
    nonnull_count = int(s.notna().sum())
    if nonnull_count == 0:
        return None
    try:
        counts = s.value_counts(dropna=True)
    except TypeError:
        return None
    if counts.empty:
        return None
    return float(counts.iloc[0]) / nonnull_count


def _looks_like_text(s: pd.Series, nunique: int | None, non_null: int) -> bool:
    """Free-form text: long values or near-unique multi-word strings."""
    if non_null == 0:
        return False
    sample = s.dropna().head(200)
    strings = [v for v in sample if isinstance(v, str)]
    if not strings:
        # Unhashable payloads (lists, dicts) get text treatment: never imputed.
        return nunique is None
    avg_len = sum(len(v) for v in strings) / len(strings)
    if avg_len > 25:
        return True
    avg_words = sum(v.count(" ") + 1 for v in strings) / len(strings)
    unique_ratio = (nunique / non_null) if nunique else 0.0
    return avg_words >= 3 and unique_ratio > 0.6


def _is_target_name(label: str, config: CleanConfig) -> bool:
    folded = label.casefold()
    if config.target_column and label == config.target_column:
        return True
    if folded in _TARGET_NAMES or folded in _TARGET_EXACT:
        return True
    return bool(_TARGET_SUFFIX.search(label))


def infer_role(name: str, s: pd.Series, config: CleanConfig) -> str:
    """Classify one column as id / target / datetime / boolean / numeric /
    text / categorical, from its name, dtype, and value shape."""
    label = str(name)
    if _is_target_name(label, config):
        return "target"
    if label in config.id_columns or _ID_NAME.search(label):
        return "id"
    if is_datetime64_any_dtype(s):
        return "datetime"
    if is_bool_dtype(s):
        return "boolean"

    nonnull = s.dropna()
    non_null = len(nonnull)
    nunique = _safe_nunique(s)

    if is_numeric_dtype(s):
        # An all-unique integer column of meaningful size is a key in disguise.
        if (nunique is not None and non_null >= 20 and nunique == non_null
                and pd.api.types.is_integer_dtype(s)):
            return "id"
        return "numeric"

    if _is_stringlike_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        if _is_stringlike_dtype(s.dtype):
            # Free text first: all-unique multi-word strings are prose, not keys.
            if _looks_like_text(s, nunique, non_null):
                return "text"
            if nunique is not None and non_null >= 20 and nunique == non_null:
                return "id"
        return "categorical"
    # Mixed/object payloads we cannot reason about: treat as text (hands off).
    return "text"


def missingness_is_informative(df: pd.DataFrame, col: object) -> bool:
    """Heuristic: does *col*'s missingness correlate with other features?

    Compares up to three other numeric columns between rows where *col* is
    missing and rows where it is present; a standardized mean difference of
    ``>= 0.5`` on any of them marks the missingness as informative (worth an
    indicator column, and worth keeping the column even when very sparse).
    """
    s = df[col]
    miss = s.isna()
    n_miss = int(miss.sum())
    if len(df) < MIN_ROWS_FOR_ENGINE or n_miss < 10 or len(df) - n_miss < 10:
        return False
    probes = [
        c for c in df.columns
        if c != col and is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
    ][:3]
    for probe in probes:
        v = df[probe]
        a = v[miss].dropna()
        b = v[~miss].dropna()
        if len(a) < 10 or len(b) < 10:
            continue
        pooled = float(np.sqrt((a.var() + b.var()) / 2.0))
        if not pooled or pd.isna(pooled):
            continue
        if abs(float(a.mean()) - float(b.mean())) / pooled >= _INFORMATIVE_SMD:
            return True
    return False


def numeric_corr_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    """Absolute correlation matrix for numeric columns (computed once per clean)."""
    cols = [
        c for c in df.columns
        if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c]) and df[c].notna().any()
    ]
    if len(cols) < 2:
        return None
    return df[cols].corr().abs()


def _needs_informative_check(n_missing: int, n_rows: int, config: CleanConfig) -> bool:
    if n_missing < 10 or n_rows < MIN_ROWS_FOR_ENGINE:
        return False
    if config.missing_indicators is False:
        return False
    ratio = n_missing / n_rows if n_rows else 0.0
    return ratio > config.missing_threshold_low


def _time_ordered(s: pd.Series, index: pd.Index) -> bool:
    """Is there a usable time order for forward/backward filling?"""
    if isinstance(index, pd.DatetimeIndex) and (
        index.is_monotonic_increasing or index.is_monotonic_decreasing
    ):
        return True
    nonnull = s.dropna()
    if len(nonnull) < 2:
        return False
    return bool(nonnull.is_monotonic_increasing or nonnull.is_monotonic_decreasing)


def build_context(df: pd.DataFrame, col: object, config: CleanConfig) -> ColumnContext:
    """Profile one column of *df* for the rule engine."""
    s = df[col]
    n = len(df)
    n_missing = int(s.isna().sum())
    non_null = n - n_missing
    nunique = _safe_nunique(s)
    role = infer_role(str(col), s, config)
    skew_series = s
    if role in ("numeric", "id") and is_numeric_dtype(s) and n > config.sample_size:
        skew_series = s.sample(n=config.sample_size, random_state=config.random_state)
    informative = False
    if _needs_informative_check(n_missing, n, config):
        informative = missingness_is_informative(df, col)
    return ColumnContext(
        name=str(col),
        role=role,
        n_rows=n,
        n_missing=n_missing,
        missing_ratio=(n_missing / n) if n else 0.0,
        nunique=nunique,
        unique_ratio=(nunique / non_null) if (nunique and non_null) else 0.0,
        skew=safe_skew(skew_series) if role in ("numeric", "id") and is_numeric_dtype(s) else None,
        mode_ratio=_mode_ratio(s),
        time_ordered=_time_ordered(s, df.index) if role == "datetime" else False,
        informative_missing=informative,
        preserve=str(col) in config.preserve_columns,
        domain_sensitive=bool(_DOMAIN_SENSITIVE.search(str(col))),
        high_cardinality=bool(
            nunique is not None and nunique > 50 and non_null and nunique / non_null > 0.5
        ),
    )


def build_contexts(df: pd.DataFrame, config: CleanConfig) -> dict:
    """Profile every column; keyed by column label."""
    return {col: build_context(df, col, config) for col in df.columns}
