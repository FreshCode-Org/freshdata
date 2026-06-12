"""Outlier detection helpers and the simple opt-in handling step.

Detection is shared by three callers: the legacy opt-in step here
(``outliers="clip" | "flag"``), the decision engine
(:mod:`freshdata.engine.outliers`), and :func:`freshdata.profile`.

Methods: Tukey fences (``iqr``, factor 1.5), mean ± k standard deviations
(``zscore``, factor 3.0), or ``auto`` — z-score for approximately normal
columns (|skewness| < 0.5), IQR otherwise.
"""

from __future__ import annotations

import math

import pandas as pd
from pandas.api.types import is_bool_dtype, is_integer_dtype, is_numeric_dtype

from ..config import _DEFAULT_FACTOR, CleanConfig
from ..report import CleanReport

#: |skewness| below which a distribution counts as approximately normal.
_NORMALISH_SKEW = 0.5


def safe_skew(s: pd.Series) -> float | None:
    """Sample skewness of a numeric series, or None when undefined."""
    nonnull = s.dropna()
    if len(nonnull) < 3:
        return None
    try:
        value = float(nonnull.skew())
    except (TypeError, ValueError):
        return None
    return None if pd.isna(value) else value


def resolve_method(s: pd.Series, config: CleanConfig) -> str:
    """The concrete detection method ("iqr" or "zscore") for one column.

    ``"auto"`` picks z-score for approximately normal distributions and IQR
    for skewed ones. ``"isolation_forest"`` resolves the same way here; its
    model-based path lives in the engine and falls back to this for previews.
    """
    method = config.outlier_method
    if method in ("iqr", "zscore"):
        return method
    nonnull = s.dropna()
    # Measure shape on the trimmed bulk: a single extreme spike must not make
    # an otherwise-normal column look "skewed" to the very detector hunting it.
    inner = detection_bounds(nonnull, "iqr", 3.0)
    if inner is not None:
        trimmed = nonnull[(nonnull >= inner[0]) & (nonnull <= inner[1])]
        if len(trimmed) >= 3:
            nonnull = trimmed
    skew = safe_skew(nonnull)
    if skew is not None and abs(skew) < _NORMALISH_SKEW:
        return "zscore"
    return "iqr"


def factor_for(config: CleanConfig, method: str) -> float:
    """The detection factor in effect for *method* (user override wins)."""
    if config.outlier_factor is not None:
        return config.outlier_factor
    return _DEFAULT_FACTOR[method]


def detection_bounds(
    s: pd.Series, method: str, factor: float
) -> tuple[float, float] | None:
    """(lower, upper) fences for *s*, or None when undefined (constant data)."""
    if method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        spread = q3 - q1
        if pd.isna(spread) or spread == 0:
            return None
        return float(q1 - factor * spread), float(q3 + factor * spread)
    mean, std = s.mean(), s.std()
    if pd.isna(std) or std == 0:
        return None
    return float(mean - factor * std), float(mean + factor * std)


def _bounds(s: pd.Series, config: CleanConfig) -> tuple[float, float] | None:
    """Config-resolved bounds (compatibility wrapper used by profiling)."""
    method = resolve_method(s, config)
    return detection_bounds(s, method, factor_for(config, method))


def integer_safe_bounds(s: pd.Series, lo: float, hi: float) -> tuple[float, float]:
    """Widen fences to integers so integer columns stay integer after clipping."""
    if is_integer_dtype(s):
        return math.floor(lo), math.ceil(hi)
    return lo, hi


def unique_flag_name(df: pd.DataFrame, base: str) -> str:
    """A column name based on *base* that does not collide with existing ones."""
    name, k = base, 1
    while name in df.columns:
        k += 1
        name = f"{base}_{k}"
    return name


def handle_outliers(df: pd.DataFrame, config: CleanConfig,
                    report: CleanReport) -> pd.DataFrame:
    """Opt-in simple handling: clip or flag outliers in every numeric column.

    This is the explicit override path (``outliers="clip" | "flag"``); the
    context-aware default behavior lives in the decision engine.
    """
    if config.outliers is None or df.empty:
        return df
    numeric_cols = [c for c in df.columns
                    if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    for col in numeric_cols:
        s = df[col]
        method = resolve_method(s, config)
        factor = factor_for(config, method)
        bounds = detection_bounds(s, method, factor)
        if bounds is None:
            continue
        lo, hi = integer_safe_bounds(s, *bounds)
        mask = (s < lo) | (s > hi)
        n = int(mask.sum())
        if n == 0:
            continue
        label = f"{method}, factor {factor:g}"
        if config.outliers == "clip":
            df[col] = s.clip(lo, hi)
            report.add("outliers", f"clipped {n} outlier(s) to [{lo:g}, {hi:g}] ({label})",
                       column=str(col), count=n)
        else:
            flag = unique_flag_name(df, f"{col}_outlier")
            df[flag] = mask.fillna(False).astype(bool)
            report.add("outliers", f"flagged {n} outlier(s) in new column {flag!r} ({label})",
                       column=str(col), count=n)
        report.outliers_handled += n
    return df
