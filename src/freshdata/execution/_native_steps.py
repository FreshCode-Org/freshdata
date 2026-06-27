"""Backend-agnostic math and labels for the native impute/outlier steps.

The Polars, DuckDB, and Spark backends compute column statistics in their own
dialect, then share the *decision* logic here so the three stay consistent with
each other and as close to the pandas reference (``steps/impute.py`` and
``steps/outliers.py``) as the backend's statistics allow. Where a backend's
quantile/percentile interpolation differs from pandas, the count of affected
rows can differ; callers record that via
:meth:`~freshdata.report.CleanReport.record_backend_difference`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..config import _DEFAULT_FACTOR

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import CleanConfig

#: Outlier methods native backends compute directly.
NATIVE_OUTLIER_METHODS = ("iqr", "zscore")


def resolve_impute_strategy(strategy: str, *, is_numeric: bool) -> str:
    """Resolve ``impute="auto"`` to ``"median"`` (numeric) or ``"mode"``.

    Mirrors ``steps/impute.py._fill_value``.
    """
    if strategy == "auto":
        return "median" if is_numeric else "mode"
    return strategy


def impute_defined_for(strategy: str, *, is_numeric: bool) -> bool:
    """``mean``/``median`` are only defined for numeric columns."""
    if strategy in ("mean", "median"):
        return is_numeric
    return True


def native_outlier_method(config: CleanConfig) -> str:
    """The concrete native detection method (``"iqr"`` or ``"zscore"``)."""
    method = config.outlier_method
    return method if method in NATIVE_OUTLIER_METHODS else "iqr"


def native_outlier_factor(config: CleanConfig, method: str) -> float:
    """Detection factor in effect for *method* (user override wins)."""
    if config.outlier_factor is not None:
        return config.outlier_factor
    return _DEFAULT_FACTOR[method]


def iqr_bounds(q1: float, q3: float, factor: float) -> tuple[float, float] | None:
    """Tukey fences from quartiles, or ``None`` for constant data."""
    spread = q3 - q1
    if spread is None or _isnan(spread) or spread == 0:
        return None
    return q1 - factor * spread, q3 + factor * spread


def zscore_bounds(mean: float, std: float, factor: float) -> tuple[float, float] | None:
    """Mean ± k·std fences, or ``None`` for constant data."""
    if std is None or _isnan(std) or std == 0:
        return None
    return mean - factor * std, mean + factor * std


def integer_safe_bounds(lo: float, hi: float, *, is_integer: bool) -> tuple[float, float]:
    """Widen fences to whole numbers so integer columns stay integer."""
    if is_integer:
        return float(math.floor(lo)), float(math.ceil(hi))
    return lo, hi


def outlier_label(method: str, factor: float) -> str:
    """Human-readable detector label, matching the pandas step."""
    return f"{method}, factor {factor:g}"


def _isnan(value: float) -> bool:
    try:
        return math.isnan(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False
