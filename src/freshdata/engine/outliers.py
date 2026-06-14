"""Context-aware outlier handling (``strategy="auto"``).

Detection per numeric column:

- ``outlier_method="iqr"`` / ``"zscore"`` — used as given.
- ``"auto"`` — z-score (3σ) for approximately normal columns, Tukey IQR
  fences (1.5×IQR) for skewed ones.
- ``"isolation_forest"`` — model-based detection when scikit-learn is
  installed and the column has at least 100 values; falls back to IQR/z-score
  otherwise (the fallback is noted in the report).

Action policy (``outlier_action``):

- ``"auto"`` (default) — context-aware: flag under ``strategy="balanced"``,
  cap under ``"aggressive"``, and flag heavy-tailed columns (>15% outlying)
  rather than rewriting real data.
- ``"cap"`` — winsorize to the detection fences. Capping beats deletion: row
  context survives, only the extreme magnitudes are tamed.
- ``"remove"`` — drop the offending rows (for clearly erroneous values).
- ``"flag"`` — add a boolean ``<col>_outlier`` column, data untouched.
- ``None`` — detect and report only.

``"cap"`` / ``"remove"`` / ``"flag"`` are *explicit directives*: they are
applied to every eligible numeric column, even heavy-tailed ones (a warning is
raised in that case). Only the protected columns below are left untouched.

Outliers are always *preserved* (with rationale) for ID/target columns, columns
explicitly listed in ``preserve_columns``, and domain-sensitive columns
(fraud/anomaly/risk/rare-event names) where extremes are usually the signal.
"""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import CleanConfig
from ..report import CleanReport
from ..steps.outliers import (
    detection_bounds,
    factor_for,
    integer_safe_bounds,
    resolve_method,
    unique_flag_name,
)
from .context import _DOMAIN_SENSITIVE, infer_role

_STEP = "outliers"

#: Minimum non-missing values for outlier detection. IQR quartiles and the
#: z-score need at least a few points to be meaningful; 4 is the floor at which
#: the fences are defined (``detection_bounds`` already no-ops on degenerate,
#: zero-spread data, so smaller-but-constant columns are still skipped safely).
_MIN_NON_NULL = 4
#: Minimum non-missing values for IsolationForest to be worth fitting.
_MIN_ISOLATION_ROWS = 100
#: Outlier share above which the distribution is treated as heavy-tailed.
_HEAVY_TAIL_SHARE = 0.15
#: Removing more than this share of rows in one clean triggers a warning.
_REMOVAL_WARN_SHARE = 0.02

#: Roles whose extreme values must never be capped or removed.
_PROTECTED_ROLES = frozenset({"id", "target"})


def auto_outliers(df: pd.DataFrame, config: CleanConfig,
                  report: CleanReport) -> pd.DataFrame:
    """Detect and handle outliers in every eligible numeric column.

    Skipped entirely when ``config.outliers`` is set — the explicit simple
    step (clip/flag everything) overrides the engine.
    """
    if config.outliers is not None or df.empty:
        return df
    rows_before = len(df)
    for col in list(df.columns):
        s = df[col]
        if not is_numeric_dtype(s) or is_bool_dtype(s):
            continue
        nonnull = s.dropna()
        if len(nonnull) < _MIN_NON_NULL:
            continue
        df = _handle_column(df, col, config, report)
    removed = rows_before - len(df)
    if removed and removed / rows_before > _REMOVAL_WARN_SHARE:
        report.add_warning(
            f"outlier removal dropped {removed} row(s) "
            f"({100 * removed / rows_before:.1f}% of the data); consider "
            'outlier_action="cap" instead'
        )
    return df


def _detect(s: pd.Series, config: CleanConfig):
    """Returns ``(mask, lo, hi, label)`` or ``None`` when nothing is detected."""
    if config.outlier_method == "isolation_forest":
        result = _isolation_detect(s, config)
        if result is not None:
            return result
        fallback_note = " (isolation_forest unavailable, fell back)"
    else:
        fallback_note = ""
    method = resolve_method(s, config)
    factor = factor_for(config, method)
    bounds = detection_bounds(s, method, factor)
    if bounds is None:
        return None
    lo, hi = integer_safe_bounds(s, *bounds)
    mask = (s < lo) | (s > hi)
    return mask, lo, hi, f"method={method}, factor={factor:g}{fallback_note}"


def _isolation_detect(s: pd.Series, config: CleanConfig):
    try:
        from sklearn.ensemble import IsolationForest  # noqa: PLC0415 — optional dependency
    except ImportError:
        return None
    nonnull = s.dropna()
    if len(nonnull) < _MIN_ISOLATION_ROWS:
        return None
    forest = IsolationForest(random_state=config.random_state, contamination="auto")
    predicted = forest.fit_predict(nonnull.to_numpy().reshape(-1, 1))
    outlying = predicted == -1
    if not outlying.any() or outlying.all():
        return None
    mask = pd.Series(False, index=s.index)
    mask.loc[nonnull.index[outlying]] = True
    inliers = nonnull[~outlying]
    lo, hi = integer_safe_bounds(s, float(inliers.min()), float(inliers.max()))
    return mask, lo, hi, "method=isolation_forest, contamination=auto"


def _handle_column(df: pd.DataFrame, col: object, config: CleanConfig,
                   report: CleanReport) -> pd.DataFrame:
    s = df[col]
    detected = _detect(s, config)
    if detected is None:
        return df
    mask, lo, hi, label = detected
    n = int(mask.sum())
    if n == 0:
        return df
    share = n / int(s.notna().sum())
    detail = f"{n} outlier(s), {100 * share:.1f}% of values ({label})"

    role = infer_role(str(col), s, config)
    preserve_reason: str | None = None
    if role in _PROTECTED_ROLES:
        preserve_reason = f"{role} column — its values must not be altered"
    elif str(col) in config.preserve_columns:
        preserve_reason = "explicitly listed in preserve_columns"
    elif _domain_sensitive(str(col)):
        preserve_reason = ("domain-sensitive column (fraud/anomaly/risk-like "
                          "name) where extreme values are usually the signal")
    elif config.outlier_action is None:
        preserve_reason = "outlier_action=None requested (detect and report only)"

    if preserve_reason is not None:
        report.add(_STEP, f"preserved {detail}", column=str(col), count=0,
                   rationale=preserve_reason, risk="low", confidence=0.9)
        if _domain_sensitive(str(col)):
            report.add_recommendation(
                f"column '{col}' has {n} extreme value(s) that were deliberately "
                "preserved; review them in their domain context"
            )
        return df

    action = config.outlier_action
    if action == "auto":
        action = "flag"
    else:
        explicit = config.outlier_action not in (None, "auto")
        if explicit and action in ("cap", "remove") and share > _HEAVY_TAIL_SHARE:
            report.add_warning(
                f"column '{col}': outlier_action={config.outlier_action!r} applied to "
                f"{n} value(s) ({100 * share:.0f}% of the column); >15% outlying often "
                "means a heavy-tailed distribution where the extremes are real — review "
                "whether this is appropriate"
            )

    confidence = 0.85 if share <= 0.02 else 0.7 if share <= 0.10 else 0.5
    risk = "low" if share <= 0.02 else "medium"

    if action == "cap":
        df[col] = s.clip(lo, hi)
        report.add(_STEP, f"capped {detail} to [{lo:g}, {hi:g}]",
                   column=str(col), count=n,
                   rationale="winsorizing keeps the rows but tames extreme "
                             "magnitudes; safer than deletion",
                   risk=risk, confidence=confidence)
        report.outliers_handled += n
    elif action == "remove":
        df = df.loc[~mask.fillna(False)]
        report.add(_STEP, f"removed rows with {detail}",
                   column=str(col), count=n,
                   rationale='outlier_action="remove" requested; rows outside '
                             f"[{lo:g}, {hi:g}] were dropped",
                   risk="medium" if share <= 0.02 else "high",
                   confidence=confidence)
        report.outliers_handled += n
    else:  # flag
        flag = unique_flag_name(df, f"{col}_outlier")
        df[flag] = mask.fillna(False).astype(bool)
        report.add(_STEP, f"flagged {detail} in new column {flag!r}",
                   column=str(col), count=n,
                   rationale="flagging records the detection without altering "
                             "any value",
                   risk="low", confidence=confidence)
        report.outliers_handled += n
    return df


def _domain_sensitive(name: str) -> bool:
    return bool(_DOMAIN_SENSITIVE.search(name))
