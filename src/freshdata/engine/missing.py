"""Threshold- and context-driven missing-value handling (``strategy="auto"``).

Decision bands, by missing ratio *r* (thresholds configurable):

- **low** (``r <= 0.05``): numeric — mean when approximately normal with no
  outliers, median otherwise; categorical — mode given a clear majority,
  ``"Unknown"`` otherwise; datetime — forward/backward fill when time order
  exists.
- **medium** (``r <= 0.30``): numeric — KNN imputation when scikit-learn is
  available and enough correlated numeric features exist, median otherwise;
  categorical — mode when dominant, ``"Missing"`` otherwise.
- **high** (``r <= 0.60``): kept (and conservatively imputed, with a warning)
  only when preserved, target-like, or the missingness is informative;
  dropped otherwise.
- **extreme** (``r > 0.60``): dropped unless explicitly preserved or a
  label/target column.

Role gates run first: targets are never modified, IDs are never imputed, free
text is never forced into numeric/sentinel fills. On frames smaller than
:data:`~freshdata.engine.context.MIN_ROWS_FOR_ENGINE` rows, only the low band
acts — ratios on tiny samples are too noisy to drop or impute aggressively.

Every decision (including "preserve") is logged with rationale, risk, and
confidence; an indicator column ``<col>_was_missing`` is added when the
missingness itself looks informative.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .._util import add_column
from ..config import CleanConfig
from ..report import CleanReport
from ..steps.missing import _mode_value
from .context import MIN_ROWS_FOR_ENGINE, ColumnContext, build_contexts
from .model_select import rank_missing_models
from .utils import _has_outliers

#: |skewness| below which mean imputation is acceptable.
_MEAN_OK_SKEW = 0.5
#: Minimum |correlation| for a column to count as a KNN imputation partner.
_KNN_MIN_CORR = 0.4
#: Above this many rows, KNN imputation is skipped (too slow).
_KNN_MAX_ROWS = 10_000

_STEP = "missing"


def auto_missing(df: pd.DataFrame, config: CleanConfig,
                 report: CleanReport,
                 contexts: dict | None = None,
                 numeric_corr: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply the missing-value rules to every column that has missing cells.

    Skipped entirely when ``config.impute`` is set — an explicit imputation
    choice overrides the engine (the legacy simple step runs instead).
    """
    if config.impute is not None or df.empty or config.engine_mode is None:
        return df
    if contexts is None:
        contexts = build_contexts(df, config)
    mode = config.engine_mode
    assert mode in ("balanced", "aggressive")
    for col in list(df.columns):
        if int(df[col].isna().sum()) == 0:
            continue
        ctx = contexts[col]
        df = _handle_column(df, col, ctx, config, report, mode=mode,
                            numeric_corr=numeric_corr)
    return df


def _band(ratio: float, config: CleanConfig) -> str:
    if ratio <= config.missing_threshold_low:
        return "low"
    if ratio <= config.missing_threshold_medium:
        return "medium"
    if ratio <= config.missing_threshold_high:
        return "high"
    return "extreme"


def _handle_column(df: pd.DataFrame, col: object, ctx: ColumnContext,
                   config: CleanConfig, report: CleanReport,
                   *, mode: str, numeric_corr: pd.DataFrame | None = None) -> pd.DataFrame:
    band = _band(ctx.missing_ratio, config)
    pct = f"{100 * ctx.missing_ratio:.1f}%"
    selection = rank_missing_models(df, col, ctx, config, mode=mode,  # type: ignore[arg-type]
                                    numeric_corr=numeric_corr)
    model_id = selection.primary.model_id

    # -- role gates: some columns must never be auto-filled -----------------
    if ctx.role == "target":
        _preserve(df, col, ctx, report,
                  rationale="label/target column — never modified automatically",
                  risk="high" if band != "low" else "medium", model_id=model_id)
        report.add_warning(
            f"target column '{col}' has {ctx.n_missing} missing value(s) ({pct}); "
            "rows without a label usually need to be dropped manually"
        )
        return df
    if ctx.role == "id":
        if (mode == "aggressive" and band == "extreme" and not ctx.preserve
                and ctx.n_rows >= MIN_ROWS_FOR_ENGINE):
            return _drop(df, col, ctx, config, report, band,
                         rationale="identifier-like column that is mostly missing "
                                   "cannot serve as a key and must not be imputed",
                         confidence=0.85, model_id=model_id)
        _preserve(df, col, ctx, report,
                  rationale="identifier-like column — imputing keys would "
                            "fabricate identities",
                  risk="medium", model_id=model_id)
        report.add_recommendation(
            f"id column '{col}' has {ctx.n_missing} missing value(s) ({pct}); "
            "check whether those rows are joinable at all"
        )
        return df
    if ctx.role == "text":
        _preserve(df, col, ctx, report,
                  rationale="free-text column — numeric or sentinel imputation "
                            "would fabricate content",
                  model_id=model_id)
        return df

    # -- tiny datasets: ratios are noise, act only on the low band ----------
    if ctx.n_rows < MIN_ROWS_FOR_ENGINE and band != "low":
        _preserve(df, col, ctx, report,
                  rationale=f"dataset too small ({ctx.n_rows} rows) to trust a "
                            f"{pct} missing ratio; imputation or dropping would "
                            "be guesswork",
                  risk="medium", confidence=0.6, model_id=model_id)
        report.add_recommendation(
            f"review '{col}' manually: {ctx.n_missing} missing value(s) in only "
            f"{ctx.n_rows} rows"
        )
        return df

    if band == "low":
        return _fill_low(df, col, ctx, config, report, model_id=model_id)
    if band == "medium":
        return _fill_medium(df, col, ctx, config, report, mode=mode, model_id=model_id,
                            numeric_corr=numeric_corr)
    if band == "high":
        return _handle_high(df, col, ctx, config, report, mode=mode, model_id=model_id)
    return _handle_extreme(df, col, ctx, config, report, mode=mode, model_id=model_id)


# -- low missingness (<= low threshold) --------------------------------------

def _fill_low(df: pd.DataFrame, col: object, ctx: ColumnContext,
              config: CleanConfig, report: CleanReport, *,
              model_id: str = "") -> pd.DataFrame:
    df = _maybe_indicator(df, col, ctx, config, report)
    s = df[col]
    if ctx.role == "numeric":
        skewed = ctx.skew is not None and abs(ctx.skew) >= _MEAN_OK_SKEW
        if skewed or _has_outliers(s):
            return _fill(df, col, ctx, report, s.median(), "median",
                         rationale="low missingness; skewed or outlier-bearing "
                                   "distribution, median is robust",
                         confidence=0.9, model_id=model_id or "median")
        return _fill(df, col, ctx, report, s.mean(), "mean",
                     rationale="low missingness; approximately normal "
                               "distribution with no strong outliers",
                     confidence=0.9, model_id=model_id or "mean")
    if ctx.role == "datetime":
        return _fill_datetime(df, col, ctx, report)
    if ctx.role == "boolean":
        return _fill(df, col, ctx, report, _mode_value(s), "mode",
                     rationale="low missingness in a boolean column",
                     confidence=0.85)
    # categorical
    if ctx.mode_ratio is not None and ctx.mode_ratio >= 0.5:
        return _fill(df, col, ctx, report, _mode_value(s), "mode",
                     rationale="low missingness with a clear majority value "
                               f"({100 * ctx.mode_ratio:.0f}% of non-missing)",
                     confidence=0.85)
    return _fill(df, col, ctx, report, "Unknown", 'sentinel "Unknown"',
                 rationale="low missingness but no dominant category; a "
                           "sentinel avoids inventing a majority",
                 confidence=0.7)


# -- medium missingness (<= medium threshold) ---------------------------------

def _fill_medium(df: pd.DataFrame, col: object, ctx: ColumnContext,
                 config: CleanConfig, report: CleanReport, *,
                 mode: str, model_id: str = "",
                 numeric_corr: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _maybe_indicator(df, col, ctx, config, report)
    s = df[col]
    if ctx.role == "numeric":
        if mode == "aggressive":
            knn = _knn_fill(df, col, config, numeric_corr=numeric_corr)
            if knn is not None:
                filled, partners = knn
                filled = filled.reindex(df.index)
                return _assign_filled(df, col, ctx, report, filled,
                                      f"KNN imputation (k=5) from correlated "
                                      f"features {partners}",
                                      rationale="medium missingness with enough "
                                                "correlated numeric features for "
                                                "model-based imputation",
                                      confidence=0.75, model_id="knn")
        return _fill(df, col, ctx, report, s.median(), "median",
                     rationale="medium missingness; median is the safe default "
                               "for numeric columns",
                     confidence=0.8, model_id=model_id or "median")
    if ctx.role == "datetime":
        return _fill_datetime(df, col, ctx, report)
    if ctx.role == "boolean":
        return _fill(df, col, ctx, report, _mode_value(s), "mode",
                     rationale="medium missingness in a boolean column",
                     confidence=0.75, risk="medium")
    if ctx.mode_ratio is not None and ctx.mode_ratio >= 0.6:
        return _fill(df, col, ctx, report, _mode_value(s), "mode",
                     rationale="medium missingness with a dominant category "
                               f"({100 * ctx.mode_ratio:.0f}% of non-missing)",
                     confidence=0.75, risk="medium")
    return _fill(df, col, ctx, report, "Missing", 'sentinel "Missing"',
                 rationale="medium missingness without a dominant category; "
                           "an explicit sentinel keeps the gap visible",
                 confidence=0.7, risk="medium")


# -- high missingness (<= high threshold) -------------------------------------

def _handle_high(df: pd.DataFrame, col: object, ctx: ColumnContext,
                 config: CleanConfig, report: CleanReport, *,
                 mode: str, model_id: str = "") -> pd.DataFrame:
    pct = f"{100 * ctx.missing_ratio:.1f}%"
    if mode == "balanced":
        _preserve(df, col, ctx, report,
                  rationale=f"{pct} missing — balanced mode preserves column "
                            "instead of dropping or force-imputing",
                  risk="high", confidence=0.7, model_id=model_id or "preserve")
        report.add_warning(
            f"column '{col}' has {pct} missing; review manually before analysis"
        )
        report.add_recommendation(
            f"column '{col}' preserved at {pct} missing in balanced mode"
        )
        return df
    keep_reason: str | None = None
    if ctx.preserve:
        keep_reason = "explicitly listed in preserve_columns"
    elif ctx.informative_missing:
        keep_reason = "its missingness correlates with other features"
    if keep_reason is None:
        return _drop(df, col, ctx, config, report, "high",
                     rationale=f"{pct} missing with no sign the column is "
                               "important (not preserved, not a label, "
                               "missingness not informative)",
                     confidence=0.7, model_id=model_id or "drop")

    df = _maybe_indicator(df, col, ctx, config, report, force=True)
    report.add_warning(
        f"column '{col}' kept at {pct} missing ({keep_reason}); its imputed "
        "values are low-confidence"
    )
    s = df[col]
    if ctx.role == "numeric":
        return _fill(df, col, ctx, report, s.median(), "median",
                     rationale=f"high missingness but kept ({keep_reason}); "
                               "conservative median fill",
                     confidence=0.5, risk="high")
    if ctx.role == "datetime":
        return _fill_datetime(df, col, ctx, report, risk="high", confidence=0.5)
    if ctx.role == "boolean":
        return _fill(df, col, ctx, report, _mode_value(s), "mode",
                     rationale=f"high missingness but kept ({keep_reason})",
                     confidence=0.5, risk="high")
    return _fill(df, col, ctx, report, "Missing", 'sentinel "Missing"',
                 rationale=f"high missingness but kept ({keep_reason}); the "
                           "sentinel keeps the gap visible",
                 confidence=0.5, risk="high")


# -- extreme missingness (> high threshold) -----------------------------------

def _handle_extreme(df: pd.DataFrame, col: object, ctx: ColumnContext,
                    config: CleanConfig, report: CleanReport, *,
                    mode: str, model_id: str = "") -> pd.DataFrame:
    pct = f"{100 * ctx.missing_ratio:.1f}%"
    if ctx.missing_ratio == 1.0 and not config.drop_empty_columns:
        _preserve(df, col, ctx, report,
                  rationale="all-missing column kept because "
                            "drop_empty_columns=False was requested",
                  risk="medium", model_id=model_id or "preserve")
        return df
    if mode == "balanced" or ctx.preserve:
        if ctx.preserve:
            rationale = (f"{pct} missing but explicitly listed in "
                         "preserve_columns; left untouched")
        else:
            rationale = f"{pct} missing — balanced mode preserves column"
        _preserve(df, col, ctx, report, rationale=rationale,
                  risk="high", confidence=0.9, model_id=model_id or "preserve")
        report.add_warning(
            f"column '{col}' is {pct} missing and was preserved; "
            "treat any statistics computed from it with caution"
        )
        report.add_recommendation(
            f"column '{col}' preserved at {pct} missing in balanced mode"
        )
        return df
    return _drop(df, col, ctx, config, report, "extreme",
                 rationale=f"{pct} missing — too sparse to impute credibly",
                 confidence=0.9, model_id=model_id or "drop")


# -- shared mechanics ---------------------------------------------------------

def _fill_datetime(df: pd.DataFrame, col: object, ctx: ColumnContext,
                   report: CleanReport, risk: str = "low",
                   confidence: float = 0.8) -> pd.DataFrame:
    if not ctx.time_ordered:
        _preserve(df, col, ctx, report,
                  rationale="datetime column without a usable time order; "
                            "forward/backward fill would invent timestamps",
                  risk="medium")
        return df
    filled = df[col].ffill().bfill()
    n = ctx.n_missing - int(filled.isna().sum())
    df[col] = filled
    report.add(_STEP, f"filled {n} missing value(s) by forward/backward fill",
               column=str(col), count=n,
               rationale="datetime column with monotonic time order",
               risk=risk, confidence=confidence)
    report.columns_imputed.append(str(col))
    return df


def _fill(df: pd.DataFrame, col: object, ctx: ColumnContext, report: CleanReport,
          value: Any, label: str, *, rationale: str, confidence: float,
          risk: str = "low", model_id: str = "") -> pd.DataFrame:
    """Fill the column's missing cells with *value*, with dtype care."""
    s = df[col]
    if value is None or pd.isna(value):
        _preserve(df, col, ctx, report,
                  rationale="no usable fill value could be derived "
                            "(non-missing values are unhashable or undefined)",
                  risk="medium", confidence=0.6, model_id="preserve")
        return df
    if isinstance(s.dtype, pd.CategoricalDtype) and value not in s.cat.categories:
        s = s.cat.add_categories([value])
    cast_note = ""
    try:
        filled = s.fillna(value)
    except (TypeError, ValueError):
        if is_numeric_dtype(s) and isinstance(value, float):
            filled = s.astype("float64").fillna(value)
            cast_note = ", column cast to float64"
        else:
            _preserve(df, col, ctx, report,
                      rationale=f"fill value not representable in dtype {s.dtype}",
                      risk="medium", confidence=0.6, model_id="preserve")
            return df
    df[col] = filled
    shown = f"{value:.6g}" if isinstance(value, float) else repr(value)
    report.add(_STEP,
               f"filled {ctx.n_missing} missing value(s) with {label} "
               f"({shown}{cast_note})",
               column=str(col), count=ctx.n_missing, rationale=rationale,
               risk=risk, confidence=confidence, model_id=model_id or label.split(maxsplit=1)[0])
    report.columns_imputed.append(str(col))
    return df


def _assign_filled(df: pd.DataFrame, col: object, ctx: ColumnContext,
                   report: CleanReport, filled_values: pd.Series, label: str, *,
                   rationale: str, confidence: float,
                   model_id: str = "knn") -> pd.DataFrame:
    """Replace only the missing positions of the column with *filled_values*."""
    s = df[col]
    filled_values = filled_values.reindex(s.index)
    try:
        combined = s.where(s.notna(), filled_values)
    except (TypeError, ValueError):
        combined = s.astype("float64").where(s.notna(), filled_values)
    df[col] = combined
    report.add(_STEP, f"filled {ctx.n_missing} missing value(s) with {label}",
               column=str(col), count=ctx.n_missing, rationale=rationale,
               risk="medium", confidence=confidence, model_id=model_id)
    report.columns_imputed.append(str(col))
    return df


def _preserve(df: pd.DataFrame, col: object, ctx: ColumnContext,
              report: CleanReport, *, rationale: str, risk: str = "low",
              confidence: float = 1.0, model_id: str = "preserve") -> None:
    report.add(_STEP, f"preserved {ctx.n_missing} missing value(s)",
               column=str(col), count=0, rationale=rationale, risk=risk,
               confidence=confidence, model_id=model_id)
    report.columns_preserved.append(str(col))


def _drop(df: pd.DataFrame, col: object, ctx: ColumnContext, config: CleanConfig,
          report: CleanReport, band: str, *, rationale: str,
          confidence: float, model_id: str = "drop") -> pd.DataFrame:
    df = df.drop(columns=[col])
    report.add(_STEP,
               f"dropped column ({100 * ctx.missing_ratio:.1f}% missing, "
               f"{band} band)",
               column=str(col), count=ctx.n_missing, rationale=rationale,
               risk="medium", confidence=confidence, model_id=model_id)
    report.columns_dropped.append(str(col))
    report.add_recommendation(
        f"column '{col}' was dropped at {100 * ctx.missing_ratio:.1f}% missing; "
        f"pass preserve_columns=({str(col)!r},) to keep it"
    )
    return df


def _maybe_indicator(df: pd.DataFrame, col: object, ctx: ColumnContext,
                     config: CleanConfig, report: CleanReport,
                     force: bool = False) -> pd.DataFrame:
    """Add ``<col>_was_missing`` when missingness may carry signal."""
    wanted = (
        config.missing_indicators is True
        or (config.missing_indicators == "auto"
            and (force or ctx.informative_missing))
    )
    if not wanted:
        return df
    name = f"{col}_was_missing"
    if name in df.columns:  # idempotent re-cleaning
        return df
    add_column(df, name, df[col].isna())
    report.add(_STEP, f"added missing indicator column {name!r}",
               column=str(col), count=ctx.n_missing,
               rationale="missingness itself may be informative; the indicator "
                         "preserves that signal through imputation",
               confidence=0.7)
    return df


def _knn_fill(
    df: pd.DataFrame,
    col: object,
    config: CleanConfig,
    *,
    numeric_corr: pd.DataFrame | None = None,
):
    """KNN-impute one numeric column from correlated numeric partners.

    Returns ``(filled_series, partner_names)`` or ``None`` when scikit-learn
    is unavailable, the frame is too large, or fewer than two sufficiently
    correlated partners exist.
    """
    if config.advanced_imputation is False or config.engine_mode != "aggressive":
        return None
    try:
        from sklearn.impute import KNNImputer  # noqa: PLC0415 — optional dependency
    except ImportError:
        return None
    # Always honor the row limit: KNN is ~O(n²); above the cap we fall back to
    # median. rank_missing_models reports the same gate, so the report and the
    # actual behavior stay consistent even when advanced_imputation=True.
    if len(df) > _KNN_MAX_ROWS:
        return None
    others = [
        c for c in df.columns
        if c != col and is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
        and df[c].notna().any()
    ]
    if len(others) < 2:
        return None
    if numeric_corr is not None and str(col) in numeric_corr.columns:
        corr = numeric_corr.loc[others, str(col)]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # constant partners yield NaN corr
            corr = df[others].corrwith(df[col]).abs()
    partners = [c for c in others if pd.notna(corr[c]) and corr[c] >= _KNN_MIN_CORR]
    if len(partners) < 2:
        return None
    partners = _non_collinear_partners(df, partners, corr)
    sub = df[[col, *partners]].astype("float64")
    # sklearn may emit benign RuntimeWarnings when partner columns are nearly
    # collinear (common on real exports); imputed values are still finite.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            filled = KNNImputer(n_neighbors=5).fit_transform(sub)
    return pd.Series(filled[:, 0], index=df.index), [str(p) for p in partners]


def _non_collinear_partners(
    df: pd.DataFrame, partners: list, corr: pd.Series, *, max_partners: int = 5
) -> list:
    """Pick partners greedily, skipping ones almost perfectly collinear."""
    _MAX_PARTNER_CORR = 0.98
    ranked = sorted(partners, key=lambda c: float(corr[c]), reverse=True)
    chosen: list = []
    for candidate in ranked:
        if len(chosen) >= max_partners:
            break
        if not chosen:
            chosen.append(candidate)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                partner_corr = df[chosen].corrwith(df[candidate]).abs().max()
        except Exception:
            partner_corr = 0.0
        if pd.isna(partner_corr) or partner_corr < _MAX_PARTNER_CORR:
            chosen.append(candidate)
    return chosen if len(chosen) >= 2 else partners[:max_partners]
