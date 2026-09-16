"""Score and rank imputation / outlier models per column context."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from ..config import CleanConfig
from .context import MIN_ROWS_FOR_ENGINE, ColumnContext
from .utils import _has_outliers

EngineMode = Literal["balanced", "aggressive"]
_MEAN_OK_SKEW = 0.5
_KNN_MIN_CORR = 0.4
_KNN_ROW_LIMIT = 10_000
#: Outlier share above which "auto" treats a column as heavy-tailed and flags
#: rather than caps (kept in sync with engine.outliers._HEAVY_TAIL_SHARE).
_HEAVY_TAIL_SHARE = 0.15


@dataclass(frozen=True)
class ModelChoice:
    """One ranked cleaning model candidate."""

    model_id: str
    confidence: float
    rationale: str
    eligible: bool = True
    rejection_reason: str = ""


@dataclass(frozen=True)
class MissingModelSelection:
    """Primary missing-value model plus ranked alternatives."""

    primary: ModelChoice
    alternatives: tuple[ModelChoice, ...]


def _band(ratio: float, config: CleanConfig) -> str:
    if ratio <= config.missing_threshold_low:
        return "low"
    if ratio <= config.missing_threshold_medium:
        return "medium"
    if ratio <= config.missing_threshold_high:
        return "high"
    return "extreme"


def _finite_float(series: pd.Series) -> pd.Series:
    """``series`` as plain ``float64`` with missing *and* non-finite cells as ``NaN``.

    Correlating the raw column is not safe on every install: a nullable dtype
    reaches ``Series.corr`` as an ``object`` array on pandas 1.5, and numpy 1.26
    cannot take the covariance of an object array at all (``np.average`` builds
    its scale factor with ``dtype.type(...)``, which for ``object`` hands back a
    bare Python ``float`` and then trips over ``scl.shape``). Casting to
    ``float64`` up front is exactly what pandas 2.x does inside ``Series.corr``,
    so ordinary numeric columns keep the values — and the correlations — they
    have today, while ``inf`` / ``-inf`` are treated as missing instead of
    poisoning the covariance.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = series.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    except (TypeError, ValueError, OverflowError):
        values = np.full(len(series), np.nan, dtype="float64")
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=series.index, name=series.name)


def _correlatable(series: pd.Series) -> bool:
    """True when a sanitized column can carry a meaningful Pearson correlation."""
    finite = series.dropna()
    if len(finite) < 2:
        return False
    return finite.nunique() >= 2


def _corr_against(others: pd.DataFrame, target: pd.Series) -> pd.Series:
    """``|corr|`` of every column of ``others`` against ``target``.

    Degenerate columns (no two distinct finite values) and correlations pandas
    or numpy cannot produce score ``NaN``, which the caller reads as "no usable
    partner" — never as a reason to abort cleaning.
    """
    blank = pd.Series(np.nan, index=others.columns, dtype="float64")
    clean_target = _finite_float(target)
    if not _correlatable(clean_target):
        return blank
    clean_others = others.apply(_finite_float)
    keep = [i for i in range(clean_others.shape[1])
            if _correlatable(clean_others.iloc[:, i])]
    if not keep:
        return blank
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = clean_others.iloc[:, keep].corrwith(clean_target).abs()
        scores = np.asarray(corr, dtype="float64")
    except Exception:  # pragma: no cover - defensive: never abort cleaning
        return blank
    out = blank.copy()
    out.iloc[keep] = scores
    return out


def _partner_info(
    df: pd.DataFrame,
    col: object,
    numeric_corr: pd.DataFrame | None = None,
) -> tuple[list[str], pd.Series | None]:
    others = [
        c for c in df.columns
        if c != col and is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
        and df[c].notna().any()
    ]
    if not others:
        return [], None
    corr: pd.Series | None = None
    if numeric_corr is not None and str(col) in numeric_corr.columns:
        try:
            corr = numeric_corr.loc[others, str(col)]
        except KeyError:  # cached matrix predates a column — recompute pairwise
            corr = None
    if corr is None:
        corr = _corr_against(df[others], df[col])
    partners = [c for c in others if pd.notna(corr[c]) and corr[c] >= _KNN_MIN_CORR]
    return partners, corr


def rank_missing_models(
    df: pd.DataFrame,
    col: object,
    ctx: ColumnContext,
    config: CleanConfig,
    *,
    mode: EngineMode,
    numeric_corr: pd.DataFrame | None = None,
) -> MissingModelSelection:
    """Rank missing-value models for one column (dry-run safe)."""
    band = _band(ctx.missing_ratio, config)
    choices: list[ModelChoice] = []

    def add(model_id: str, score: float, rationale: str, *, eligible: bool = True,
            rejection: str = "") -> None:
        choices.append(ModelChoice(model_id, max(0.0, min(1.0, score)), rationale,
                                eligible=eligible, rejection_reason=rejection))

    if ctx.role == "target":
        add("preserve", 1.0, "target/label column")
    elif ctx.role == "id":
        add("preserve", 0.95, "identifier column")
        if mode == "aggressive" and band == "extreme" and not ctx.preserve:
            add("drop", 0.7, "mostly-missing id in aggressive mode", eligible=True)
    elif ctx.role == "text":
        add("preserve", 1.0, "free-text column")
    elif ctx.n_rows < MIN_ROWS_FOR_ENGINE and band != "low":
        add("preserve", 0.9, f"dataset too small ({ctx.n_rows} rows)")
    elif mode == "balanced" and band in ("high", "extreme"):
        add("preserve", 0.85, f"{band} missingness — balanced mode keeps columns")
    elif mode == "aggressive" and band in ("high", "extreme") and not ctx.preserve:
        if not ctx.informative_missing:
            add("drop", 0.75, f"{band} missingness without informative signal")
        else:
            add("preserve", 0.6, "informative missingness or preserved column")
    elif ctx.role == "datetime":
        if ctx.time_ordered:
            add("time_fill", 0.9, "monotonic datetime")
        else:
            add("preserve", 0.8, "datetime without usable order")
    elif ctx.role == "numeric":
        skewed = ctx.skew is not None and abs(ctx.skew) >= _MEAN_OK_SKEW
        outlier_bearing = _has_outliers(df[col])
        partners, corr = _partner_info(df, col, numeric_corr)
        if band == "low" and not skewed and not outlier_bearing:
            add("mean", 0.9, "low missingness, ~normal distribution")
            add("median", 0.7, "robust alternative", rejection="normal distribution")
        else:
            add("median", 0.88, "robust numeric default")
            add("mean", 0.5, "mean less robust under skew/outliers",
                rejection="skew or outliers present")
        if band == "medium" and partners:
            best = max(partners, key=lambda c: float(corr[c]))  # type: ignore[index]
            add("partner_median", 0.82, f"correlated partner {best!r}")
            knn_ok = (
                mode == "aggressive"
                and len(partners) >= 2
                and len(df) <= _KNN_ROW_LIMIT
            )
            add("knn", 0.78 if knn_ok else 0.3,
                "KNN imputation from correlated features",
                eligible=knn_ok,
                rejection="" if knn_ok else "balanced mode or insufficient partners/rows")
            add("linear", 0.72 if mode == "aggressive" else 0.25,
                "linear regression from best partner",
                eligible=mode == "aggressive",
                rejection="" if mode == "aggressive" else "balanced strategy")
    elif ctx.role in ("categorical", "boolean"):
        threshold = 0.6 if band == "medium" else 0.5
        if ctx.mode_ratio is not None and ctx.mode_ratio >= threshold:
            add("mode", 0.85, "dominant category present")
            add("sentinel", 0.55, "sentinel fallback", rejection="clear majority exists")
        else:
            add("sentinel", 0.8, "no dominant category — explicit sentinel")
            add("mode", 0.45, "mode without majority", rejection="no dominant category")

    eligible = [c for c in choices if c.eligible]
    if not eligible:
        eligible = choices
    ranked = sorted(eligible, key=lambda c: c.confidence, reverse=True)
    primary = ranked[0]
    alts = tuple(c for c in sorted(choices, key=lambda c: c.confidence, reverse=True)
                 if c.model_id != primary.model_id)
    return MissingModelSelection(primary=primary, alternatives=alts[:4])


def select_outlier_action(
    ctx: ColumnContext,
    config: CleanConfig,
    *,
    mode: EngineMode,
    share: float,
) -> tuple[str | None, ModelChoice]:
    """Choose the outlier action; returns ``(action or None, primary choice)``.

    ``None`` means "detect but preserve". Protected columns (id / target /
    name-matched domain-sensitive columns when ``domain_sensitive_names=True``)
    are always preserved. ``outlier_action="auto"`` (the default) FLAGS under
    every strategy — balanced and aggressive alike — and never rewrites
    values; capping happens only on an explicit directive. An explicit
    ``"cap"`` / ``"remove"`` / ``"flag"`` is honored on every eligible column;
    the caller raises a warning when a directive acts on a heavy-tailed column.
    """
    if ctx.role in ("id", "target") or (
        config.domain_sensitive_names and ctx.domain_sensitive
    ):
        return None, ModelChoice("preserve", 0.95, "protected column role or domain")
    action = config.outlier_action
    if action is None:
        return None, ModelChoice("preserve", 1.0, "outlier_action=None")
    if action == "auto":
        # "auto" never winsorizes: rewriting values is explicit-only.
        if share > _HEAVY_TAIL_SHARE:  # heavy-tailed: the extremes are likely real
            return "flag", ModelChoice("flag", 0.9,
                                       "heavy-tailed distribution (>15% outlying)")
        return "flag", ModelChoice(
            "flag", 0.85,
            'auto flags without altering values; capping requires outlier_action="cap"',
        )
    # Explicit directive ("cap" / "remove" / "flag") — honor it on every column.
    conf = 0.85 if share <= 0.02 else 0.7
    return action, ModelChoice(str(action), conf, f"explicit outlier_action={action!r}")
