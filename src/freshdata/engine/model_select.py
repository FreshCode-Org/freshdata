"""Score and rank imputation / outlier models per column context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    if numeric_corr is not None and str(col) in numeric_corr.columns:
        corr = numeric_corr.loc[others, str(col)]
    else:
        corr = df[others].corrwith(df[col]).abs()
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
    domain-sensitive) are always preserved. ``outlier_action="auto"`` (the
    default) is context-aware — it flags under ``mode="balanced"`` and caps
    under ``"aggressive"``, and flags heavy-tailed columns (>15% outlying)
    rather than rewriting real data. An explicit ``"cap"`` / ``"remove"`` /
    ``"flag"`` is honored as a directive on every eligible column; the caller
    raises a warning when a directive acts on a heavy-tailed column.
    """
    if ctx.role in ("id", "target") or ctx.domain_sensitive:
        return None, ModelChoice("preserve", 0.95, "protected column role or domain")
    action = config.outlier_action
    if action is None:
        return None, ModelChoice("preserve", 1.0, "outlier_action=None")
    if action == "auto":
        if share > _HEAVY_TAIL_SHARE:  # heavy-tailed: the extremes are likely real
            return "flag", ModelChoice("flag", 0.9,
                                       "heavy-tailed distribution (>15% outlying)")
        if mode == "balanced":
            return "flag", ModelChoice("flag", 0.85, "balanced mode flags rather than caps")
        conf = 0.85 if share <= 0.02 else 0.7
        return "cap", ModelChoice("cap", conf, "aggressive mode winsorizes to fences")
    # Explicit directive ("cap" / "remove" / "flag") — honor it on every column.
    conf = 0.85 if share <= 0.02 else 0.7
    return action, ModelChoice(str(action), conf, f"explicit outlier_action={action!r}")
