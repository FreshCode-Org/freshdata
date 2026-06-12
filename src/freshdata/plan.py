"""Dry-run cleaning plans: preview engine decisions before mutating data."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .cleaner import run_pipeline
from .config import CleanConfig, merge_options
from .engine.context import build_contexts
from .engine.model_select import ModelChoice, rank_missing_models, select_outlier_action
from .engine.outliers import _MIN_NON_NULL, _detect


@dataclass(frozen=True)
class ColumnPlan:
    """Primary and alternative models for one column."""

    column: str
    missing: ModelChoice | None = None
    missing_alternatives: tuple[ModelChoice, ...] = ()
    outlier: ModelChoice | None = None
    outlier_action: str | None = None
    n_outliers: int = 0


@dataclass
class CleanPlan:
    """Recommended cleaning configuration and per-column model choices."""

    config: CleanConfig
    column_plans: dict[str, ColumnPlan] = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable primary model per column."""
        lines = [
            f"freshdata clean plan (strategy={self.config.strategy!r})",
            f"  columns: {len(self.column_plans)}",
        ]
        if not self.column_plans:
            lines.append("  (no engine actions — conservative or empty frame)")
            return "\n".join(lines)
        name_w = min(24, max(6, *(len(c) for c in self.column_plans)))
        lines.append(f"  {'column':<{name_w}}  missing_model    outlier_action")
        for col, plan in sorted(self.column_plans.items()):
            miss = plan.missing.model_id if plan.missing else "-"
            out = plan.outlier_action or (plan.outlier.model_id if plan.outlier else "-")
            lines.append(f"  {col:<{name_w}}  {miss:<15}  {out}")
        return "\n".join(lines)

    def alternatives(self) -> pd.DataFrame:
        """One row per (column, model, rank) for notebook review."""
        rows: list[tuple[str, str, str, int, float, str, bool, str]] = []
        for col, plan in sorted(self.column_plans.items()):
            if plan.missing:
                rows.append((
                    col, "missing", plan.missing.model_id, 0,
                    plan.missing.confidence, plan.missing.rationale,
                    plan.missing.eligible, plan.missing.rejection_reason,
                ))
                for rank, alt in enumerate(plan.missing_alternatives, start=1):
                    rows.append((
                        col, "missing", alt.model_id, rank,
                        alt.confidence, alt.rationale,
                        alt.eligible, alt.rejection_reason,
                    ))
            if plan.outlier:
                rows.append((
                    col, "outlier", plan.outlier.model_id, 0,
                    plan.outlier.confidence, plan.outlier.rationale,
                    plan.outlier.eligible, plan.outlier.rejection_reason,
                ))
        if not rows:
            return pd.DataFrame(columns=[
                "column", "step", "model_id", "rank", "confidence",
                "rationale", "eligible", "rejection_reason",
            ])
        return pd.DataFrame(
            rows,
            columns=[
                "column", "step", "model_id", "rank", "confidence",
                "rationale", "eligible", "rejection_reason",
            ],
        )

    def to_frame(self) -> pd.DataFrame:
        """One row per column with primary missing/outlier choices."""
        return pd.DataFrame([
            {
                "column": col,
                "missing_model": p.missing.model_id if p.missing else None,
                "missing_confidence": p.missing.confidence if p.missing else None,
                "outlier_action": p.outlier_action,
                "outlier_model": p.outlier.model_id if p.outlier else None,
                "n_outliers": p.n_outliers,
            }
            for col, p in sorted(self.column_plans.items())
        ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.config.strategy,
            "columns": {
                col: {
                    "missing": _choice_dict(plan.missing),
                    "missing_alternatives": [
                        _choice_dict(c) for c in plan.missing_alternatives
                    ],
                    "outlier": _choice_dict(plan.outlier),
                    "outlier_action": plan.outlier_action,
                    "n_outliers": plan.n_outliers,
                }
                for col, plan in self.column_plans.items()
            },
        }

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<CleanPlan: {len(self.column_plans)} column(s), "
            f"strategy={self.config.strategy!r}>"
        )


def _choice_dict(choice: ModelChoice | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    return {
        "model_id": choice.model_id,
        "confidence": choice.confidence,
        "rationale": choice.rationale,
        "eligible": choice.eligible,
        "rejection_reason": choice.rejection_reason,
    }


def _repair_preview(df: pd.DataFrame, config: CleanConfig) -> pd.DataFrame:
    """Representation-repair preview used for dry-run planning."""
    preview_cfg = merge_options(
        config,
        strategy="conservative",
        verbose=False,
        impute=None,
        outliers=None,
    )
    preview, _ = run_pipeline(df, preview_cfg)
    return preview


def suggest_plan(
    df: pd.DataFrame,
    *,
    config: CleanConfig | None = None,
    **options: object,
) -> CleanPlan:
    """Preview engine model choices without mutating *df*."""
    cfg = merge_options(config, **options)
    if cfg.engine_mode is None:
        return CleanPlan(config=cfg)
    preview = _repair_preview(df, cfg)
    if preview.empty:
        return CleanPlan(config=cfg)
    mode = cfg.engine_mode
    assert mode in ("balanced", "aggressive")
    contexts = build_contexts(preview, cfg)
    plans: dict[str, ColumnPlan] = {}
    for col in preview.columns:
        ctx = contexts[col]
        missing_choice: ModelChoice | None = None
        missing_alts: tuple[ModelChoice, ...] = ()
        if int(preview[col].isna().sum()) > 0:
            sel = rank_missing_models(preview, col, ctx, cfg, mode=mode)  # type: ignore[arg-type]
            missing_choice = sel.primary
            missing_alts = sel.alternatives
        outlier_choice: ModelChoice | None = None
        outlier_action: str | None = None
        n_outliers = 0
        s = preview[col]
        if (is_numeric_dtype(s) and not is_bool_dtype(s)
                and int(s.notna().sum()) >= _MIN_NON_NULL and cfg.outliers is None):
            detected = _detect(s, cfg)
            if detected is not None:
                mask, _, _, _ = detected
                n_outliers = int(mask.sum())
                if n_outliers:
                    share = n_outliers / int(s.notna().sum())
                    action, choice = select_outlier_action(
                        ctx, cfg, mode=mode, share=share  # type: ignore[arg-type]
                    )
                    outlier_choice = choice
                    outlier_action = action
        if missing_choice or outlier_choice:
            plans[str(col)] = ColumnPlan(
                column=str(col),
                missing=missing_choice,
                missing_alternatives=missing_alts,
                outlier=outlier_choice,
                outlier_action=outlier_action,
                n_outliers=n_outliers,
            )
    return CleanPlan(config=cfg, column_plans=plans)


def compare_plans(
    df: pd.DataFrame,
    *,
    strategies: tuple[str, ...] = ("conservative", "balanced", "aggressive"),
    config: CleanConfig | None = None,
    include_metrics: bool = False,
    **options: object,
) -> pd.DataFrame:
    """Side-by-side primary models for each strategy.

    With ``include_metrics=True``, adds actual clean outcomes (missing_after,
    duration_seconds, …) from :func:`compare_clean`.
    """
    base = merge_options(config, **options)
    rows: list[dict[str, Any]] = []
    metrics: pd.DataFrame | None = None
    if include_metrics:
        metrics = compare_clean(df, strategies=strategies, config=base)
        metrics = metrics.set_index("strategy")
    for strategy in strategies:
        plan = suggest_plan(df, config=merge_options(base, strategy=strategy))
        for col, cp in plan.column_plans.items():
            row: dict[str, Any] = {
                "column": col,
                "strategy": strategy,
                "missing_model": cp.missing.model_id if cp.missing else None,
                "outlier_action": cp.outlier_action,
                "n_outliers": cp.n_outliers,
            }
            if metrics is not None and strategy in metrics.index:
                m = metrics.loc[strategy]
                row["missing_after"] = m["missing_after"]
                row["duration_seconds"] = m["duration_seconds"]
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[
            "column", "strategy", "missing_model", "outlier_action", "n_outliers",
        ])
    return pd.DataFrame(rows)


def _primary_models_from_report(report) -> dict[str, str]:
    """Map column name to last engine model_id from a clean report."""
    models: dict[str, str] = {}
    for action in report:
        if action.step not in ("missing", "outliers") or not action.column:
            continue
        if action.model_id:
            models[str(action.column)] = action.model_id
        elif action.step == "missing" and "preserved" in action.description:
            models[str(action.column)] = "preserve"
    return models


def compare_clean(
    df: pd.DataFrame,
    *,
    strategies: tuple[str, ...] = ("conservative", "balanced", "aggressive"),
    config: CleanConfig | None = None,
    **options: object,
) -> pd.DataFrame:
    """Run clean under each strategy and compare quality + efficiency metrics."""
    base = merge_options(config, **options)
    rows: list[dict[str, Any]] = []
    n_rows = len(df)
    for strategy in strategies:
        cfg = merge_options(base, strategy=strategy, verbose=False)
        _, report = run_pipeline(df, cfg)
        rows.append({
            "strategy": strategy,
            "rows_before": report.rows_before,
            "rows_after": report.rows_after,
            "cols_before": report.cols_before,
            "cols_after": report.cols_after,
            "missing_before": report.missing_before,
            "missing_after": report.missing_after,
            "missing_delta": report.missing_after - report.missing_before,
            "cols_delta": report.cols_after - report.cols_before,
            "duplicates_removed": report.duplicates_removed,
            "outliers_handled": report.outliers_handled,
            "columns_dropped": len(report.columns_dropped),
            "columns_imputed": len(report.columns_imputed),
            "duration_seconds": round(report.duration_seconds, 4),
            "rows_per_second": round(n_rows / report.duration_seconds, 1)
            if report.duration_seconds > 0 else None,
            "primary_models": json.dumps(_primary_models_from_report(report)),
        })
    return pd.DataFrame(rows)
