"""Dry-run cleaning plans and repair artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from .cleaner import run_pipeline
from .config import CleanConfig, merge_options
from .engine.context import build_contexts
from .engine.model_select import ModelChoice, rank_missing_models, select_outlier_action
from .engine.outliers import _MIN_NON_NULL, _detect
from .report import Action, CleanReport

REPAIR_MODES = ("inspect", "suggest", "repair_safe", "repair_reviewed", "repair_aggressive")


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


@dataclass(frozen=True)
class RepairPatch:
    """One reversible row, column, or cell-level change proposed by a repair plan."""

    patch_id: str
    operation: str
    row: Any | None = None
    column: str | None = None
    old_value: Any = None
    new_value: Any = None
    step: str = "observed_diff"
    rationale: str = ""
    risk: str = "low"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for JSON serialization."""
        return {
            "patch_id": self.patch_id,
            "operation": self.operation,
            "row": _json_value(self.row),
            "column": self.column,
            "old_value": _json_value(self.old_value),
            "new_value": _json_value(self.new_value),
            "step": self.step,
            "rationale": self.rationale,
            "risk": self.risk,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class ReviewItem:
    """A proposed change that should be explicitly approved before application."""

    patch_id: str
    reason: str
    risk: str
    confidence: float
    row: Any | None = None
    column: str | None = None
    operation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form suitable for review queues and JSON logging."""
        return {
            "patch_id": self.patch_id,
            "reason": self.reason,
            "risk": self.risk,
            "confidence": self.confidence,
            "row": _json_value(self.row),
            "column": self.column,
            "operation": self.operation,
        }


@dataclass
class RepairPlan:
    """Previewable, serializable, and reversible repair artifact."""

    config: CleanConfig
    mode: str
    source_fingerprint: str
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]
    report: CleanReport = field(default_factory=CleanReport)
    patches: tuple[RepairPatch, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    _before: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False, compare=False)
    _after: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False, compare=False)

    @property
    def patch_count(self) -> int:
        """Number of proposed changes."""
        return len(self.patches)

    def apply(self, approved_patch_ids: set[str] | None = None) -> pd.DataFrame:
        """Apply all patches, or only the supplied approved patch identifiers."""
        if approved_patch_ids is None:
            return self._after.copy(deep=True)
        approved = set(approved_patch_ids)
        out = self._before.copy(deep=True)
        for patch in self.patches:
            if patch.patch_id not in approved:
                continue
            if patch.operation == "update_cell" and patch.column is not None:
                out.at[patch.row, patch.column] = patch.new_value
            elif patch.operation == "drop_row" and patch.row in out.index:
                out = out.drop(index=patch.row)
            elif patch.operation == "drop_column" and patch.column in out.columns:
                out = out.drop(columns=[patch.column])
            elif patch.operation == "add_column" and patch.column is not None:
                out[patch.column] = patch.new_value
        return out

    def rollback(self) -> pd.DataFrame:
        """Return the original frame captured when the plan was built."""
        return self._before.copy(deep=True)

    def review_queue(self) -> pd.DataFrame:
        """Review items as a DataFrame for notebooks, CSV, or lightweight UIs."""
        return pd.DataFrame([item.to_dict() for item in self.review_items])

    def to_frame(self) -> pd.DataFrame:
        """One row per patch."""
        return pd.DataFrame([patch.to_dict() for patch in self.patches])

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, suitable for JSON serialization or logging."""
        return {
            "mode": self.mode,
            "source_fingerprint": self.source_fingerprint,
            "before_shape": self.before_shape,
            "after_shape": self.after_shape,
            "patch_count": len(self.patches),
            "review_count": len(self.review_items),
            "report": self.report.to_dict(),
            "patches": [patch.to_dict() for patch in self.patches],
            "review_items": [item.to_dict() for item in self.review_items],
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialize the repair plan to JSON."""
        return json.dumps(self.to_dict(), **kwargs)

    def to_markdown(self) -> str:
        """Human-readable Markdown summary for PRs, run logs, and notebooks."""
        lines = [
            "# freshdata repair plan",
            "",
            f"- mode: `{self.mode}`",
            f"- source fingerprint: `{self.source_fingerprint}`",
            f"- shape: `{self.before_shape}` -> `{self.after_shape}`",
            f"- patches: `{len(self.patches)}`",
            f"- review items: `{len(self.review_items)}`",
        ]
        if self.patches:
            lines.extend(["", "## Proposed patches", ""])
            for patch in self.patches[:20]:
                target = patch.column or "row"
                lines.append(
                    f"- `{patch.patch_id}` `{patch.operation}` `{target}` "
                    f"risk=`{patch.risk}` confidence=`{patch.confidence:.2f}`"
                )
            if len(self.patches) > 20:
                lines.append(f"- ... {len(self.patches) - 20} more patch(es)")
        return "\n".join(lines)

    def to_dbt_failures(self) -> pd.DataFrame:
        """Export patches in a shape that can be joined with dbt failure rows."""
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(columns=["unique_id", "column_name", "issue", "patch_id"])
        return pd.DataFrame({
            "unique_id": frame["row"],
            "column_name": frame["column"],
            "issue": frame["step"],
            "patch_id": frame["patch_id"],
        })

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        return "\n".join([
            f"freshdata repair plan (mode={self.mode!r})",
            f"  shape:   {self.before_shape} -> {self.after_shape}",
            f"  patches: {len(self.patches)}",
            f"  review:  {len(self.review_items)} item(s)",
            f"  source:  {self.source_fingerprint}",
        ])

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<RepairPlan: {len(self.patches)} patch(es), "
            f"mode={self.mode!r}, shape={self.before_shape}->{self.after_shape}>"
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


def _json_value(value: Any) -> Any:
    """Convert pandas/numpy scalar values into JSON-friendly Python values.

    Array-like inputs (numpy arrays, pandas Series) are handled before
    calling ``pd.isna`` to avoid ambiguous truth-value checks that raise
    DeprecationWarning in newer pandas/numpy.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_json_value(v) for v in value]
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    # Handle array-like inputs before calling pd.isna to avoid array
    # truth-value ambiguity.
    if isinstance(value, pd.Series):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError, TypeError):
            pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _values_equal(left: Any, right: Any) -> bool:
    left_missing = _is_missing_scalar(left)
    right_missing = _is_missing_scalar(right)
    if left_missing and right_missing:
        return True
    if left_missing != right_missing:
        return False
    try:
        return bool(left == right) and type(left) is type(right)
    except (TypeError, ValueError):
        return False


def _is_missing_scalar(value: Any) -> bool:
    if isinstance(value, (list, tuple, dict, np.ndarray, pd.Series)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _fingerprint(df: pd.DataFrame) -> str:
    try:
        payload = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    except TypeError:
        payload = df.astype(str).to_csv(index=True).encode()
    columns = "|".join(str(c) for c in df.columns).encode()
    shape = f"{df.shape[0]}x{df.shape[1]}".encode()
    return hashlib.sha256(payload + columns + shape).hexdigest()[:16]


def _action_for_column(report: CleanReport, column: str | None) -> Action | None:
    if column is not None:
        for action in reversed(report.actions):
            if action.column == column:
                return action
    for action in reversed(report.actions):
        if action.column is None and action.count:
            return action
    return None


def _patch_from_action(
    patch_id: str,
    operation: str,
    action: Action | None,
    *,
    row: Any | None = None,
    column: str | None = None,
    old_value: Any = None,
    new_value: Any = None,
) -> RepairPatch:
    return RepairPatch(
        patch_id=patch_id,
        operation=operation,
        row=row,
        column=column,
        old_value=old_value,
        new_value=new_value,
        step=action.step if action else "observed_diff",
        rationale=action.rationale if action else "",
        risk=action.risk if action else "low",
        confidence=action.confidence if action else 1.0,
    )


def _build_patches(
    before: pd.DataFrame,
    after: pd.DataFrame,
    report: CleanReport,
) -> tuple[RepairPatch, ...]:
    patches: list[RepairPatch] = []
    common_rows = before.index.intersection(after.index)
    common_cols = before.columns.intersection(after.columns)

    for row in before.index.difference(after.index):
        action = _action_for_column(report, None)
        patches.append(_patch_from_action(
            f"p{len(patches) + 1:06d}",
            "drop_row",
            action,
            row=row,
            old_value=before.loc[row].to_dict(),
        ))

    for column in before.columns.difference(after.columns):
        action = _action_for_column(report, str(column))
        patches.append(_patch_from_action(
            f"p{len(patches) + 1:06d}",
            "drop_column",
            action,
            column=str(column),
            old_value=before[column].tolist(),
        ))

    for column in after.columns.difference(before.columns):
        action = _action_for_column(report, str(column))
        patches.append(_patch_from_action(
            f"p{len(patches) + 1:06d}",
            "add_column",
            action,
            column=str(column),
            new_value=after[column].tolist(),
        ))

    for column in common_cols:
        action = _action_for_column(report, str(column))
        for row in common_rows:
            old = before.at[row, column]
            new = after.at[row, column]
            if _values_equal(old, new):
                continue
            patches.append(_patch_from_action(
                f"p{len(patches) + 1:06d}",
                "update_cell",
                action,
                row=row,
                column=str(column),
                old_value=old,
                new_value=new,
            ))
    return tuple(patches)


def _build_review_items(patches: tuple[RepairPatch, ...]) -> tuple[ReviewItem, ...]:
    items: list[ReviewItem] = []
    for patch in patches:
        if patch.risk == "low" and patch.confidence >= 0.9:
            continue
        reason = (
            f"{patch.risk}-risk repair"
            if patch.risk != "low"
            else f"confidence {patch.confidence:.2f} is below automatic threshold"
        )
        items.append(ReviewItem(
            patch_id=patch.patch_id,
            reason=reason,
            risk=patch.risk,
            confidence=patch.confidence,
            row=patch.row,
            column=patch.column,
            operation=patch.operation,
        ))
    return tuple(items)


def _empty_report(df: pd.DataFrame) -> CleanReport:
    return CleanReport(
        rows_before=len(df),
        rows_after=len(df),
        cols_before=df.shape[1],
        cols_after=df.shape[1],
        missing_before=int(df.isna().sum().sum()),
        missing_after=int(df.isna().sum().sum()),
    )


def _config_for_repair_mode(
    mode: str,
    config: CleanConfig | None,
    options: dict[str, object],
) -> CleanConfig:
    if mode not in REPAIR_MODES:
        raise ValueError(f"mode must be one of {REPAIR_MODES}, got {mode!r}")
    cfg = merge_options(config, **options)
    if mode == "repair_safe":
        return merge_options(
            cfg,
            strategy="conservative",
            impute=None,
            outliers=None,
            outlier_action=None,
        )
    if mode == "repair_aggressive":
        return merge_options(cfg, strategy="aggressive")
    return cfg


def build_repair_plan(
    df: pd.DataFrame,
    *,
    mode: str = "suggest",
    config: CleanConfig | None = None,
    **options: object,
) -> RepairPlan:
    """Build a repair artifact with patches, review items, and rollback data."""
    cfg = _config_for_repair_mode(mode, config, dict(options))
    before = df.copy(deep=True)
    if mode == "inspect":
        report = _empty_report(before)
        return RepairPlan(
            config=cfg,
            mode=mode,
            source_fingerprint=_fingerprint(before),
            before_shape=before.shape,
            after_shape=before.shape,
            report=report,
            _before=before,
            _after=before,
        )
    run_cfg = merge_options(cfg, verbose=False, preserve_original=True)
    after, report = run_pipeline(before, run_cfg)
    patches = _build_patches(before, after, report)
    review_items = _build_review_items(patches)
    return RepairPlan(
        config=cfg,
        mode=mode,
        source_fingerprint=_fingerprint(before),
        before_shape=before.shape,
        after_shape=after.shape,
        report=report,
        patches=patches,
        review_items=review_items,
        _before=before,
        _after=after.copy(deep=True),
    )


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
