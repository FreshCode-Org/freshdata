"""Reverse-engineering helpers: explain what clean() did and why."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from pandas.api.types import is_numeric_dtype

from .adapters.polars import to_pandas
from .api import infer_roles
from .cleaner import run_pipeline
from .config import CleanConfig, merge_options
from .engine.context import build_contexts
from .render.mixins import HtmlReprMixin
from .report import Action, CleanReport


def _column_stats(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        entry: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "null_pct": round(float(s.isna().mean()), 4),
            "nunique": int(s.nunique(dropna=True)),
        }
        if is_numeric_dtype(s):
            nonnull = s.dropna()
            if len(nonnull):
                entry["min"] = float(nonnull.min())
                entry["max"] = float(nonnull.max())
        stats[str(col)] = entry
    return stats


def _cell_changes(before: pd.DataFrame, after: pd.DataFrame) -> dict[str, int]:
    """Count cells whose value genuinely changed, per shared column.

    When cleaning added or removed rows, the frames are first aligned on
    their shared index labels -- row additions/removals are reported as row
    counts, not as cell edits.  Cells missing on both sides are unchanged; a
    value<->missing transition is a change; a dtype conversion alone does not
    mark untouched values as changed.  If either index carries duplicate
    labels, alignment is ambiguous and the conservative whole-column count is
    kept.  Columns that exist only in ``after`` count all their cells.
    """
    changes: dict[str, int] = {}
    shared = [c for c in after.columns if c in before.columns]
    aligned_before, aligned_after = before, after
    if len(before) != len(after):
        if before.index.is_unique and after.index.is_unique:
            common = before.index.intersection(after.index)
            aligned_before = before.loc[common]
            aligned_after = after.loc[common]
        else:
            for col in after.columns:
                changes[col] = len(after)
            return changes
    for col in shared:
        left = aligned_before[col]
        right = aligned_after[col]
        lna = left.isna()
        rna = right.isna()
        try:
            # Kleene logic keeps this NA-free: (left != right) is only <NA>
            # where a side is missing, and `& False` collapses that to False.
            valid_diff = (left != right) & ~lna & ~rna
            changed = int(((lna != rna) | valid_diff).sum())
        except (TypeError, ValueError):
            changed = sum(
                1
                for a, b in zip(left, right)
                if pd.isna(a) != pd.isna(b)
                or (not pd.isna(a) and not pd.isna(b) and a != b)
            )
        changes[col] = changed
    for col in after.columns:
        if col not in before.columns:
            changes[col] = len(after)
    return changes



def _narratives(
    contexts: dict,
    actions: list[Action],
    *,
    strategy: str,
) -> list[str]:
    lines: list[str] = []
    by_col: dict[str, list[Action]] = defaultdict(list)
    for action in actions:
        if action.column:
            by_col[action.column].append(action)

    for col, ctx in sorted(contexts.items()):
        col_actions = by_col.get(col, [])
        engine_actions = [a for a in col_actions if a.rationale]
        if engine_actions:
            primary = engine_actions[0]
            miss_pct = round(ctx.missing_ratio * 100, 1)
            lines.append(
                f"`{col}`: {primary.description} "
                f"(role={ctx.role}, missing={miss_pct}%, strategy={strategy!r})"
            )
        elif ctx.missing_ratio > 0 and ctx.role in ("target", "id", "text"):
            lines.append(
                f"`{col}`: preserved despite {ctx.missing_ratio:.1%} missing "
                f"(role={ctx.role})"
            )
    return lines


def _narrative_contexts(
    df: pd.DataFrame,
    config: CleanConfig,
    actions: list[Action],
) -> dict:
    """Profile only columns that can contribute an explanation narrative."""
    needed = {
        action.column
        for action in actions
        if action.column is not None and action.rationale
    }
    needed.update(
        str(col) for col, has_missing in df.isna().any().items() if has_missing
    )
    columns = [col for col in df.columns if str(col) in needed]
    return build_contexts(df, config, columns=columns) if columns else {}


@dataclass
class ExplainReport(HtmlReprMixin):
    """Structured explanation of a clean() run."""

    _render_kind = "explain"

    strategy: str
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    before_stats: dict[str, dict[str, Any]]
    after_stats: dict[str, dict[str, Any]]
    cell_changes: dict[str, int]
    actions_by_step: dict[str, list[dict[str, Any]]]
    narratives: list[str]
    report: CleanReport
    roles: pd.DataFrame = field(repr=False)

    def summary(self) -> str:
        lines = [
            f"freshdata explain (strategy={self.strategy!r})",
            f"  shape: {self.rows_before}x{self.cols_before} -> "
            f"{self.rows_after}x{self.cols_after}",
            f"  missing: {self.report.missing_before} -> {self.report.missing_after}",
            f"  actions: {len(self.report)} steps logged",
        ]
        if self.narratives:
            lines.append("  decisions:")
            lines.extend(f"    - {n}" for n in self.narratives[:12])
            if len(self.narratives) > 12:
                lines.append(f"    … and {len(self.narratives) - 12} more")
        if self.report.warnings:
            lines.append("  warnings:")
            lines.extend(f"    - {w}" for w in self.report.warnings[:5])
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """One row per column: before/after dtype and changed-cell count."""
        rows = []
        for col in sorted(set(self.before_stats) | set(self.after_stats)):
            before = self.before_stats.get(col, {})
            after = self.after_stats.get(col, {})
            rows.append({
                "column": col,
                "before_dtype": before.get("dtype"),
                "after_dtype": after.get("dtype"),
                "changed_cells": self.cell_changes.get(col, 0),
            })
        return pd.DataFrame(
            rows, columns=["column", "before_dtype", "after_dtype", "changed_cells"]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "before_stats": self.before_stats,
            "after_stats": self.after_stats,
            "cell_changes": self.cell_changes,
            "actions_by_step": self.actions_by_step,
            "narratives": self.narratives,
            "warnings": list(self.report.warnings),
            "recommendations": list(getattr(self.report, "recommendations", []) or []),
        }


def explain_clean(
    df: pd.DataFrame,
    *,
    strategy: str = "balanced",
    config: CleanConfig | None = None,
    **options: object,
) -> ExplainReport:
    """Run clean() and return a structured before/after explanation."""
    cfg = merge_options(config, strategy=strategy, **options)
    df = to_pandas(df)  # accept polars frames like the other public entry points
    before_stats = _column_stats(df)
    cleaned, report = run_pipeline(df, cfg)

    roles_df = infer_roles(df, config=cfg)
    actions = list(report)

    actions_by_step: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        actions_by_step[action.step].append({
            "column": action.column,
            "description": action.description,
            "count": action.count,
            "rationale": action.rationale,
            "risk": action.risk,
            "confidence": round(action.confidence, 4),
            "model_id": action.model_id,
        })

    post_contexts = _narrative_contexts(cleaned, cfg, actions)
    return ExplainReport(
        strategy=cfg.strategy,
        rows_before=len(df),
        rows_after=len(cleaned),
        cols_before=df.shape[1],
        cols_after=cleaned.shape[1],
        before_stats=before_stats,
        after_stats=_column_stats(cleaned),
        cell_changes=_cell_changes(df, cleaned),
        actions_by_step=dict(actions_by_step),
        narratives=_narratives(post_contexts, actions, strategy=cfg.strategy),
        report=report,
        roles=roles_df,
    )
