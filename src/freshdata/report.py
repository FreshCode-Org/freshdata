"""Structured record of everything :func:`freshdata.clean` did.

Trust is the core feature of an auto-cleaner: every transformation is recorded
as an :class:`Action` — with a rationale, a risk level, and a confidence score
when it came from the decision engine — so users can audit exactly what
changed, how much, and why. Columns that were deliberately *not* touched get
an action too, so remaining NaNs are always explained.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ._util import format_bytes

#: Valid risk levels, in increasing order of severity.
RISK_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class Action:
    """One transformation (or deliberate non-transformation) of the data.

    Attributes
    ----------
    step:
        Machine-readable step name, e.g. ``"fix_dtypes"`` or ``"missing"``.
    column:
        Column the action applied to, or ``None`` for table-level actions.
    description:
        Human-readable summary of what happened.
    count:
        Number of cells or rows affected (0 for informational notes).
    rationale:
        Why the decision engine chose this action ("" for non-engine steps).
    risk:
        "low", "medium", or "high" — how likely the action is to need review.
    confidence:
        Engine confidence in the decision, in [0, 1] (1.0 for non-engine steps,
        which are deterministic representation repairs).
    """

    step: str
    column: str | None
    description: str
    count: int = 0
    rationale: str = ""
    risk: str = "low"
    confidence: float = 1.0
    model_id: str = ""

    def __str__(self) -> str:
        target = f"{self.column!r}: " if self.column is not None else ""
        return f"[{self.step}] {target}{self.description}"


@dataclass
class CleanReport:
    """Everything one :func:`freshdata.clean` run did, in order.

    Iterable and sized: ``len(report)`` is the number of actions, and
    ``for action in report`` walks them in execution order. ``bool(report)``
    is ``True`` iff anything was changed.

    Beyond the action log, the report carries a cleaning summary (missing
    cells before/after, duplicates removed, outliers handled, columns
    dropped/imputed/preserved), engine warnings for risky columns, and
    recommendations for manual review.
    """

    actions: list[Action] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    memory_before: int = 0
    memory_after: int = 0
    duration_seconds: float = 0.0
    missing_before: int = 0
    missing_after: int = 0
    duplicates_removed: int = 0
    outliers_handled: int = 0
    columns_dropped: list[str] = field(default_factory=list)
    columns_imputed: list[str] = field(default_factory=list)
    columns_preserved: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def add(self, step: str, description: str, *, column: str | None = None,
            count: int = 0, rationale: str = "", risk: str = "low",
            confidence: float = 1.0, model_id: str = "") -> None:
        """Record one action (internal; called by the pipeline)."""
        self.actions.append(Action(step=step, column=column, description=description,
                                   count=int(count), rationale=rationale, risk=risk,
                                   confidence=float(confidence), model_id=model_id))

    def add_warning(self, message: str) -> None:
        """Record a warning about a risky column or decision (internal)."""
        if message not in self.warnings:
            self.warnings.append(message)

    def add_recommendation(self, message: str) -> None:
        """Record a suggestion for manual review (internal)."""
        if message not in self.recommendations:
            self.recommendations.append(message)

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __bool__(self) -> bool:
        return any(a.count for a in self.actions)

    @property
    def cells_changed(self) -> int:
        """Total affected cells/rows summed across all actions."""
        return sum(a.count for a in self.actions)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, suitable for JSON serialization or logging."""
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "cols_before": self.cols_before,
            "cols_after": self.cols_after,
            "memory_before": self.memory_before,
            "memory_after": self.memory_after,
            "duration_seconds": self.duration_seconds,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "duplicates_removed": self.duplicates_removed,
            "outliers_handled": self.outliers_handled,
            "columns_dropped": list(self.columns_dropped),
            "columns_imputed": list(self.columns_imputed),
            "columns_preserved": list(self.columns_preserved),
            "warnings": list(self.warnings),
            "recommendations": list(self.recommendations),
            "actions": [
                {"step": a.step, "column": a.column, "description": a.description,
                 "count": a.count, "rationale": a.rationale, "risk": a.risk,
                 "confidence": a.confidence, "model_id": a.model_id}
                for a in self.actions
            ],
        }

    def to_frame(self) -> pd.DataFrame:
        """One row per action, as a DataFrame — convenient for notebooks."""
        return pd.DataFrame(
            [(a.step, a.column, a.description, a.count, a.rationale, a.risk,
              a.confidence, a.model_id)
             for a in self.actions],
            columns=["step", "column", "description", "count", "rationale", "risk",
                     "confidence", "model_id"],
        )

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        d_rows = self.rows_after - self.rows_before
        d_cols = self.cols_after - self.cols_before
        lines = [
            "freshdata clean report",
            f"  rows:    {self.rows_before:,} -> {self.rows_after:,} ({d_rows:+,})",
            f"  columns: {self.cols_before:,} -> {self.cols_after:,} ({d_cols:+,})",
            f"  missing: {self.missing_before:,} -> {self.missing_after:,} cell(s)",
            f"  memory:  {format_bytes(self.memory_before)} -> "
            f"{format_bytes(self.memory_after)}",
            f"  time:    {self.duration_seconds:.3f}s",
        ]
        facts = []
        if self.duplicates_removed:
            facts.append(f"{self.duplicates_removed} duplicate row(s) removed")
        if self.outliers_handled:
            facts.append(f"{self.outliers_handled} outlier(s) handled")
        if self.columns_dropped:
            facts.append(f"dropped: {', '.join(self.columns_dropped)}")
        if self.columns_imputed:
            facts.append(f"imputed: {', '.join(self.columns_imputed)}")
        if self.columns_preserved:
            facts.append(f"preserved: {', '.join(self.columns_preserved)}")
        if facts:
            lines.append("  engine:  " + "; ".join(facts))
        if self.actions:
            lines.append(f"  actions ({len(self.actions)}):")
            lines.extend(f"    - {a}" for a in self.actions)
        else:
            lines.append("  actions: none — data was already clean")
        if self.warnings:
            lines.append(f"  warnings ({len(self.warnings)}):")
            lines.extend(f"    ! {w}" for w in self.warnings)
        if self.recommendations:
            lines.append(f"  review ({len(self.recommendations)}):")
            lines.extend(f"    ? {r}" for r in self.recommendations)
        return "\n".join(lines)

    def brief(self) -> str:
        """Compact summary for ``verbose=True`` console output."""
        line = (
            f"freshdata: rows {self.rows_before:,}->{self.rows_after:,}, "
            f"cols {self.cols_before}->{self.cols_after}, "
            f"missing {self.missing_before:,}->{self.missing_after:,}"
        )
        extras = []
        if self.duplicates_removed:
            extras.append(f"{self.duplicates_removed} dup(s) removed")
        if self.outliers_handled:
            extras.append(f"{self.outliers_handled} outlier(s) handled")
        if self.columns_dropped:
            extras.append(f"dropped {len(self.columns_dropped)} column(s)")
        if extras:
            line += " (" + ", ".join(extras) + ")"
        for w in self.warnings:
            line += f"\n  warning: {w}"
        return line

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<CleanReport: {len(self.actions)} actions, "
            f"rows {self.rows_before:,}->{self.rows_after:,}, "
            f"cols {self.cols_before}->{self.cols_after}>"
        )
