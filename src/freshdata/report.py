# src/freshdata/report.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CleanAction:
    step: str
    column: str | None
    description: str
    count: int = 0
    risk: str = "low"


@dataclass
class CleanReport:
    engine_used: str = "pandas"
    rows_before: int = 0
    rows_after: int = 0
    duplicates_removed: int = 0
    values_imputed: int = 0
    outliers_clipped: int = 0
    dtype_changes: dict[str, tuple[str, str]] = field(default_factory=dict)
    skipped_columns: list[dict[str, Any]] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    actions: list[CleanAction] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Phase 3 Streaming Telemetry
    streaming_requested: bool = False
    streaming_executed_fully: bool = False
    non_streaming_stages: list[str] = field(default_factory=list)
    row_order_deterministic: bool = True
    plan_explain: str | None = None

    def summary(self) -> str:
        return (
            f"CleanReport(engine={self.engine_used}, "
            f"rows={self.rows_before}->{self.rows_after}, "
            f"duplicates_removed={self.duplicates_removed}, "
            f"values_imputed={self.values_imputed}, "
            f"dtype_changes={len(self.dtype_changes)}, "
            f"streaming_full={self.streaming_executed_fully})"
        )
