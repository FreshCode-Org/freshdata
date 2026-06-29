"""Build a :class:`~freshdata.CleanReport` from a native backend run.

Native backends record actions with ``report.add(...)`` using the *same* step
names the pandas pipeline emits (``column_names``, ``strip_whitespace``,
``normalize_sentinels``, ``drop_empty_columns``, ``drop_empty_rows``,
``drop_duplicates``, ``impute``, ``outliers``) so downstream consumers
(``freshdata.compliance``, ``freshdata.integrations``) keep working unchanged.
These helpers handle the before/after bookkeeping run_pipeline does.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ..report import CleanReport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._metadata import ColumnMetadata


def init_report(metadata: list[ColumnMetadata], memory_before: int = 0) -> CleanReport:
    """Initialise a report's "before" counts from cheap column metadata."""
    rows_before = metadata[0].row_count if metadata else 0
    missing_before = sum(m.row_count - m.non_null_count for m in metadata)
    return CleanReport(
        rows_before=rows_before,
        cols_before=len(metadata),
        memory_before=memory_before,
        missing_before=missing_before,
    )


def _frame_stats(frame: Any) -> tuple[int, int, int, int]:
    """Return ``(rows, cols, missing_cells, memory_bytes)`` for a cleaned frame."""
    import pandas as pd

    if isinstance(frame, pd.DataFrame):
        rows, cols = frame.shape
        missing = int(frame.isna().sum().sum())
        memory = int(frame.memory_usage(deep=True).sum())
        return rows, cols, missing, memory

    # polars frame
    try:
        import polars as pl

        if isinstance(frame, pl.DataFrame):
            rows, cols = frame.height, frame.width
            nulls = frame.null_count()
            missing = int(sum(nulls.row(0))) if cols else 0
            memory = int(frame.estimated_size())
            return rows, cols, missing, memory
    except ImportError:  # pragma: no cover
        pass

    raise TypeError(f"cannot compute stats for {type(frame).__name__}")


def finalize_report(report: CleanReport, cleaned: Any, started: float) -> CleanReport:
    """Fill the report's "after" counts and duration from the cleaned frame."""
    rows, cols, missing, memory = _frame_stats(cleaned)
    report.rows_after = rows
    report.cols_after = cols
    report.missing_after = missing
    report.memory_after = memory
    report.duration_seconds = time.perf_counter() - started
    return report


def finalize_report_native(report: CleanReport, started: float) -> CleanReport:
    """Finalize a report whose result was returned as a native, un-materialized
    handle (a DuckDB relation or a Polars ``LazyFrame``).

    The "after" counts are deliberately left uncomputed — measuring them would
    force a scan/collect and defeat the out-of-core path the caller asked for.
    """
    report.materialized = False
    report.duration_seconds = time.perf_counter() - started
    return report
