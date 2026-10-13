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
    from ..config import CleanConfig
    from ._metadata import ColumnMetadata


def record_extension_dtype_changes(
    source: Any, result: Any, config: CleanConfig, report: CleanReport
) -> None:
    """Disclose native pandas-output changes without recasting data or handles."""
    backend = report.backend
    if backend is None or backend not in ("polars", "duckdb"):
        return
    import pandas as pd
    from pandas.api.types import is_dtype_equal, is_extension_array_dtype

    if not isinstance(source, pd.DataFrame) or not isinstance(result, pd.DataFrame):
        return
    from ._plan import PlanGenerator

    source_dtypes = source.dtypes
    if not any(is_extension_array_dtype(dtype) for dtype in source_dtypes):
        return
    original_names = list(source.columns)
    if backend == "duckdb":
        import duckdb

        # Registration deduplicates case-insensitive names before the plan's
        # renames. Inspect relation metadata without fetching any rows.
        native_names = duckdb.from_df(source).columns
        plan_names = original_names
    else:
        native_names = [str(label) for label in original_names]
        plan_names = native_names
    rename_map = PlanGenerator(config, backend=backend).plan(plan_names).rename_map
    # Native stages retain source order and append derived columns. Match each
    # retained source position to the actual final label: a SQL projection can
    # disambiguate registration/normalization collisions a second time.
    retained = [
        position for position, name in enumerate(native_names)
        if rename_map.get(name, name) not in report.columns_dropped
    ]
    for output_position, position in enumerate(retained):
        if output_position >= len(result.columns):
            break
        original_dtype = source_dtypes.iloc[position]
        if not is_extension_array_dtype(original_dtype):
            continue
        output_label = str(result.columns[output_position])
        output_dtype = result.iloc[:, output_position].dtype
        if not is_dtype_equal(original_dtype, output_dtype):
            # Categorical repr includes actual category values. Report dtype
            # names and storage, not a domain that can contain sensitive data.
            before = str(original_dtype)
            after = str(output_dtype)
            if isinstance(original_dtype, pd.StringDtype):
                before = f"string[{original_dtype.storage}]"
            if isinstance(output_dtype, pd.StringDtype):
                after = f"string[{output_dtype.storage}]"
            if isinstance(original_dtype, pd.SparseDtype):
                before = f"Sparse[{original_dtype.subtype}]"
            report.record_backend_difference(
                backend,
                "output_dtype",
                f"pandas output changed the input extension dtype/metadata "
                f"from {before} to {after}",
                column=output_label,
            )


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

    # arrow table (DuckDB fetches Arrow directly for output_format="arrow")
    try:
        import pyarrow as pa

        if isinstance(frame, pa.Table):
            missing = sum(column.null_count for column in frame.columns)
            return frame.num_rows, frame.num_columns, int(missing), int(frame.nbytes)
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


def zero_column_frame(backend: str, output_format: str, report: CleanReport) -> Any:
    """pandas result for a native run that dropped every column as empty.

    The pandas pipeline keeps the row count on a zero-column frame, so the native
    backends return the same. Polars and Arrow frames cannot hold rows without
    columns; for those output formats the difference is recorded on the report.
    """
    import pandas as pd

    if output_format != "pandas":
        report.record_backend_difference(
            backend, "drop_empty_columns",
            f"every column was empty; a zero-column {output_format} result cannot carry "
            f"the row count ({report.rows_before} row(s) in pandas output)",
        )
    return pd.DataFrame(index=pd.RangeIndex(report.rows_before))


def finalize_report_native(report: CleanReport, started: float) -> CleanReport:
    """Finalize a report whose result was returned as a native, un-materialized
    handle (a DuckDB relation or a Polars ``LazyFrame``).

    The "after" counts are deliberately left uncomputed — measuring them would
    force a scan/collect and defeat the out-of-core path the caller asked for.
    """
    report.materialized = False
    report.duration_seconds = time.perf_counter() - started
    return report
