"""Pluggable out-of-core execution backends for freshdata.

This package adds Polars (LazyFrame + streaming) and DuckDB (SQL + spill-to-disk)
execution paths alongside the in-memory pandas pipeline, so ``fd.clean`` can run
on larger-than-RAM data. The pandas backend remains the reference: any step a
native backend cannot run is delegated to it, so output is unchanged.

Public entry point: :func:`run_with_engine`, wired into :func:`freshdata.clean`
via its ``engine`` / ``output_format`` / ``engine_config`` keyword arguments.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ._base import ExecutionEngine
from ._config import (
    FALLBACK_POLICIES,
    EngineConfig,
    EngineSelector,
    FallbackError,
    FallbackWarning,
    enforce_fallback_policy,
)
from ._metadata import ColumnMetadata, MetadataScanner
from ._plan import NativePlan, PlanGenerator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..config import CleanConfig

__all__ = [
    "EngineConfig",
    "EngineSelector",
    "ExecutionEngine",
    "ColumnMetadata",
    "MetadataScanner",
    "NativePlan",
    "PlanGenerator",
    "FALLBACK_POLICIES",
    "FallbackError",
    "FallbackWarning",
    "enforce_fallback_policy",
    "run_with_engine",
]


def _is_spark_frame(frame: Any) -> bool:
    from ._lazy import has_pyspark

    if not has_pyspark():
        return False
    from pyspark.sql import DataFrame as SparkDataFrame

    return isinstance(frame, SparkDataFrame)


def _convert_output(frame: Any, output_format: str) -> Any:
    """Convert a backend-native frame to the requested output format."""
    import pandas as pd

    is_spark = _is_spark_frame(frame)

    # Native, un-materialized handles: hand the backend's own lazy object back
    # untouched. The backend is responsible for *not* having collected/fetched
    # it (see the DuckDB/Polars engines). We never silently materialize here.
    #
    # A pandas frame at this point means the backend transparently fell back to
    # the pandas pipeline (e.g. the balanced decision engine, which only runs on
    # pandas). That fallback is already disclosed on the report
    # (``fallback_events`` + ``backend="pandas"``), so we return the materialized
    # frame rather than raising — the caller can read the report to see why the
    # native handle wasn't available.
    if output_format == "duckdb":
        try:
            import duckdb
        except ImportError:  # pragma: no cover - guarded upstream
            duckdb = None  # type: ignore[assignment]
        if duckdb is not None and isinstance(frame, duckdb.DuckDBPyRelation):
            return frame
        return frame  # disclosed pandas fallback
    if output_format == "polars-lazy":
        from ._lazy import require_polars

        pl = require_polars()
        if isinstance(frame, pl.LazyFrame):
            return frame
        if isinstance(frame, pl.DataFrame):
            return frame.lazy()
        return frame  # disclosed pandas fallback

    if output_format == "spark":
        if is_spark:
            return frame
        from ._lazy import require_pyspark

        sql = require_pyspark()
        session = sql.SparkSession.builder.appName("freshdata").getOrCreate()
        pdf = frame if isinstance(frame, pd.DataFrame) else frame.to_pandas()
        return session.createDataFrame(pdf)

    # All non-spark formats operate on pandas/polars; collapse Spark frames first.
    if is_spark:
        frame = frame.toPandas()

    is_pandas = isinstance(frame, pd.DataFrame)

    if output_format == "pandas":
        if is_pandas:
            return frame
        return frame.to_pandas()

    if output_format == "polars":
        from ._lazy import require_polars

        pl = require_polars()
        if is_pandas:
            return pl.from_pandas(frame)
        return frame  # already polars

    if output_format == "arrow":
        from ._lazy import require_pyarrow

        require_pyarrow()
        if is_pandas:
            import pyarrow as pa

            return pa.Table.from_pandas(frame, preserve_index=False)
        return frame.to_arrow()  # polars

    raise ValueError(f"unknown output_format {output_format!r}")


def run_with_engine(
    source: Any,
    config: CleanConfig,
    *,
    engine: str = "pandas",
    output_format: str = "pandas",
    engine_config: EngineConfig | None = None,
    return_report: bool = False,
) -> Any:
    """Clean *source* through the selected backend.

    *config* is the usual :class:`~freshdata.CleanConfig` (the cleaning
    decisions). *engine* / *output_format* / *engine_config* control execution.
    Returns the cleaned frame, or ``(cleaned, report)`` when ``return_report``.
    """
    if engine_config is None:
        engine_config = EngineConfig(engine=engine, output_format=output_format)

    requested = engine_config.engine
    resolved = engine_config.engine
    if resolved == "auto":
        resolved = EngineSelector.select(source, engine_config)
        engine_config = replace(engine_config, engine=resolved)

    # Semantic cleaning is scored on the pandas reference path. On a native
    # engine we keep the frame native and run the semantic stage over a
    # *natively extracted* distinct table (see freshdata.semantic.native) — no
    # full-frame materialization. Only a non-default backend / learned profile,
    # which the native distinct path does not reproduce byte-for-byte, still
    # routes the whole clean through pandas with a disclosed fallback.
    if (
        config.semantic_enabled
        and resolved != "pandas"
        and not _native_semantic_supported(config)
    ):
        reason = (
            "semantic cleaning with a non-default backend requires the pandas "
            "in-memory path"
        )
        enforce_fallback_policy(engine_config, resolved, "semantic", reason)
        from ..cleaner import run_pipeline
        from .backends._pandas import materialize_to_pandas

        frame = materialize_to_pandas(source)
        cleaned, report = run_pipeline(frame, config)
        report.backend = "pandas"
        report.record_fallback(resolved, "semantic", reason)
        result = _convert_output(cleaned, engine_config.output_format)
        _finish_report(report, requested, "pandas", result)
        return (result, report) if return_report else result

    backend = EngineSelector.get_engine(resolved, engine_config)
    cleaned_native, report = backend.execute(source, config, engine_config)
    # Run the semantic stage over the native handle unless the backend already
    # fell back to pandas (in which case semantic already ran in that pipeline).
    if config.semantic_enabled and report.backend == resolved and resolved in ("polars", "duckdb"):
        from ..semantic.native import run_semantic_native

        cleaned_native = run_semantic_native(cleaned_native, config, report, engine=resolved)
    result = _convert_output(cleaned_native, engine_config.output_format)
    _finish_report(report, requested, resolved, result)
    return (result, report) if return_report else result


def _peak_rss_bytes() -> int | None:
    """Process peak RSS in bytes, or ``None`` where unsupported (Windows)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _finish_report(report: Any, requested: str, resolved: str, result: Any) -> None:
    """Stamp execution-honesty fields on the report after output conversion."""
    import pandas as pd

    report.requested_backend = requested
    if report.backend is None:
        report.backend = resolved
    report.peak_memory = _peak_rss_bytes()
    if isinstance(result, pd.DataFrame):
        report.rows_materialized = len(result)
    else:
        try:
            import polars as pl

            if isinstance(result, pl.DataFrame):
                report.rows_materialized = result.height
        except ImportError:
            pass
        try:
            import pyarrow as pa

            if isinstance(result, pa.Table):
                report.rows_materialized = result.num_rows
        except ImportError:
            pass
    # Native handles (LazyFrame / DuckDBPyRelation) and Spark frames stay None:
    # nothing was pulled into memory / counting would trigger a job.


def _native_semantic_supported(config: CleanConfig) -> bool:
    from ..semantic.native import can_run_native

    return can_run_native(config, memory=None, profile=None)
