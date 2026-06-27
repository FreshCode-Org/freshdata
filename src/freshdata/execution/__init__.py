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
from ._config import EngineConfig, EngineSelector
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

    resolved = engine_config.engine
    if resolved == "auto":
        resolved = EngineSelector.select(source, engine_config)
        engine_config = replace(engine_config, engine=resolved)

    backend = EngineSelector.get_engine(resolved, engine_config)
    cleaned_native, report = backend.execute(source, config, engine_config)
    result = _convert_output(cleaned_native, engine_config.output_format)
    return (result, report) if return_report else result
