"""Engine selection and execution configuration.

:class:`EngineConfig` controls *how* a clean runs (which backend, output format,
streaming, memory limits) — never *what* the clean decides. The decision logic
stays in :class:`~freshdata.CleanConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._lazy import has_polars, has_pyspark

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._base import ExecutionEngine

#: Valid backend names (``"auto"`` resolves to one of the concrete backends).
ENGINE_NAMES = ("pandas", "polars", "duckdb", "spark", "auto")
#: Valid output formats for the cleaned frame.
OUTPUT_FORMATS = ("pandas", "polars", "arrow", "spark")


@dataclass
class EngineConfig:
    """Execution behaviour for a single :func:`freshdata.clean` call."""

    engine: str = "pandas"
    output_format: str = "pandas"
    streaming: bool = True
    memory_limit_gb: float = 8.0
    temp_directory: str = "/tmp/freshdata_spill"
    polars_n_threads: int | None = None
    duckdb_threads: int | None = None
    #: Number of shuffle partitions for the Spark backend (``None`` = Spark default).
    spark_shuffle_partitions: int | None = None
    #: An existing ``SparkSession`` to reuse; ``None`` uses the active/default one.
    spark_session: Any = None
    #: ``engine="auto"`` uses polars above this row count, duckdb above the next.
    row_count_auto_threshold_polars: int = 10_000_000
    row_count_auto_threshold_duckdb: int = 100_000_000

    def __post_init__(self) -> None:
        if self.engine not in ENGINE_NAMES:
            raise ValueError(f"engine must be one of {ENGINE_NAMES}, got {self.engine!r}")
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {self.output_format!r}"
            )


def _is_parquet_path(source: Any) -> bool:
    return isinstance(source, str) and source.lower().endswith((".parquet", ".pq"))


def _is_tabular_file_path(source: Any) -> bool:
    return isinstance(source, str) and source.lower().endswith(
        (".parquet", ".pq", ".csv", ".ipc", ".feather", ".arrow")
    )


class EngineSelector:
    """Resolve ``engine="auto"`` and construct backend instances lazily."""

    @staticmethod
    def select(source: Any, config: EngineConfig) -> str:
        """Return a concrete backend name for *source* under *config*.

        Spark frames stay on Spark; file paths go to DuckDB (it reads them
        without loading into Python); polars frames stay on polars; Arrow tables
        prefer polars (zero-copy) else duckdb; pandas frames are sized to choose
        pandas / polars / duckdb by row count.
        """
        if has_pyspark():
            from pyspark.sql import DataFrame as SparkDataFrame

            if isinstance(source, SparkDataFrame):
                return "spark"

        if _is_parquet_path(source):
            return "duckdb"
        if _is_tabular_file_path(source):
            return "duckdb"

        if has_polars():
            import polars as pl

            if isinstance(source, (pl.DataFrame, pl.LazyFrame)):
                return "polars"

        # arrow tables / record batches: polars handles them zero-copy if present
        try:
            import pyarrow as pa

            if isinstance(source, (pa.Table, pa.RecordBatch)):
                return "polars" if has_polars() else "duckdb"
        except ImportError:
            pass

        # duckdb relation
        try:
            import duckdb

            if isinstance(source, duckdb.DuckDBPyRelation):
                return "duckdb"
        except ImportError:
            pass

        try:
            import pandas as pd

            if isinstance(source, pd.DataFrame):
                n = len(source)
                if n < config.row_count_auto_threshold_polars:
                    return "pandas"
                if n < config.row_count_auto_threshold_duckdb:
                    return "polars" if has_polars() else "duckdb"
                return "duckdb"
        except ImportError:  # pragma: no cover - pandas is a hard dependency
            pass

        return "pandas"

    @staticmethod
    def get_engine(name: str, config: EngineConfig) -> ExecutionEngine:
        """Return a backend instance, importing the backend module lazily."""
        if name == "pandas":
            from .backends._pandas import PandasEngine

            return PandasEngine()
        if name == "polars":
            from .backends._polars import PolarsEngine

            return PolarsEngine()
        if name == "duckdb":
            from .backends._duckdb import DuckDBEngine

            return DuckDBEngine()
        if name == "spark":
            from .backends._spark import SparkEngine

            return SparkEngine()
        raise ValueError(f"unknown engine {name!r}; expected one of {ENGINE_NAMES}")
