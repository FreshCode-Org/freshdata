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
ENGINE_NAMES = ("pandas", "polars", "duckdb", "spark", "freshcore", "auto")
#: Valid output formats for the cleaned frame.
#:
#: The first four **materialize** the whole cleaned result into memory before
#: returning it (``pandas``/``polars``/``spark`` eager frames, an Arrow table).
#: The last two are **native, un-materialized handles** for honest out-of-core
#: work: ``"duckdb"`` returns the un-fetched ``DuckDBPyRelation`` and
#: ``"polars-lazy"`` returns the uncollected ``LazyFrame``. With those you, the
#: caller, decide when (and whether) to pull rows into RAM — freshdata never
#: silently calls ``.fetchdf()`` / ``.collect()`` behind your back.
OUTPUT_FORMATS = ("pandas", "polars", "arrow", "spark", "duckdb", "polars-lazy")

#: Output formats that pull the entire result into memory before returning.
MATERIALIZING_FORMATS = frozenset({"pandas", "polars", "arrow", "spark"})
#: Output formats that hand back a native, lazy/streaming handle instead.
NATIVE_HANDLE_FORMATS = frozenset({"duckdb", "polars-lazy"})
#: What to do when a native backend must delegate to the pandas reference:
#: ``"allow"`` (record silently on the report), ``"warn"`` (also emit a
#: :class:`FallbackWarning`), ``"error"`` (raise :class:`FallbackError` before
#: any pandas work runs — the strict out-of-core guarantee).
FALLBACK_POLICIES = ("allow", "warn", "error")


class FallbackWarning(UserWarning):
    """A native backend delegated work to the pandas reference implementation."""


class FallbackError(RuntimeError):
    """Raised under ``fallback_policy="error"`` when a native backend would
    have delegated to pandas. The message names the exact trigger."""


def enforce_fallback_policy(
    engine_config: EngineConfig, backend: str, step: str, reason: str
) -> None:
    """Apply the configured fallback policy for a pandas delegation.

    Called by native backends *before* they run the pandas pipeline, so
    ``"error"`` prevents the materialization instead of reporting it after
    the fact. ``"allow"`` is a no-op (the event is still recorded on the
    report by the caller).
    """
    policy = engine_config.fallback_policy
    if policy == "allow":
        return
    message = (
        f"freshdata {backend} backend fell back to pandas at step {step!r}: {reason}. "
        'strategy="conservative" with default options is the fully native path; '
        "see docs/fallback-matrix.md for what each backend runs natively."
    )
    if policy == "warn":
        import warnings

        warnings.warn(message, FallbackWarning, stacklevel=3)
        return
    raise FallbackError(message)


def materializes(output_format: str) -> bool:
    """Return ``True`` if *output_format* loads the whole result into memory."""
    return output_format in MATERIALIZING_FORMATS


@dataclass
class EngineConfig:
    """Execution behaviour for a single :func:`freshdata.clean` call."""

    engine: str = "pandas"
    output_format: str = "pandas"
    streaming: bool = True
    #: When ``streaming`` is on, keep exact full-row deduplication streaming-safe
    #: by *not* forcing ``maintain_order`` on the Polars backend (the order-
    #: preserving path materializes and defeats streaming). Set ``False`` to opt
    #: into order-preserving dedup; freshdata then warns that it materializes.
    streaming_dedup: bool = True
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
    #: What happens when the requested native backend must delegate to pandas:
    #: ``"allow"`` (default, recorded on the report), ``"warn"``
    #: (:class:`FallbackWarning`), or ``"error"`` (:class:`FallbackError` raised
    #: before any pandas materialization — no silent full-frame pull into RAM).
    #: Governs *unrequested* pandas execution only; it is inert when
    #: ``engine="pandas"`` was asked for.
    fallback_policy: str = "allow"

    def __post_init__(self) -> None:
        if self.engine not in ENGINE_NAMES:
            raise ValueError(f"engine must be one of {ENGINE_NAMES}, got {self.engine!r}")
        if self.output_format not in OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {OUTPUT_FORMATS}, got {self.output_format!r}"
            )
        if self.fallback_policy not in FALLBACK_POLICIES:
            raise ValueError(
                f"fallback_policy must be one of {FALLBACK_POLICIES}, "
                f"got {self.fallback_policy!r}"
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
        if name == "freshcore":
            from .backends._freshcore import FreshCoreEngine

            return FreshCoreEngine()
        raise ValueError(f"unknown engine {name!r}; expected one of {ENGINE_NAMES}")
