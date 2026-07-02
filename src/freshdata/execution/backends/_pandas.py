"""Pandas backend — the reference implementation.

It delegates to the existing :func:`freshdata.cleaner.run_pipeline`, so its
output is byte-for-byte identical to ``fd.clean(df)``. The Polars and DuckDB
backends fall back to this engine for any step they cannot run natively, which
is how parity is guaranteed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._base import ExecutionEngine

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from ...config import CleanConfig
    from ...report import CleanReport
    from .._config import EngineConfig


def materialize_to_pandas(source: Any) -> pd.DataFrame:
    """Load *source* into a pandas DataFrame, reading file paths if needed."""
    import pandas as pd

    if isinstance(source, pd.DataFrame):
        return source
    if isinstance(source, str):
        low = source.lower()
        if low.endswith((".parquet", ".pq")):
            return pd.read_parquet(source)
        if low.endswith(".csv"):
            return pd.read_csv(source)
        if low.endswith((".ipc", ".feather", ".arrow")):
            return pd.read_feather(source)
        raise ValueError(f"unsupported file type for path {source!r}")
    # polars frame
    to_pandas = getattr(source, "to_pandas", None)
    if callable(to_pandas):
        return to_pandas()
    # duckdb relation
    to_df = getattr(source, "df", None)
    if callable(to_df):
        return to_df()
    raise TypeError(f"cannot materialize source of type {type(source).__name__}")


class PandasEngine(ExecutionEngine):
    name = "pandas"

    def execute(
        self,
        source: Any,
        config: CleanConfig,
        engine_config: EngineConfig,
    ) -> tuple[pd.DataFrame, CleanReport]:
        from ...cleaner import run_pipeline

        df = materialize_to_pandas(source)
        return run_pipeline(df, config)
