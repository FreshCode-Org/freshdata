"""Benchmark missing value operations across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_numeric_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkMissing:
    """Benchmark suite for missing value handling operations."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas", "polars", "feature_engine", "sklearn"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset and adapter for the benchmark."""
        self.adapter = get_adapter(library)
        if not self.adapter.available:
            raise NotImplementedError(f"{library} is not available")

        # Get base dataset
        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        self.numeric_cols = get_numeric_columns(self.df)
        
        # Clean up memory
        gc_collect()

    def time_drop_missing(self, n_rows: int, library: str) -> None:
        """Measure time to drop rows with missing values."""
        df = fresh_copy(self.df)
        self.adapter.drop_missing_rows(df)

    def peakmem_drop_missing(self, n_rows: int, library: str) -> None:
        """Measure peak memory when dropping rows with missing values."""
        df = fresh_copy(self.df)
        self.adapter.drop_missing_rows(df)

    def time_fill_mean(self, n_rows: int, library: str) -> None:
        """Measure time to fill missing values with the column mean."""
        df = fresh_copy(self.df)
        self.adapter.fill_missing_mean(df, self.numeric_cols)

    def peakmem_fill_mean(self, n_rows: int, library: str) -> None:
        """Measure peak memory when filling missing values with column mean."""
        df = fresh_copy(self.df)
        self.adapter.fill_missing_mean(df, self.numeric_cols)

    def time_fill_median(self, n_rows: int, library: str) -> None:
        """Measure time to fill missing values with the column median."""
        df = fresh_copy(self.df)
        self.adapter.fill_missing_median(df, self.numeric_cols)

    def peakmem_fill_median(self, n_rows: int, library: str) -> None:
        """Measure peak memory when filling missing values with column median."""
        df = fresh_copy(self.df)
        self.adapter.fill_missing_median(df, self.numeric_cols)

    def time_fill_mode(self, n_rows: int, library: str) -> None:
        """Measure time to fill missing values with the column mode.
        Supported by pandas, sklearn, and freshdata.
        """
        df = fresh_copy(self.df)
        self.adapter.fill_missing_mode(df, self.numeric_cols)

    def time_fill_ffill(self, n_rows: int, library: str) -> None:
        """Measure time to forward-fill missing values.
        Supported by pandas, polars, and freshdata.
        """
        df = fresh_copy(self.df)
        self.adapter.fill_missing_ffill(df)

    def time_fill_bfill(self, n_rows: int, library: str) -> None:
        """Measure time to backward-fill missing values.
        Supported by pandas, polars, and freshdata.
        """
        df = fresh_copy(self.df)
        self.adapter.fill_missing_bfill(df)

    def track_throughput_drop_missing(self, n_rows: int, library: str) -> float:
        """Calculate rows processed per second for drop_missing."""
        import time as _time
        df = fresh_copy(self.df)
        
        start = _time.perf_counter()
        self.adapter.drop_missing_rows(df)
        elapsed = _time.perf_counter() - start
        
        return n_rows / elapsed if elapsed > 0 else float("inf")

    track_throughput_drop_missing.unit = "rows/s"
