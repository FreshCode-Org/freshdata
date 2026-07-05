"""Benchmark data type conversion and optimization across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkTypes:
    """Benchmark suite for type inference, conversion, and memory optimization."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas", "polars"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset and adapter for the benchmark."""
        self.adapter = get_adapter(library)
        if not self.adapter.available:
            raise NotImplementedError(f"{library} is not available")

        # Get base dataset
        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        
        # Define specific columns to target
        self.numeric_targets = ["int_col_1", "int_col_2", "float_col_1"]
        self.date_targets = ["date_col"]
        
        # Clean up memory
        gc_collect()

    def time_convert_numeric(self, n_rows: int, library: str) -> None:
        """Measure time to convert string columns to numeric types."""
        df = fresh_copy(self.df)
        self.adapter.convert_numeric(df, self.numeric_targets)

    def time_convert_datetime(self, n_rows: int, library: str) -> None:
        """Measure time to parse datetime strings."""
        df = fresh_copy(self.df)
        self.adapter.convert_datetime(df, self.date_targets)

    def time_optimize_dtypes(self, n_rows: int, library: str) -> None:
        """Measure time to perform global type optimization (downcasting)."""
        df = fresh_copy(self.df)
        self.adapter.optimize_dtypes(df)

    def peakmem_optimize_dtypes(self, n_rows: int, library: str) -> None:
        """Measure peak memory during global type optimization."""
        df = fresh_copy(self.df)
        self.adapter.optimize_dtypes(df)
