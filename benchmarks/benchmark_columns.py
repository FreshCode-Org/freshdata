"""Benchmark column-level operations across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkColumns:
    """Benchmark suite for column renaming, dropping, and selection."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas", "polars", "pyjanitor"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset and adapter for the benchmark."""
        self.adapter = get_adapter(library)
        if not self.adapter.available:
            raise NotImplementedError(f"{library} is not available")

        # Get base dataset
        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        
        # Define columns for operations
        cols = list(self.df.columns)
        self.rename_mapping = {c: f"renamed_{c}" for c in cols[:5]}
        self.drop_cols = cols[-3:]
        self.select_cols = cols[:5]
        
        # Clean up memory
        gc_collect()

    def time_rename_columns(self, n_rows: int, library: str) -> None:
        """Measure time to rename columns."""
        df = fresh_copy(self.df)
        self.adapter.rename_columns(df, self.rename_mapping)

    def time_drop_columns(self, n_rows: int, library: str) -> None:
        """Measure time to drop specific columns."""
        df = fresh_copy(self.df)
        self.adapter.drop_columns(df, self.drop_cols)

    def time_select_columns(self, n_rows: int, library: str) -> None:
        """Measure time to select a subset of columns."""
        df = fresh_copy(self.df)
        self.adapter.select_columns(df, self.select_cols)
