"""Benchmark string cleaning operations across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_string_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkStrings:
    """Benchmark suite for string manipulation and cleaning."""

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
        self.string_cols = get_string_columns(self.df)
        
        # Clean up memory
        gc_collect()

    def time_trim_whitespace(self, n_rows: int, library: str) -> None:
        """Measure time to trim whitespace from all string columns."""
        df = fresh_copy(self.df)
        self.adapter.trim_whitespace(df, self.string_cols)

    def peakmem_trim_whitespace(self, n_rows: int, library: str) -> None:
        """Measure peak memory when trimming whitespace."""
        df = fresh_copy(self.df)
        self.adapter.trim_whitespace(df, self.string_cols)

    def time_lowercase(self, n_rows: int, library: str) -> None:
        """Measure time to convert string columns to lowercase."""
        df = fresh_copy(self.df)
        self.adapter.to_lowercase(df, self.string_cols)

    def peakmem_lowercase(self, n_rows: int, library: str) -> None:
        """Measure peak memory when converting to lowercase."""
        df = fresh_copy(self.df)
        self.adapter.to_lowercase(df, self.string_cols)

    def time_uppercase(self, n_rows: int, library: str) -> None:
        """Measure time to convert string columns to uppercase."""
        df = fresh_copy(self.df)
        self.adapter.to_uppercase(df, self.string_cols)

    def time_regex_replace(self, n_rows: int, library: str) -> None:
        """Measure time to perform a regex replacement on string columns."""
        df = fresh_copy(self.df)
        pattern = r'[^a-zA-Z0-9 ]'
        replacement = ''
        self.adapter.regex_replace(df, self.string_cols, pattern, replacement)
