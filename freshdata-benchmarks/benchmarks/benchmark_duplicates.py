"""Benchmark duplicate row operations across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkDuplicates:
    """Benchmark suite for duplicate detection and removal."""

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
        
        # Clean up memory
        gc_collect()

    def time_detect_duplicates(self, n_rows: int, library: str) -> None:
        """Measure time to detect duplicate rows."""
        df = fresh_copy(self.df)
        self.adapter.detect_duplicates(df)

    def peakmem_detect_duplicates(self, n_rows: int, library: str) -> None:
        """Measure peak memory when detecting duplicate rows."""
        df = fresh_copy(self.df)
        self.adapter.detect_duplicates(df)

    def time_drop_duplicates(self, n_rows: int, library: str) -> None:
        """Measure time to drop duplicate rows."""
        df = fresh_copy(self.df)
        self.adapter.drop_duplicates(df)

    def peakmem_drop_duplicates(self, n_rows: int, library: str) -> None:
        """Measure peak memory when dropping duplicate rows."""
        df = fresh_copy(self.df)
        self.adapter.drop_duplicates(df)

    def track_duplicate_count(self, n_rows: int, library: str) -> int:
        """Track the number of duplicate rows dropped."""
        df = fresh_copy(self.df)
        start_rows = len(df)
        result = self.adapter.drop_duplicates(df)
        
        # Approximate counting since libraries return different formats
        try:
            return start_rows - len(result)
        except Exception:
            return -1

    track_duplicate_count.unit = "rows"
