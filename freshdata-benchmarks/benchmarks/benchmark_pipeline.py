"""Benchmark the complete end-to-end cleaning pipeline.

This is the headline benchmark for the FreshData suite, comparing the full
auto-clean pipeline against equivalent manual operations in other libraries.
"""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkPipeline:
    """Benchmark suite for full dataset cleaning pipelines."""

    timeout = 600
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000, 5_000_000],
        ["freshdata", "pandas", "polars", "pyjanitor", "autoclean"],
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

    def time_full_clean(self, n_rows: int, library: str) -> None:
        """Measure time to run the complete data cleaning pipeline."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    def peakmem_full_clean(self, n_rows: int, library: str) -> None:
        """Measure peak memory used during the full cleaning pipeline."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    def track_throughput(self, n_rows: int, library: str) -> float:
        """Calculate pipeline throughput in rows per second."""
        import time as _time
        df = fresh_copy(self.df)
        
        start = _time.perf_counter()
        self.adapter.full_clean(df)
        elapsed = _time.perf_counter() - start
        
        return n_rows / elapsed if elapsed > 0 else float("inf")

    track_throughput.unit = "rows/s"

    def track_output_rows(self, n_rows: int, library: str) -> int:
        """Track the number of rows remaining after cleaning."""
        df = fresh_copy(self.df)
        result = self.adapter.full_clean(df)
        try:
            return len(result)
        except Exception:
            return -1

    track_output_rows.unit = "rows"

    def track_output_cols(self, n_rows: int, library: str) -> int:
        """Track the number of columns remaining after cleaning."""
        df = fresh_copy(self.df)
        result = self.adapter.full_clean(df)
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame):
                return len(result.columns)
            return len(getattr(result, "columns", []))
        except Exception:
            return -1

    track_output_cols.unit = "cols"
