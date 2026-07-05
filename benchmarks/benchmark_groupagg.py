"""Benchmark group aggregation operations comparing freshdata vs pandas."""

from __future__ import annotations

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_numeric_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkGroupAgg:
    """Benchmark suite for group aggregation operations."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset and adapter for the benchmark."""
        self.adapter = get_adapter(library)
        if not self.adapter.available:
            raise NotImplementedError(f"{library} is not available")

        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        self.numeric_cols = get_numeric_columns(self.df)

        gc_collect()

    def time_group_agg_single(self, n_rows: int, library: str) -> None:
        """Measure time for single-column groupby with mean aggregation."""
        df = fresh_copy(self.df)
        self.adapter.group_agg_single(df)

    def time_group_agg_multi(self, n_rows: int, library: str) -> None:
        """Measure time for multi-column groupby with multiple agg functions."""
        df = fresh_copy(self.df)
        self.adapter.group_agg_multi(df, self.numeric_cols)

    def time_group_agg_transform(self, n_rows: int, library: str) -> None:
        """Measure time for groupby transform (centering within groups)."""
        df = fresh_copy(self.df)
        self.adapter.group_agg_transform(df, self.numeric_cols)

    def peakmem_group_agg_multi(self, n_rows: int, library: str) -> None:
        """Measure peak memory for multi-column groupby aggregation."""
        df = fresh_copy(self.df)
        self.adapter.group_agg_multi(df, self.numeric_cols)

    def track_throughput_group_agg(self, n_rows: int, library: str) -> float:
        """Calculate rows processed per second for group aggregation."""
        import time as _time
        df = fresh_copy(self.df)

        start = _time.perf_counter()
        self.adapter.group_agg_multi(df, self.numeric_cols)
        elapsed = _time.perf_counter() - start

        return n_rows / elapsed if elapsed > 0 else float("inf")

    track_throughput_group_agg.unit = "rows/s"
