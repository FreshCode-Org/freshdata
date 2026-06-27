"""Benchmark categorical encoding operations across libraries.

Note: FreshData does not provide standalone encoding features in its core API,
so this benchmark compares the broader ecosystem (pandas, sklearn, feature_engine).
"""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_categorical_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkEncoding:
    """Benchmark suite for one-hot and label encoding."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["pandas", "sklearn", "feature_engine", "polars", "pyjanitor"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset and adapter for the benchmark."""
        self.adapter = get_adapter(library)
        if not self.adapter.available:
            raise NotImplementedError(f"{library} is not available")

        # Get base dataset
        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        self.cat_cols = get_categorical_columns(self.df)
        
        # Clean up memory
        gc_collect()

    def time_onehot_encode(self, n_rows: int, library: str) -> None:
        """Measure time to perform one-hot encoding on categorical columns."""
        df = fresh_copy(self.df)
        self.adapter.onehot_encode(df, self.cat_cols)

    def peakmem_onehot_encode(self, n_rows: int, library: str) -> None:
        """Measure peak memory during one-hot encoding."""
        df = fresh_copy(self.df)
        self.adapter.onehot_encode(df, self.cat_cols)

    def time_label_encode(self, n_rows: int, library: str) -> None:
        """Measure time to perform label (ordinal) encoding."""
        df = fresh_copy(self.df)
        self.adapter.label_encode(df, self.cat_cols)

    def peakmem_label_encode(self, n_rows: int, library: str) -> None:
        """Measure peak memory during label encoding."""
        df = fresh_copy(self.df)
        self.adapter.label_encode(df, self.cat_cols)
