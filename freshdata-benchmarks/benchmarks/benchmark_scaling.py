"""Benchmark feature scaling operations across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_numeric_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkScaling:
    """Benchmark suite for feature scaling (Standardization, MinMax)."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["pandas", "sklearn", "feature_engine", "polars"],
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
        
        # Ensure clean numeric data for sklearn/feature_engine
        import pandas as pd
        self.df_numeric = pd.DataFrame()
        for col in self.numeric_cols:
            self.df_numeric[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df_numeric = self.df_numeric.dropna().reset_index(drop=True)
        
        # Clean up memory
        gc_collect()

    def _get_target_df(self, library: str):
        """Return the appropriate DataFrame depending on library requirements."""
        if library in ["sklearn", "feature_engine"]:
            return fresh_copy(self.df_numeric)
        return fresh_copy(self.df)

    def time_standard_scale(self, n_rows: int, library: str) -> None:
        """Measure time to apply StandardScaler (Z-score normalization)."""
        df = self._get_target_df(library)
        self.adapter.standard_scale(df, self.numeric_cols)

    def peakmem_standard_scale(self, n_rows: int, library: str) -> None:
        """Measure peak memory during StandardScaler."""
        df = self._get_target_df(library)
        self.adapter.standard_scale(df, self.numeric_cols)

    def time_minmax_scale(self, n_rows: int, library: str) -> None:
        """Measure time to apply MinMaxScaler (0-1 normalization)."""
        df = self._get_target_df(library)
        self.adapter.minmax_scale(df, self.numeric_cols)

    def peakmem_minmax_scale(self, n_rows: int, library: str) -> None:
        """Measure peak memory during MinMaxScaler."""
        df = self._get_target_df(library)
        self.adapter.minmax_scale(df, self.numeric_cols)
