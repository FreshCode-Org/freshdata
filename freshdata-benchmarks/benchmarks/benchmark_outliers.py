"""Benchmark outlier detection algorithms across libraries."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_numeric_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkOutliers:
    """Benchmark suite for outlier detection techniques."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas", "feature_engine", "sklearn", "polars"],
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
        
        # Scikit-learn and Feature Engine require clean, dense numeric data (no NaNs)
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

    def time_detect_iqr(self, n_rows: int, library: str) -> None:
        """Measure time to detect outliers using the Interquartile Range (IQR) method."""
        df = self._get_target_df(library)
        self.adapter.detect_outliers_iqr(df, self.numeric_cols)

    def peakmem_detect_iqr(self, n_rows: int, library: str) -> None:
        """Measure peak memory during IQR outlier detection."""
        df = self._get_target_df(library)
        self.adapter.detect_outliers_iqr(df, self.numeric_cols)

    def time_detect_zscore(self, n_rows: int, library: str) -> None:
        """Measure time to detect outliers using Z-score (standard deviations)."""
        df = self._get_target_df(library)
        self.adapter.detect_outliers_zscore(df, self.numeric_cols)

    def peakmem_detect_zscore(self, n_rows: int, library: str) -> None:
        """Measure peak memory during Z-score outlier detection."""
        df = self._get_target_df(library)
        self.adapter.detect_outliers_zscore(df, self.numeric_cols)

    def track_outlier_count_iqr(self, n_rows: int, library: str) -> int:
        """Track the total number of outliers detected by IQR across all numeric columns."""
        df = self._get_target_df(library)
        result = self.adapter.detect_outliers_iqr(df, self.numeric_cols)
        
        outlier_count = 0
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame):
                outlier_cols = [c for c in result.columns if c.endswith("_outlier")]
                for col in outlier_cols:
                    outlier_count += int(result[col].sum())
        except Exception:
            pass
            
        return outlier_count if outlier_count > 0 else -1

    track_outlier_count_iqr.unit = "outliers"
