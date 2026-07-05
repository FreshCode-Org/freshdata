"""Benchmark schema and data validation performance."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect


class BenchmarkValidation:
    """Benchmark suite for schema and data quality validation.
    
    Compares FreshData's built-in profiler against manual pandas validation.
    """

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset for the benchmark."""
        if library == "freshdata":
            try:
                import freshdata  # noqa: F401
            except ImportError:
                raise NotImplementedError("freshdata is not available")
        
        # Get base dataset
        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        
        # Clean up memory
        gc_collect()

    def time_profile_validation(self, n_rows: int, library: str) -> None:
        """Measure time to run full data profiling/validation."""
        df = fresh_copy(self.df)
        if library == "freshdata":
            import freshdata as fd
            fd.profile(df)
        else:  # pandas manual profile
            import pandas as pd
            _ = df.dtypes
            _ = df.isnull().sum()
            _ = df.nunique()
            _ = df.describe(include='all')

    def time_null_check(self, n_rows: int, library: str) -> None:
        """Measure time to check for nulls across all columns."""
        df = fresh_copy(self.df)
        if library == "freshdata":
            import freshdata as fd
            # FreshData profile includes null checks
            fd.profile(df)
        else:
            _ = df.isnull().any()

    def time_type_check(self, n_rows: int, library: str) -> None:
        """Measure time to validate data types."""
        df = fresh_copy(self.df)
        if library == "freshdata":
            import freshdata as fd
            fd.profile(df)
        else:
            _ = df.dtypes

    def time_range_check(self, n_rows: int, library: str) -> None:
        """Measure time to check numeric ranges."""
        df = fresh_copy(self.df)
        if library == "freshdata":
            import freshdata as fd
            fd.profile(df)
        else:
            import pandas as pd
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                _ = numeric_df.min()
                _ = numeric_df.max()
