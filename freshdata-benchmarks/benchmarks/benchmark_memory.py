"""Memory-focused benchmarks and profiling."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkMemory:
    """Benchmark suite dedicated to memory usage analysis."""

    timeout = 600
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000, 5_000_000],
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
        
        # Clean up memory
        gc_collect()

    def peakmem_full_clean(self, n_rows: int, library: str) -> None:
        """Measure peak RSS memory used during the full cleaning pipeline."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    def track_memory_before(self, n_rows: int, library: str) -> float:
        """Track the deep memory usage of the uncleaned input DataFrame (in MB)."""
        df = fresh_copy(self.df)
        import pandas as pd
        if isinstance(df, pd.DataFrame):
            return df.memory_usage(deep=True).sum() / (1024 * 1024)
        return -1.0

    track_memory_before.unit = "MB"

    def track_memory_after(self, n_rows: int, library: str) -> float:
        """Track the deep memory usage of the cleaned output DataFrame (in MB)."""
        df = fresh_copy(self.df)
        result = self.adapter.full_clean(df)
        try:
            import pandas as pd
            if isinstance(result, pd.DataFrame):
                return result.memory_usage(deep=True).sum() / (1024 * 1024)
            else:
                # Approximate memory for Polars
                return result.estimated_size() / (1024 * 1024)
        except Exception:
            return -1.0

    track_memory_after.unit = "MB"

    def track_memory_ratio(self, n_rows: int, library: str) -> float:
        """Track the ratio of output memory / input memory.
        Values < 1.0 indicate memory savings (e.g. via downcasting).
        """
        df = fresh_copy(self.df)
        
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            return float('nan')
            
        mem_before = df.memory_usage(deep=True).sum()
        if mem_before == 0:
            return float('nan')
            
        result = self.adapter.full_clean(df)
        
        try:
            if isinstance(result, pd.DataFrame):
                mem_after = result.memory_usage(deep=True).sum()
            else:
                mem_after = result.estimated_size()
            return float(mem_after) / float(mem_before)
        except Exception:
            return float('nan')

    track_memory_ratio.unit = "ratio"
