"""Analysis of scaling efficiency across dataset sizes."""

from __future__ import annotations

import gc

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkScalingCurves:
    """Benchmark suite to map out Big-O scaling complexity."""

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
        
        gc_collect()

    def time_clean_scaling(self, n_rows: int, library: str) -> None:
        """Time the full pipeline to map out scaling curves."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    def peakmem_clean_scaling(self, n_rows: int, library: str) -> None:
        """Peak memory of the full pipeline to map out scaling curves."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    def track_throughput(self, n_rows: int, library: str) -> float:
        """Rows per second at each dataset size."""
        import time as _time
        df = fresh_copy(self.df)
        
        start = _time.perf_counter()
        self.adapter.full_clean(df)
        elapsed = _time.perf_counter() - start
        
        return n_rows / elapsed if elapsed > 0 else float("inf")

    track_throughput.unit = "rows/s"
    
    def track_speedup_vs_pandas(self, n_rows: int, library: str) -> float:
        """Speedup relative to Pandas for the full cleaning pipeline."""
        import time as _time
        from benchmarks.utils.library_wrappers import get_adapter
        
        pd_adapter = get_adapter("pandas")
        if not pd_adapter.available:
            return float("nan")
            
        # Time pandas baseline
        df_pd = fresh_copy(self.df)
        start = _time.perf_counter()
        pd_adapter.full_clean(df_pd)
        pandas_time = _time.perf_counter() - start
        
        if pandas_time <= 0:
            return float("nan")
            
        # If library IS pandas, speedup is exactly 1.0
        if library == "pandas":
            return 1.0
            
        # Time target library
        df_lib = fresh_copy(self.df)
        start = _time.perf_counter()
        self.adapter.full_clean(df_lib)
        library_time = _time.perf_counter() - start
        
        return pandas_time / library_time if library_time > 0 else float("nan")
        
    track_speedup_vs_pandas.unit = "ratio"
