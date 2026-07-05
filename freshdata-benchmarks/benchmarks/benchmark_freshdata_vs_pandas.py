"""Focused freshdata-vs-pandas comparison across six core data operations.

This is the headline benchmark for CI: it covers loading, missing-value
handling, outlier detection/filtering, duplicate resolution, group
aggregations, and a representative preprocessing pipeline — all measured
at three configurable dataset sizes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time as _time

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import fresh_copy, gc_collect, get_numeric_columns
from benchmarks.utils.library_wrappers import get_adapter


class BenchmarkFreshDataVsPandas:
    """Headline benchmark comparing FreshData and Pandas across six operations."""

    timeout = 600
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas"],
    ]

    def setup(self, n_rows: int, library: str) -> None:
        """Prepare dataset, adapter, and temp files for the benchmark."""
        self.adapter = get_adapter(library)
        if not self.adapter.available:
            raise NotImplementedError(f"{library} is not available")

        base_df = get_dataset(n_rows)
        self.df = fresh_copy(base_df)
        self.numeric_cols = get_numeric_columns(self.df)

        # Write a CSV for loading benchmarks
        self._tmpdir = tempfile.mkdtemp(prefix="asv_fdvspd_")
        self.csv_path = os.path.join(self._tmpdir, f"test_{n_rows}.csv")
        self.df.to_csv(self.csv_path, index=False)

        gc_collect()

    def teardown(self, n_rows: int, library: str) -> None:
        """Remove temporary files."""
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── 1. Loading ─────────────────────────────────────────────

    def time_load_csv(self, n_rows: int, library: str) -> None:
        """Measure time to load a CSV file and infer types."""
        if library == "freshdata":
            import freshdata as fd
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            fd.clean(df, strategy="conservative", preserve_original=True)
        else:
            import pandas as pd
            pd.read_csv(self.csv_path)

    def peakmem_load_csv(self, n_rows: int, library: str) -> None:
        """Measure peak memory when loading a CSV file."""
        if library == "freshdata":
            import freshdata as fd
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            fd.clean(df, strategy="conservative", preserve_original=True)
        else:
            import pandas as pd
            pd.read_csv(self.csv_path)

    # ── 2. Missing Value Handling ──────────────────────────────

    def time_handle_missing(self, n_rows: int, library: str) -> None:
        """Measure time to fill missing values with column means."""
        df = fresh_copy(self.df)
        self.adapter.fill_missing_mean(df, self.numeric_cols)

    def peakmem_handle_missing(self, n_rows: int, library: str) -> None:
        """Measure peak memory when filling missing values."""
        df = fresh_copy(self.df)
        self.adapter.fill_missing_mean(df, self.numeric_cols)

    # ── 3. Outlier Detection / Filtering ───────────────────────

    def time_detect_outliers(self, n_rows: int, library: str) -> None:
        """Measure time to detect outliers using IQR method."""
        df = fresh_copy(self.df)
        self.adapter.detect_outliers_iqr(df, self.numeric_cols)

    def peakmem_detect_outliers(self, n_rows: int, library: str) -> None:
        """Measure peak memory during IQR outlier detection."""
        df = fresh_copy(self.df)
        self.adapter.detect_outliers_iqr(df, self.numeric_cols)

    # ── 4. Duplicate Resolution ────────────────────────────────

    def time_resolve_duplicates(self, n_rows: int, library: str) -> None:
        """Measure time to detect and drop duplicate rows."""
        df = fresh_copy(self.df)
        self.adapter.drop_duplicates(df)

    def peakmem_resolve_duplicates(self, n_rows: int, library: str) -> None:
        """Measure peak memory when dropping duplicate rows."""
        df = fresh_copy(self.df)
        self.adapter.drop_duplicates(df)

    # ── 5. Group Aggregations ──────────────────────────────────

    def time_group_aggregations(self, n_rows: int, library: str) -> None:
        """Measure time for multi-column group aggregations."""
        df = fresh_copy(self.df)
        self.adapter.group_agg_multi(df, self.numeric_cols)

    def peakmem_group_aggregations(self, n_rows: int, library: str) -> None:
        """Measure peak memory for group aggregations."""
        df = fresh_copy(self.df)
        self.adapter.group_agg_multi(df, self.numeric_cols)

    # ── 6. Full Preprocessing Pipeline ─────────────────────────

    def time_full_pipeline(self, n_rows: int, library: str) -> None:
        """Measure time for the complete preprocessing pipeline."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    def peakmem_full_pipeline(self, n_rows: int, library: str) -> None:
        """Measure peak memory for the complete preprocessing pipeline."""
        df = fresh_copy(self.df)
        self.adapter.full_clean(df)

    # ── Throughput Tracker ─────────────────────────────────────

    def track_pipeline_throughput(self, n_rows: int, library: str) -> float:
        """Calculate rows processed per second for the full pipeline."""
        df = fresh_copy(self.df)

        start = _time.perf_counter()
        self.adapter.full_clean(df)
        elapsed = _time.perf_counter() - start

        return n_rows / elapsed if elapsed > 0 else float("inf")

    track_pipeline_throughput.unit = "rows/s"
