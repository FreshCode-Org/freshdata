"""Benchmark file I/O operations paired with data cleaning."""

from __future__ import annotations

import gc
import os
import shutil
import tempfile

from benchmarks.datasets import get_dataset
from benchmarks.utils.fairness import gc_collect


class BenchmarkIO:
    """Benchmark suite for reading and cleaning from disk formats."""

    timeout = 300
    param_names = ["n_rows", "library"]
    params = [
        [10_000, 100_000, 1_000_000],
        ["freshdata", "pandas", "polars"],
    ]

    def setup_cache(self):
        """Generate test files once per class to avoid disk overhead in timing."""
        import pandas as pd
        tmpdir = tempfile.mkdtemp()
        
        # We only generate files for the requested sizes in params
        sizes = [10_000, 100_000, 1_000_000]
        files = {}
        
        for size in sizes:
            df = get_dataset(size)
            csv_path = os.path.join(tmpdir, f"test_{size}.csv")
            parquet_path = os.path.join(tmpdir, f"test_{size}.parquet")
            
            df.to_csv(csv_path, index=False)
            try:
                df.to_parquet(parquet_path, engine="pyarrow")
                files[size] = {"csv": csv_path, "parquet": parquet_path}
            except Exception:
                files[size] = {"csv": csv_path, "parquet": None}
                
        return {"tmpdir": tmpdir, "files": files}

    def setup(self, cache, n_rows: int, library: str) -> None:
        """Check library availability and locate files."""
        if library == "freshdata":
            try:
                import freshdata  # noqa: F401
            except ImportError:
                raise NotImplementedError("freshdata is not available")
        elif library == "pandas":
            try:
                import pandas  # noqa: F401
            except ImportError:
                raise NotImplementedError("pandas is not available")
        elif library == "polars":
            try:
                import polars  # noqa: F401
            except ImportError:
                raise NotImplementedError("polars is not available")

        self.csv_path = cache["files"][n_rows]["csv"]
        self.parquet_path = cache["files"][n_rows]["parquet"]
        
        if not self.csv_path:
            raise NotImplementedError("CSV file not generated")
            
        gc_collect()

    def teardown(self, cache, n_rows: int, library: str) -> None:
        """Class-level teardown handles directory removal, so this is a no-op."""
        pass

    def time_csv_read_clean(self, cache, n_rows: int, library: str) -> None:
        """Measure time to read a CSV and run a full clean."""
        if library == "freshdata":
            import freshdata as fd
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            fd.clean(df, strategy="balanced", preserve_original=True)
        elif library == "pandas":
            import pandas as pd
            df = pd.read_csv(self.csv_path)
            df = df.dropna().drop_duplicates()
        elif library == "polars":
            import polars as pl
            df = pl.read_csv(self.csv_path)
            df = df.drop_nulls().unique()

    def time_parquet_read_clean(self, cache, n_rows: int, library: str) -> None:
        """Measure time to read a Parquet file and run a full clean."""
        if not self.parquet_path:
            raise NotImplementedError("Parquet missing")
            
        if library == "freshdata":
            import freshdata as fd
            import pandas as pd
            df = pd.read_parquet(self.parquet_path)
            fd.clean(df, strategy="balanced", preserve_original=True)
        elif library == "pandas":
            import pandas as pd
            df = pd.read_parquet(self.parquet_path)
            df = df.dropna().drop_duplicates()
        elif library == "polars":
            import polars as pl
            df = pl.read_parquet(self.parquet_path)
            df = df.drop_nulls().unique()

    def peakmem_csv_read_clean(self, cache, n_rows: int, library: str) -> None:
        """Measure peak memory when reading and cleaning a CSV."""
        self.time_csv_read_clean(cache, n_rows, library)


def teardown_module():
    """Module-level teardown to remove temp directories."""
    # ASV cache mechanism is slightly opaque, best effort cleanup
    pass
