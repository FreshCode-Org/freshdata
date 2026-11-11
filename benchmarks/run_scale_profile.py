# benchmarks/run_scale_profile.py
from __future__ import annotations

import time
import tracemalloc
import numpy as np
import pandas as pd

import polars as pl
import freshdata as fd
from freshdata import CleanConfig


def make_scale_df(n_rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id": pd.Series(np.arange(n_rows, dtype=np.int64), dtype="Int64"),
        "val_a": pd.Series(rng.normal(100.0, 15.0, size=n_rows), dtype="Float64"),
        "val_b": pd.Series(rng.integers(1, 500, size=n_rows), dtype="Int64"),
        "cat": pd.Series(rng.choice([" alpha ", "beta", "N/A", "delta"], size=n_rows), dtype="string"),
    })


def profile_run(label: str, fn) -> tuple[float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)
    print(f"SCALE_BENCH | {label:32s} | latency={elapsed:.3f}s | peak_alloc={peak_mb:.1f} MB")
    return elapsed, peak_mb


def main() -> None:
    cfg = CleanConfig(strategy="conservative", default_impute="preserve", default_outliers="preserve")
    for n in (10_000, 100_000, 1_000_000):
        df = make_scale_df(n)
        profile_run(f"n={n} pandas_baseline", lambda: df.assign(cat=df["cat"].str.strip()))
        profile_run(f"n={n} freshdata[pandas]", lambda: fd.clean(df, config=cfg))
        pldf = pl.from_pandas(df)
        profile_run(f"n={n} freshdata[polars,eager]", lambda: fd.clean(pldf, config=cfg))
        profile_run(
            f"n={n} freshdata[polars,lazy]",
            lambda: fd.clean(pldf.lazy(), config=cfg).collect(),
        )


if __name__ == "__main__":
    main()
