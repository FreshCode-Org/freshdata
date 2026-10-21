# benchmarks/run_scale_profile.py
from __future__ import annotations

import resource
import time
import tracemalloc
import numpy as np
import pandas as pd

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.execution import EngineConfig


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
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, drop_duplicates=False, impute_missing=False)
    for n in (10_000, 100_000, 1_000_000):
        df = make_scale_df(n)
        profile_run(f"n={n} pandas_baseline", lambda: df.assign(cat=df["cat"].str.strip()))
        profile_run(f"n={n} freshdata[pandas]", lambda: fd.clean(df, config=cfg, engine="pandas"))
        profile_run(
            f"n={n} freshdata[polars,streaming]",
            lambda: fd.clean(df, config=cfg, engine="polars", engine_config=EngineConfig(engine="polars", streaming=True)),
        )
        profile_run(f"n={n} freshdata[duckdb]", lambda: fd.clean(df, config=cfg, engine="duckdb"))


if __name__ == "__main__":
    main()
