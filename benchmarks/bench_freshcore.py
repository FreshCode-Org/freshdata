#!/usr/bin/env python
"""Benchmark FreshCore against pandas and FreshData's reference path.

This script is intentionally separate from the main benchmark harness because
FreshCore is an optional native extension. It reports observed timings only; it
does not claim speedups unless the native module is installed and the parity
checks pass.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import freshdata as fd  # noqa: E402
from baselines import pandas_baseline  # noqa: E402


def make_dataset(rows: int, *, wide: bool = False, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_extra = 40 if wide else 4
    df = pd.DataFrame(
        {
            " Customer ID ": [f"C{i:07d}" for i in range(rows)],
            "amount": rng.normal(100.0, 15.0, rows),
            "category": rng.choice([" A ", "b", "C", "N/A"], rows),
            "name": rng.choice([" Alice ", "BOB", " carol", "DAN "], rows),
            "flag": rng.choice(["yes", "no"], rows),
            "date_text": pd.date_range("2024-01-01", periods=rows, freq="min")
            .astype(str)
            .to_numpy(),
        }
    )
    if rows > 10:
        df.loc[::17, "amount"] = np.nan
        df.loc[::251, "amount"] = 10_000
        df.loc[::23, "category"] = None
        df.iloc[1::997] = df.iloc[0].to_numpy()
    for i in range(n_extra):
        df[f"extra_{i}"] = rng.normal(i, 1.0, rows)
        if i % 3 == 0:
            df.loc[::31, f"extra_{i}"] = np.nan
    return df


def freshcore_config(workload: str) -> fd.CleanConfig:
    base = {"strategy": "conservative", "verbose": False}
    if workload == "strings":
        return fd.CleanConfig(**base, fix_dtypes=False, string_case="lower")
    if workload == "impute":
        return fd.CleanConfig(**base, fix_dtypes=False, impute="median")
    if workload == "casts":
        return fd.CleanConfig(**base, fix_dtypes=True, drop_duplicates=False)
    if workload == "duplicates":
        return fd.CleanConfig(**base, fix_dtypes=False, drop_duplicates=True)
    if workload == "outliers":
        return fd.CleanConfig(**base, fix_dtypes=False, outliers="clip", outlier_method="iqr")
    return fd.CleanConfig(
        **base,
        fix_dtypes=True,
        string_case="lower",
        impute="median",
        outliers="clip",
        outlier_method="iqr",
    )


def bench(label: str, fn: Callable[[], object], repeat: int) -> dict[str, object]:
    times = []
    for _ in range(repeat):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    gc.collect()
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return {
        "label": label,
        "p50_sec": statistics.median(times),
        "min_sec": min(times),
        "peak_mb": peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--workload", default="full",
                        choices=["strings", "impute", "casts", "duplicates", "outliers", "full"])
    parser.add_argument("--wide", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    rows = []
    for n in args.rows:
        df = make_dataset(n, wide=args.wide)
        cfg = freshcore_config(args.workload)
        results = [
            bench(
                "pandas_baseline",
                lambda df=df: pandas_baseline.run(df.copy()),
                args.repeat,
            ),
            bench(
                "freshdata_pandas",
                lambda df=df, cfg=cfg: fd.clean(df.copy(), config=cfg),
                args.repeat,
            ),
            bench(
                "freshcore",
                lambda df=df, cfg=cfg: fd.clean(df.copy(), config=cfg, engine="freshcore"),
                args.repeat,
            ),
        ]
        fc_out, fc_report = fd.clean(df.copy(), config=cfg, engine="freshcore", return_report=True)
        ref_out = fd.clean(df.copy(), config=cfg)
        parity = {
            "shape_equal": tuple(fc_out.shape) == tuple(ref_out.shape),
            "fallback_events": fc_report.to_dict().get("fallback_events", []),
            "stage_timings": fc_report.to_dict().get("stage_timings", []),
        }
        row = {"rows": n, "cols": df.shape[1], "workload": args.workload,
               "wide": args.wide, "results": results, "parity": parity}
        rows.append(row)
        print(json.dumps(row, indent=2, default=str))

    if args.out:
        args.out.write_text(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
