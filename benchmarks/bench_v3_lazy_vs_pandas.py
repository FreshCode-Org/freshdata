from __future__ import annotations

import argparse
import json
import platform
try:
    import resource
except ImportError:
    resource = None
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import polars as pl

from freshdata.polars_engine import CleanConfig, apply_plan, suggest_plan


def get_peak_rss_mb() -> float:
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports ru_maxrss in bytes; Linux reports in kilobytes
        if platform.system() == "Darwin":
            return usage / (1024 * 1024)
        return usage / 1024.0
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def generate_messy_parquet(path: Path, n_rows: int) -> None:
    base = pl.int_range(0, n_rows, eager=True)
    df = pl.DataFrame({"idx": base}).with_columns(
        [
            pl.col("idx").mod(n_rows // 10 * 9).cast(pl.String).alias("Customer ID"),
            pl.when(pl.col("idx").mod(10) == 0)
            .then(None)
            .otherwise(20 + pl.col("idx").mod(50))
            .cast(pl.Float64)
            .alias("Age"),
            pl.when(pl.col("idx").mod(200) == 0)
            .then(99999.0)
            .otherwise(50.0 + (pl.col("idx").mod(100) * 1.5))
            .alias("Spend"),
            pl.when(pl.col("idx").mod(15) == 0)
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("idx").mod(3) == 0)
            .then(pl.lit("  Gold "))
            .otherwise(pl.lit("Silver"))
            .alias("Segment"),
            pl.when(pl.col("idx").mod(20) == 0)
            .then(None)
            .otherwise(pl.col("idx").mod(2))
            .cast(pl.Int64)
            .alias("Churn"),
        ]
    ).drop("idx")
    df.write_parquet(path)


def run_pandas_fallback(src: Path, dst: Path) -> dict[str, float]:
    t0 = time.perf_counter()
    df = pl.scan_parquet(src).collect().to_pandas()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["segment"] = df["segment"].str.strip()
    df["age"] = df["age"].fillna(df["age"].median())
    df["segment"] = df["segment"].fillna(df["segment"].mode().iloc[0])

    q25, q75 = df["spend"].quantile(0.25), df["spend"].quantile(0.75)
    iqr = q75 - q25
    df["spend"] = df["spend"].clip(lower=q25 - 1.5 * iqr, upper=q75 + 1.5 * iqr)

    q25_a, q75_a = df["age"].quantile(0.25), df["age"].quantile(0.75)
    iqr_a = q75_a - q25_a
    df["age"] = df["age"].clip(lower=q25_a - 1.5 * iqr_a, upper=q75_a + 1.5 * iqr_a)

    try:
        df = df.drop_duplicates()
    except TypeError:
        mask = ~df.duplicated().to_numpy()
        df = pd.DataFrame({c: df[c].to_numpy()[mask] for c in df.columns})
    pl.from_pandas(df).write_parquet(dst)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "plan_ms": 0.0,
        "compile_ms": 0.0,
        "execute_ms": total_ms,
        "total_ms": total_ms,
        "peak_rss_mb": get_peak_rss_mb(),
    }


def run_polars_eager_unique(src: Path, dst: Path) -> dict[str, float]:
    t0 = time.perf_counter()
    lf = pl.scan_parquet(src)
    config = CleanConfig(
        id_columns=("customer_id",),
        target_column="churn",
        default_impute="auto",
        default_outliers="clip",
        drop_duplicates=False,
    )
    plan = suggest_plan(lf, config=config)
    t_plan = time.perf_counter()

    repaired_lf = apply_plan(lf, plan)
    # Simulate legacy Issue #53: maintain_order=True + eager .collect()
    repaired_lf = repaired_lf.unique(maintain_order=True)
    t_compile = time.perf_counter()

    repaired_lf.collect().write_parquet(dst)
    t_end = time.perf_counter()

    return {
        "plan_ms": (t_plan - t0) * 1000.0,
        "compile_ms": (t_compile - t_plan) * 1000.0,
        "execute_ms": (t_end - t_compile) * 1000.0,
        "total_ms": (t_end - t0) * 1000.0,
        "peak_rss_mb": get_peak_rss_mb(),
    }


def run_freshdata_v3_lazy(src: Path, dst: Path) -> dict[str, float]:
    t0 = time.perf_counter()
    lf = pl.scan_parquet(src)
    config = CleanConfig(
        id_columns=("customer_id",),
        target_column="churn",
        default_impute="auto",
        default_outliers="clip",
        drop_duplicates=True,
    )
    plan = suggest_plan(lf, config=config)
    t_plan = time.perf_counter()

    cleaned_lf = apply_plan(lf, plan)
    t_compile = time.perf_counter()

    cleaned_lf.sink_parquet(dst)
    t_end = time.perf_counter()

    return {
        "plan_ms": (t_plan - t0) * 1000.0,
        "compile_ms": (t_compile - t_plan) * 1000.0,
        "execute_ms": (t_end - t_compile) * 1000.0,
        "total_ms": (t_end - t0) * 1000.0,
        "peak_rss_mb": get_peak_rss_mb(),
    }


WORKERS = {
    "pandas_fallback": run_pandas_fallback,
    "polars_eager_unique": run_polars_eager_unique,
    "freshdata_v3_lazy": run_freshdata_v3_lazy,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark freshdata v3.0 LazyFrame engine.")
    parser.add_argument("--rows", type=int, default=5_000_000, help="Number of rows to benchmark.")
    parser.add_argument("--worker", choices=list(WORKERS.keys()), help="Internal worker mode.")
    parser.add_argument("--src", type=Path, help="Input parquet path for worker.")
    parser.add_argument("--dst", type=Path, help="Output parquet path for worker.")
    args = parser.parse_args()

    if args.worker:
        metrics = WORKERS[args.worker](args.src, args.dst)
        print(json.dumps(metrics))
        return 0

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        src_path = tmp / "input.parquet"
        print(f"Generating {args.rows:,}-row messy Parquet fixture...", file=sys.stderr)
        generate_messy_parquet(src_path, args.rows)

        results: dict[str, dict[str, float]] = {}
        for mode in WORKERS:
            dst_path = tmp / f"out_{mode}.parquet"
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    mode,
                    "--src",
                    str(src_path),
                    "--dst",
                    str(dst_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            results[mode] = json.loads(proc.stdout.strip().splitlines()[-1])

    print("\n| Pipeline Mode | Plan (ms) | Compile (ms) | Total (ms) | Peak RSS (MB) |")
    print("| :--- | ---: | ---: | ---: | ---: |")
    for mode, m in results.items():
        print(
            f"| {mode} | {m['plan_ms']:.1f} | {m['compile_ms']:.2f} "
            f"| {m['total_ms']:.1f} | {m['peak_rss_mb']:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
