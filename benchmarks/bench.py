"""Quick benchmark for freshdata.clean on synthetic and fixture data.

Run::

    python benchmarks/bench.py                  # synthetic sizes
    python benchmarks/bench.py --fixtures       # all test fixtures
    python benchmarks/bench.py --fixtures --compare
    python benchmarks/bench.py --online --compare
    python benchmarks/bench.py --online-all --compare
    python benchmarks/bench.py --online-all --tier 1
    python benchmarks/bench.py 10000 100000      # custom row counts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import freshdata as fd

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
ONLINE_CACHE_DIR = FIXTURES_DIR / "online" / "cache"
FIXTURES = [
    "aqi_sample",
    "sales_export",
    "survey_responses",
    "sensor_timeseries",
    "fraud_signals",
    "tiny_cohort",
    "wide_sparse",
    "large_panel",
    "duplicate_heavy",
    "locale_numbers",
    "mixed_roles",
]
ONLINE_REGISTRY_PATH = FIXTURES_DIR / "online" / "registry.json"
def _online_names(*, tier: int | None = None) -> list[str]:
    if not ONLINE_CACHE_DIR.exists():
        return []
    cached = {p.stem for p in ONLINE_CACHE_DIR.glob("*.csv")}
    if tier is None or not ONLINE_REGISTRY_PATH.exists():
        return sorted(cached)
    registry = json.loads(ONLINE_REGISTRY_PATH.read_text())
    return sorted(k for k in cached if int(registry.get(k, {}).get("tier", 2)) == tier)


ONLINE_FIXTURES = _online_names()


def make_messy(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = rng.choice([" Alice ", "Bob", "carol", "N/A", "Dave "], n)
    return pd.DataFrame(
        {
            "Customer ID": np.arange(n),
            " Name ": names,
            "Order Total": [f"${x:,.2f}" for x in rng.uniform(5, 9999, n)],
            "Quantity": rng.integers(1, 50, n).astype(str),
            "Order Date": pd.date_range("2022-01-01", periods=n, freq="min").astype(str),
            "Is Member": rng.choice(["yes", "no", "YES", "No"], n),
            "Notes": rng.choice(["", "-", "expedite", "  gift wrap  ", "null"], n),
            "Unused": [None] * n,
        }
    )


def _timed_clean(df: pd.DataFrame, *, strategy: str) -> tuple[float, int, int, int]:
    fd.clean(df.head(min(len(df), 1000)), strategy=strategy, verbose=False)  # warm-up
    missing_before = int(df.isna().sum().sum())
    start = time.perf_counter()
    _, report = fd.clean(df, strategy=strategy, return_report=True, verbose=False)
    elapsed = time.perf_counter() - start
    cols_delta = report.cols_after - df.shape[1]
    return elapsed, len(report), report.missing_after - missing_before, cols_delta


def bench_synthetic(n: int, *, compare: bool = False) -> None:
    df = make_messy(n)
    if compare:
        hdr = (
            f"{'rows':>10}  {'strategy':<11}  {'seconds':>8}  "
            f"{'rows/s':>8}  {'missingΔ':>9}  {'colsΔ':>6}"
        )
        print(hdr)
    for strategy in ("balanced", "aggressive"):
        elapsed, n_actions, miss_d, cols_d = _timed_clean(df, strategy=strategy)
        rate = n / elapsed if elapsed else 0
        if compare:
            print(
                f"{n:>10,}  {strategy:<11}  {elapsed:8.3f}  {rate:8.0f}  {miss_d:>9}  {cols_d:>6}"
            )
        else:
            print(
                f"{n:>10,} rows  {strategy:<11}  {elapsed:8.3f}s  {rate / 1e6:6.2f}M rows/s  "
                f"{n_actions} actions"
            )


def bench_fixtures(
    *, compare: bool = False, online: bool = False, tier: int | None = None
) -> None:
    names = _online_names(tier=tier) if online else FIXTURES
    base = ONLINE_CACHE_DIR if online else FIXTURES_DIR
    header = (
        f"{'fixture':<20} {'strategy':<11} {'seconds':>8}  "
        f"{'rows/s':>8}  {'missingΔ':>9}  {'colsΔ':>6}"
    )
    if compare:
        print(header)
    else:
        print(f"{'fixture':<20} {'strategy':<11} {'seconds':>8}  {'rows':>6}  actions")
    for name in names:
        path = base / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for strategy in ("balanced", "aggressive"):
            elapsed, n_actions, miss_d, cols_d = _timed_clean(df, strategy=strategy)
            rate = len(df) / elapsed if elapsed else 0
            if compare:
                row = (
                    f"{name:<20} {strategy:<11} {elapsed:8.3f}  "
                    f"{rate:8.0f}  {miss_d:>9}  {cols_d:>6}"
                )
                print(row)
            else:
                print(
                    f"{name:<20} {strategy:<11} {elapsed:8.3f}  {len(df):>6}  {n_actions}"
                )
        if compare:
            bench_compare_table(df, name)


def bench_compare_table(df: pd.DataFrame, label: str = "data") -> None:
    table = fd.compare_clean(df)
    print(f"\n=== compare_clean: {label} ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    print(f"freshdata {fd.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
    compare = "--compare" in sys.argv
    online = "--online" in sys.argv or "--online-all" in sys.argv
    tier = None
    if "--tier" in sys.argv:
        tier = int(sys.argv[sys.argv.index("--tier") + 1])
    skip = {"--fixtures", "--compare", "--online", "--online-all", "--tier"}
    if "--tier" in sys.argv:
        skip.add(str(tier))
    args = [a for a in sys.argv[1:] if a not in skip]
    if len(sys.argv) > 1 and sys.argv[1] == "--fixtures":
        bench_fixtures(compare=compare, online=online, tier=tier)
    elif online and (compare or "--online-all" in sys.argv):
        bench_fixtures(compare=compare or "--online-all" in sys.argv, online=True, tier=tier)
    elif args and args[0] == "--compare-matrix":
        df = pd.read_csv(FIXTURES_DIR / (args[1] if len(args) > 1 else "aqi_sample.csv"))
        bench_compare_table(df, args[1] if len(args) > 1 else "aqi_sample")
    else:
        sizes = [int(a.replace("_", "")) for a in args] or [10_000, 100_000, 1_000_000]
        for size in sizes:
            bench_synthetic(size, compare=compare)
