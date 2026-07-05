#!/usr/bin/env python
"""Reproducible benchmarks for the freshdata Strategic Report surfaces.

Five honest, runnable benchmarks — no numbers are committed to the repo; you run
them in *your* environment and read the JSON it writes.

    python benchmarks/bench_report.py --all
    python benchmarks/bench_report.py csv_ingest --mb 100
    python benchmarks/bench_report.py profile --rows 1_000_000
    python benchmarks/bench_report.py nullfill --rows 10_000_000
    python benchmarks/bench_report.py import_time
    python benchmarks/bench_report.py memory --rows 1_000_000

Where applicable each case is measured for ``strategy="balanced"`` (default,
accuracy-first) and ``strategy="aggressive"`` (zero-NaN scrub) so the trade-off
is visible. Results are written to ``benchmarks/results/report_bench.json``.

Requires ``freshdata-cleaner[bench]`` (pyarrow + psutil). Memory figures need psutil.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESULTS = os.path.join(_HERE, "results")
_DATA = os.environ.get("FRESHDATA_BENCH_DATA", "/tmp/freshdata_bench")

# Reuse the packaged data generator + peak-RSS sampler.
from freshdata.benchmarks._data_gen import generate_parquet  # noqa: E402
from freshdata.benchmarks._harness import _PeakRSS  # noqa: E402

STRATEGIES = ("balanced", "aggressive")


def _ensure_parquet(n_rows: int) -> str:
    os.makedirs(_DATA, exist_ok=True)
    path = os.path.join(_DATA, f"report_{n_rows}.parquet")
    if not os.path.exists(path):
        print(f"  generating {n_rows:,} rows -> {path}", flush=True)
        generate_parquet(n_rows, path)
    return path


def _csv_from_parquet(n_rows: int) -> str:
    import pandas as pd

    path = os.path.join(_DATA, f"report_{n_rows}.csv")
    if not os.path.exists(path):
        pq = _ensure_parquet(n_rows)
        pd.read_parquet(pq).to_csv(path, index=False)
    return path


def _time(fn) -> tuple[float, float]:
    """Return ``(wall_seconds, peak_rss_increase_mb)`` for *fn*."""
    gc.collect()
    with _PeakRSS() as mem:
        t0 = time.perf_counter()
        fn()
        wall = time.perf_counter() - t0
    return round(wall, 3), round(mem.peak_increase_mb, 1)


def bench_csv_ingest(mb: int = 100) -> list[dict]:
    """~`mb` MB CSV ingest + clean, balanced vs aggressive."""
    import freshdata as fd

    # ~rows for the target CSV size (the generator's mixed schema is ~100 B/row).
    n_rows = max(10_000, mb * 1_000_000 // 100)
    csv = _csv_from_parquet(n_rows)
    actual_mb = round(os.path.getsize(csv) / 1024 / 1024, 1)
    out = []
    for strat in STRATEGIES:
        wall, peak = _time(lambda s=strat: fd.clean_csv(csv, strategy=s))
        out.append({
            "benchmark": "csv_ingest", "strategy": strat, "rows": n_rows,
            "csv_mb": actual_mb, "wall_sec": wall, "peak_rss_mb": peak,
        })
    return out


def bench_profile(rows: int = 1_000_000) -> list[dict]:
    """1M-row mixed-schema profile (read-only)."""
    import pandas as pd

    import freshdata as fd

    df = pd.read_parquet(_ensure_parquet(rows))
    wall, peak = _time(lambda: fd.profile(df))
    return [{
        "benchmark": "profile", "strategy": "n/a", "rows": rows,
        "wall_sec": wall, "peak_rss_mb": peak,
    }]


def bench_nullfill(rows: int = 10_000_000) -> list[dict]:
    """10M-row null-fill / flag pass, balanced vs aggressive."""
    import pandas as pd

    import freshdata as fd

    df = pd.read_parquet(_ensure_parquet(rows))
    out = []
    for strat in STRATEGIES:
        wall, peak = _time(lambda s=strat: fd.clean(df.copy(), strategy=s))
        out.append({
            "benchmark": "nullfill", "strategy": strat, "rows": rows,
            "wall_sec": wall, "peak_rss_mb": peak,
        })
    return out


def bench_import_time() -> list[dict]:
    """Cold ``import freshdata`` wall time (median of 5 fresh subprocesses)."""
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        subprocess.run(
            [sys.executable, "-c", "import freshdata"],
            check=True, capture_output=True,
        )
        times.append(time.perf_counter() - t0)
    times.sort()
    return [{
        "benchmark": "import_time", "strategy": "n/a", "rows": 0,
        "import_sec_median": round(times[len(times) // 2], 4),
        "import_sec_min": round(times[0], 4),
    }]


def bench_memory(rows: int = 1_000_000) -> list[dict]:
    """Peak-RSS of a full clean, balanced vs aggressive (psutil required)."""
    import pandas as pd

    import freshdata as fd

    df = pd.read_parquet(_ensure_parquet(rows))
    base_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 1)
    out = []
    for strat in STRATEGIES:
        wall, peak = _time(lambda s=strat: fd.clean(df.copy(), strategy=s))
        out.append({
            "benchmark": "memory", "strategy": strat, "rows": rows,
            "input_mb": base_mb, "wall_sec": wall, "peak_rss_mb": peak,
        })
    return out


def _write(results: list[dict]) -> str:
    os.makedirs(_RESULTS, exist_ok=True)
    path = os.path.join(_RESULTS, "report_bench.json")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "results": results,
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark", nargs="?", default="all",
                   choices=["all", "csv_ingest", "profile", "nullfill",
                            "import_time", "memory"])
    p.add_argument("--all", action="store_true", help="run every benchmark")
    p.add_argument("--mb", type=int, default=100)
    p.add_argument("--rows", type=int, default=None)
    args = p.parse_args(argv)

    which = "all" if args.all else args.benchmark
    results: list[dict] = []
    if which in ("all", "csv_ingest"):
        print("csv_ingest ...", flush=True)
        results += bench_csv_ingest(args.mb)
    if which in ("all", "profile"):
        print("profile ...", flush=True)
        results += bench_profile(args.rows or 1_000_000)
    if which in ("all", "nullfill"):
        print("nullfill ...", flush=True)
        results += bench_nullfill(args.rows or 10_000_000)
    if which in ("all", "import_time"):
        print("import_time ...", flush=True)
        results += bench_import_time()
    if which in ("all", "memory"):
        print("memory ...", flush=True)
        results += bench_memory(args.rows or 1_000_000)

    path = _write(results)
    print(f"\nwrote {len(results)} result(s) -> {path}")
    for r in results:
        print("  ", json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
