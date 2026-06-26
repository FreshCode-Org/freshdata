"""Stable-memory benchmark for freshdata streaming / micro-batch cleaning.

This is an **out-of-core / streaming** test, not a single-DataFrame test: synthetic
rows are generated lazily, one batch at a time, and fed to a
:class:`freshdata.StreamingCleaner`. The whole dataset is never materialized, so peak
memory should stay roughly flat as ``--rows`` grows — that flatness is the property the
"clean 100M rows out-of-core with stable memory" claim depends on.

    python benchmarks/bench_streaming.py --rows 100000000 --batch-size 100000 --cols 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator

import numpy as np
import pandas as pd

import freshdata as fd


def _rss_mb() -> float:
    """Resident set size in MB (psutil if available, else the resource module)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss / 1e6 if sys.platform == "darwin" else maxrss / 1e3  # bytes vs KB


def synth_batches(rows: int, batch_size: int, cols: int, *, seed: int = 0
                  ) -> Iterator[pd.DataFrame]:
    """Yield ``rows`` rows as lazily-generated batches — never all at once."""
    rng = np.random.default_rng(seed)
    produced = 0
    t0 = np.datetime64("2020-01-01")
    while produced < rows:
        n = min(batch_size, rows - produced)
        data: dict[str, object] = {
            "customer_id": np.arange(produced, produced + n, dtype="int64"),
            "churn": rng.integers(0, 2, n),
            "event_time": t0 + np.arange(produced, produced + n).astype("timedelta64[s]"),
            "region": rng.choice(["north", "north", "north", "south", "east", None], n),
        }
        for j in range(max(0, cols - len(data))):
            vals = rng.normal(50, 12, n)
            vals[rng.random(n) < 0.1] = np.nan          # ~10% missing
            vals[rng.random(n) < 0.005] *= 40           # rare outliers
            data[f"metric_{j}"] = vals
        produced += n
        yield pd.DataFrame(data)


def run_benchmark(*, rows: int, batch_size: int, cols: int) -> dict:
    """Stream ``rows`` rows through a StreamingCleaner and measure memory/throughput."""
    cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                                  window_size=batch_size, warmup_batches=3)
    base = _rss_mb()
    peak = base
    growth_samples: list[float] = []
    processed = 0
    start = time.perf_counter()
    for cleaned, report in cleaner.clean_batches(synth_batches(rows, batch_size, cols)):
        processed += len(cleaned)
        rss = _rss_mb()
        peak = max(peak, rss)
        # Sample post-warmup RSS so growth reflects the steady state, not startup.
        if report.streaming["batch_id"] > 3:
            growth_samples.append(round(rss, 1))
    elapsed = time.perf_counter() - start
    final = cleaner.finalize()

    if processed < rows:
        raise SystemExit(f"FAIL: processed {processed:,} of {rows:,} requested rows")

    # Stable memory = the steady-state *tail* is flat. The allocator ramps to a
    # working-set ceiling over the first batches (not a function of total rows);
    # measuring the second half of the run excludes that ramp.
    tail = growth_samples[len(growth_samples) // 2:] or growth_samples
    steady_growth = round(max(tail) - min(tail), 1) if tail else 0.0
    return {
        "rows_processed": processed,
        "batch_size": batch_size,
        "columns": cols,
        "elapsed_seconds": round(elapsed, 2),
        "rows_per_second": int(processed / elapsed) if elapsed else 0,
        "baseline_memory_mb": round(base, 1),
        "peak_memory_mb": round(peak, 1),
        "warmup_ramp_mb": round(peak - base, 1),
        "steady_state_growth_mb": steady_growth,
        "steady_state_samples_mb": tail,
        "final_cumulative_trust_score": final.streaming["cumulative_trust_score"],
        # Flat tail (no linear growth with rows) is the acceptance criterion.
        "stable_memory": steady_growth < max(50.0, 0.15 * peak),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freshdata streaming stable-memory benchmark")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--cols", type=int, default=20)
    parser.add_argument("--report", metavar="FILE", help="write the benchmark JSON here")
    args = parser.parse_args(argv)

    result = run_benchmark(rows=args.rows, batch_size=args.batch_size, cols=args.cols)
    text = json.dumps(result, indent=2)
    if args.report:
        with open(args.report, "w") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
