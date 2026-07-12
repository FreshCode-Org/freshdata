"""CLI entry point for the freshdata out-of-core benchmark harness.

Example::

    python -m freshdata.benchmarks.run_benchmarks \\
        --sizes 10k,100k,1m,10m --engines pandas,polars,duckdb
"""

from __future__ import annotations

import argparse

from ._harness import BenchmarkHarness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="freshdata out-of-core benchmark runner")
    parser.add_argument(
        "--sizes", default="10k,100k,1m",
        help="comma-separated size keys: 10k,100k,1m,10m,25m,100m,1b",
    )
    parser.add_argument(
        "--engines", default="pandas,polars,duckdb",
        help="comma-separated engines: pandas,polars,duckdb",
    )
    parser.add_argument("--data-dir", default="/tmp/freshdata_bench")
    parser.add_argument("--results-dir", default="src/freshdata/benchmarks/results")
    args = parser.parse_args(argv)

    harness = BenchmarkHarness()
    unknown = [s for s in args.sizes.split(",") if s not in harness.SIZES]
    if unknown:
        parser.error(f"unknown size(s): {unknown}; choose from {list(harness.SIZES)}")

    results = harness.run(
        sizes=args.sizes.split(","),
        engines=args.engines.split(","),
        data_dir=args.data_dir,
        results_dir=args.results_dir,
    )
    ok = [r for r in results if not r.oom and not r.error]
    print(f"\n{len(ok)}/{len(results)} benchmark(s) completed cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
