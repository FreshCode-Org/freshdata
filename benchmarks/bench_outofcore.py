"""Peak-RSS evidence for the out-of-core claims (issues #52 / #53).

Measures, in an isolated subprocess per scenario, the peak resident set size
of cleaning a parquet fixture through each engine/output_format combination:

    engine=duckdb  output_format=pandas       (default: materializes at the end)
    engine=duckdb  output_format=duckdb       (un-fetched relation handle)
    engine=polars  output_format=pandas       (default: materializes at the end)
    engine=polars  output_format=polars-lazy  (un-collected LazyFrame handle)

The fixture is written batch-by-batch so generating it never holds the whole
frame in memory. Each scenario runs in its own subprocess so allocator reuse
cannot blur the numbers; peak RSS is read from ``resource.ru_maxrss`` at exit.

Usage::

    python benchmarks/bench_outofcore.py --rows 2000000

Not part of the test suite: RSS numbers are machine-specific evidence, not
assertions.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCENARIOS = [
    ("duckdb", "pandas"),
    ("duckdb", "duckdb"),
    ("polars", "pandas"),
    ("polars", "polars-lazy"),
]

_BATCH = 250_000


def make_fixture(path: Path, rows: int) -> None:
    """Write a mixed-dtype parquet fixture in bounded batches."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    rng = np.random.default_rng(0)
    written = 0
    while written < rows:
        n = min(_BATCH, rows - written)
        idx = np.arange(written, written + n)
        df = pd.DataFrame(
            {
                "id": idx,
                "amount": rng.normal(100.0, 25.0, n),
                "count": rng.integers(0, 1_000, n),
                "city": pd.Series(idx % 500).map("city_{:03d}".format),
                "note": pd.Series(idx % 1_000).map("note text {:04d} padding".format),
            }
        )
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(path, table.schema)
        writer.write_table(table)
        written += n
    if writer is not None:
        writer.close()


def child(engine: str, output_format: str, fixture: str) -> None:
    """Run one clean in this process and print a JSON result line."""
    import resource

    import freshdata as fd

    start = time.perf_counter()
    # fix_dtypes uses sampled pandas heuristics, so it forces a fallback that
    # materializes the frame; disabling it is part of the native-handle recipe.
    result, report = fd.clean(
        fixture,
        engine=engine,
        output_format=output_format,
        strategy="conservative",
        fix_dtypes=False,
        return_report=True,
    )
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":  # linux reports KiB, macOS bytes
        peak *= 1024
    print(
        json.dumps(
            {
                "engine": engine,
                "output_format": output_format,
                "materialized": report.materialized,
                "peak_rss_mb": round(peak / 1e6, 1),
                "seconds": round(elapsed, 2),
                "result_type": type(result).__name__,
            }
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--fixture", help="reuse an existing parquet fixture")
    ap.add_argument("--_child", nargs=3, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._child:
        child(*args._child)
        return

    with tempfile.TemporaryDirectory() as tmp:
        fixture = args.fixture or str(Path(tmp) / "fixture.parquet")
        if not args.fixture:
            print(f"generating {args.rows:,}-row fixture ...", file=sys.stderr)
            make_fixture(Path(fixture), args.rows)

        results = []
        for engine, fmt in SCENARIOS:
            proc = subprocess.run(
                [sys.executable, __file__, "--_child", engine, fmt, fixture],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                print(f"{engine}/{fmt} FAILED:\n{proc.stderr}", file=sys.stderr)
                continue
            results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    print(f"\n{'engine':<8} {'output_format':<13} {'materialized':<13} "
          f"{'peak RSS (MB)':>13} {'seconds':>8}")
    for r in results:
        print(f"{r['engine']:<8} {r['output_format']:<13} "
              f"{str(r['materialized']):<13} {r['peak_rss_mb']:>13} {r['seconds']:>8}")


if __name__ == "__main__":
    main()
