"""Scale benchmark for the native distinct-value semantic path (Phase 6).

The semantic stage only reasons about a column's *distinct values*. On a native
engine we extract that bounded distinct table natively (Polars group-by / DuckDB
GROUP BY), score it through the pandas gate, and map repairs back natively — so
peak memory tracks the *distinct* cardinality, not the row count. This benchmark
generates a wide, mostly-low-cardinality frame and compares:

  * pandas       — reference path; whole frame in memory, semantics in pandas
  * polars-lazy  — native distinct path; frame scanned lazily from parquet
  * duckdb       — native distinct path; frame scanned from parquet

For each it records rows/sec, peak RSS, and confirms the same repairs landed
(boolean synonyms -> Boolean, spelled numbers -> Int). The headline property:
polars-lazy / duckdb peak RSS stays far below the pandas path as ``--rows`` grows.

    python benchmarks/bench_native_semantic.py --rows 1000000
    python benchmarks/bench_native_semantic.py --rows 10000000 --engines polars-lazy,duckdb
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from freshdata.config import CleanConfig
from freshdata.execution import run_with_engine

# A native-compatible config: conservative engine + no sampled dtype fixes, so
# the deterministic stages run natively and only the semantic stage is measured.
CONFIG = CleanConfig(
    strategy="conservative", fix_dtypes=False, semantic_mode="auto", verbose=False
)


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss / 1e6 if sys.platform == "darwin" else maxrss / 1e3


def write_parquet(path: str, rows: int, batch: int, seed: int = 0) -> None:
    """Write ``rows`` rows to parquet in batches — never all in memory at once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    active = np.array(["yes", "y", "YES", "no", "n", "NO"], dtype=object)
    qty = np.array(["one", "two", "three", "4", "5", "6"], dtype=object)
    region = np.array(["North", "north", "NORTH", "South", "south"], dtype=object)
    writer: pq.ParquetWriter | None = None
    produced = 0
    try:
        while produced < rows:
            n = min(batch, rows - produced)
            tbl = pa.table(
                {
                    "active": pa.array(rng.choice(active, n)),
                    "qty": pa.array(rng.choice(qty, n)),
                    "region": pa.array(rng.choice(region, n)),
                    "user_id": pa.array([f"U{i}" for i in range(produced, produced + n)]),
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(path, tbl.schema)
            writer.write_table(tbl)
            produced += n
    finally:
        if writer is not None:
            writer.close()


def _run_pandas(path: str) -> tuple[pd.DataFrame, float, float]:
    t0 = time.perf_counter()
    df = pd.read_parquet(path)
    out = run_with_engine(df, CONFIG, engine="pandas", output_format="pandas")
    return out, time.perf_counter() - t0, _rss_mb()


def _run_polars_lazy(path: str) -> tuple[pd.DataFrame, float, float]:
    import polars as pl

    t0 = time.perf_counter()
    lf = pl.scan_parquet(path)
    out = run_with_engine(lf, CONFIG, engine="polars", output_format="polars-lazy")
    frame = out.collect(engine="streaming").to_pandas()
    return frame, time.perf_counter() - t0, _rss_mb()


def _run_duckdb(path: str) -> tuple[pd.DataFrame, float, float]:
    import duckdb  # noqa: F401 - ensure the optional dependency is present

    t0 = time.perf_counter()
    # Pass the parquet path: the engine reads it through a lazy view, so the
    # frame is never fully materialized before the native distinct pass.
    out = run_with_engine(path, CONFIG, engine="duckdb", output_format="duckdb")
    frame = out.df()
    return frame, time.perf_counter() - t0, _rss_mb()


_RUNNERS = {
    "pandas": _run_pandas,
    "polars-lazy": _run_polars_lazy,
    "duckdb": _run_duckdb,
}


def _repairs_ok(frame: pd.DataFrame) -> bool:
    active = set(frame["active"])
    qty = {int(v) for v in frame["qty"]}
    return active <= {True, False} and qty == {1, 2, 3, 4, 5, 6}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=250_000)
    ap.add_argument("--engines", default="pandas,polars-lazy,duckdb")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    import tempfile

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=True) as tmp:
        write_parquet(tmp.name, args.rows, args.batch_size)
        results = []
        for engine in engines:
            runner = _RUNNERS.get(engine)
            if runner is None:
                print(f"skip unknown engine {engine!r}", file=sys.stderr)
                continue
            try:
                frame, elapsed, peak_rss = runner(tmp.name)
            except ImportError as exc:
                print(f"skip {engine}: {exc}", file=sys.stderr)
                continue
            rec = {
                "engine": engine,
                "rows": args.rows,
                "seconds": round(elapsed, 3),
                "rows_per_second": round(args.rows / elapsed) if elapsed else None,
                "peak_rss_mb": round(peak_rss, 1),
                "repairs_ok": _repairs_ok(frame),
            }
            results.append(rec)
            print(json.dumps(rec))

    payload = {"benchmark": "native_semantic", "rows": args.rows, "results": results}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
    # Every engine must produce the same repairs.
    if not all(r["repairs_ok"] for r in results):
        print("FAIL: repair mismatch across engines", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
