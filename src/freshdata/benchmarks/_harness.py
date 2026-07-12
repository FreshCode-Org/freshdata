"""Benchmark harness: run ``fd.clean`` across engines and record time + memory.

Peak memory is measured as the increase in process resident set size (RSS) via
``psutil`` — unlike ``tracemalloc`` this captures the native allocations of
polars and duckdb, which is what matters for the out-of-core comparison.

Timing note: the pandas backend must load the dataset into memory first, so its
timing includes the read. The polars and duckdb backends receive the Parquet
*path* and stream it, so their timing reflects true out-of-core execution. This
asymmetry is intentional and is documented in the emitted report.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
from dataclasses import asdict, dataclass

from ._data_gen import generate_parquet

#: The cleaning config used for benchmarks. ``strategy="conservative"`` keeps the
#: work inside the native (out-of-core) subset so polars/duckdb stream rather than
#: fall back to pandas; ``fix_dtypes=False`` avoids the sampled dtype heuristics.
_BENCH_STRATEGY = "conservative"


@dataclass
class BenchmarkResult:
    engine: str
    n_rows: int
    wall_time_sec: float
    peak_memory_mb: float
    throughput_rps: float
    actions_count: int
    trust_score: float
    file_size_mb: float
    oom: bool = False
    error: str | None = None


class _PeakRSS:
    """Sample peak process RSS (MB) on a background thread."""

    def __init__(self, interval: float = 0.02) -> None:
        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._peak = 0
        self._baseline = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> _PeakRSS:
        gc.collect()
        self._baseline = self._proc.memory_info().rss
        self._peak = self._baseline
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            rss = self._proc.memory_info().rss
            self._peak = max(self._peak, rss)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_increase_mb(self) -> float:
        return max(0.0, (self._peak - self._baseline) / 1024 / 1024)


class BenchmarkHarness:
    """Run freshdata cleaning across engines and row counts."""

    SIZES = {
        "10k": 10_000,
        "100k": 100_000,
        "1m": 1_000_000,
        "10m": 10_000_000,
        "25m": 25_000_000,
        "100m": 100_000_000,
        "1b": 1_000_000_000,
    }

    def run(
        self,
        sizes: list[str],
        engines: list[str],
        data_dir: str = "/tmp/freshdata_bench",
        results_dir: str = "src/freshdata/benchmarks/results",
        write: bool = True,
    ) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for size_key in sizes:
            n_rows = self.SIZES[size_key]
            parquet_path = os.path.join(data_dir, f"bench_{size_key}.parquet")
            if not os.path.exists(parquet_path):
                print(f"Generating {size_key} rows -> {parquet_path} ...", flush=True)
                generate_parquet(n_rows, parquet_path)
            file_mb = os.path.getsize(parquet_path) / 1024 / 1024
            for engine in engines:
                print(f"  {engine} @ {size_key} ...", end=" ", flush=True)
                result = self._run_one(engine, n_rows, parquet_path, file_mb)
                results.append(result)
                if result.oom:
                    print("OOM")
                elif result.error:
                    print(f"ERROR ({result.error[:40]})")
                else:
                    print(f"{result.wall_time_sec:.2f}s  "
                          f"{result.peak_memory_mb:.0f}MB  "
                          f"{result.throughput_rps:,.0f} rows/s")
        if write:
            self._write_results(results, results_dir)
        return results

    def _run_one(
        self, engine: str, n_rows: int, parquet_path: str, file_mb: float
    ) -> BenchmarkResult:
        import freshdata as fd
        from freshdata.config import CleanConfig
        from freshdata.enterprise.metrics import compute_trust_score

        config = CleanConfig(strategy=_BENCH_STRATEGY, fix_dtypes=False, verbose=False)
        gc.collect()
        try:
            with _PeakRSS() as mem:
                t0 = time.perf_counter()
                if engine == "pandas":
                    import pandas as pd

                    source = pd.read_parquet(parquet_path)  # pandas must preload
                else:
                    source = parquet_path  # path-based: no preload
                cleaned, report = fd.clean(
                    source, config=config, engine=engine, return_report=True
                )
                len(cleaned)  # force materialisation for a fair comparison
                wall = time.perf_counter() - t0
            trust = compute_trust_score(
                cleaned if hasattr(cleaned, "isna") else cleaned.to_pandas()
            ).overall
            return BenchmarkResult(
                engine=engine,
                n_rows=n_rows,
                wall_time_sec=round(wall, 3),
                peak_memory_mb=round(mem.peak_increase_mb, 1),
                throughput_rps=round(n_rows / wall) if wall > 0 else 0,
                actions_count=len(report.actions),
                trust_score=round(float(trust), 2),
                file_size_mb=round(file_mb, 1),
            )
        except MemoryError:
            return BenchmarkResult(engine, n_rows, 0, 0, 0, 0, 0, round(file_mb, 1), oom=True)
        except Exception as exc:  # noqa: BLE001 - benchmarks must not crash the run
            return BenchmarkResult(
                engine, n_rows, 0, 0, 0, 0, 0, round(file_mb, 1), error=str(exc)
            )

    def _write_results(self, results: list[BenchmarkResult], results_dir: str) -> None:
        from datetime import datetime, timezone

        os.makedirs(results_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        json_path = os.path.join(results_dir, f"benchmark_{ts}.json")
        with open(json_path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        md_path = os.path.join(results_dir, f"benchmark_{ts}.md")
        with open(md_path, "w") as f:
            f.write("# freshdata out-of-core benchmark\n\n")
            f.write(f"Generated: {ts} UTC\n\n")
            f.write(
                "Config: `strategy=\"conservative\", fix_dtypes=False` (native subset).\n"
                "pandas timing includes the in-memory load; polars/duckdb stream the "
                "Parquet path (no preload).\n\n"
            )
            f.write(
                "| Engine | Rows | File (MB) | Wall (s) | Peak RSS (MB) | "
                "Throughput (rows/s) | Trust | Status |\n"
            )
            f.write(
                "|--------|-----:|----------:|---------:|--------------:|"
                "--------------------:|------:|--------|\n"
            )
            for r in results:
                status = "OOM" if r.oom else ("ERROR" if r.error else "OK")
                wall = "—" if (r.oom or r.error) else f"{r.wall_time_sec:.2f}"
                mem = "—" if (r.oom or r.error) else f"{r.peak_memory_mb:.0f}"
                tp = "—" if (r.oom or r.error) else f"{r.throughput_rps:,.0f}"
                trust = "—" if (r.oom or r.error) else f"{r.trust_score:.1f}"
                f.write(
                    f"| {r.engine} | {r.n_rows:,} | {r.file_size_mb:.1f} | {wall} | "
                    f"{mem} | {tp} | {trust} | {status} |\n"
                )
        print(f"\nResults written:\n  {json_path}\n  {md_path}")
