# freshdata out-of-core benchmark — reference run

Reproduce with::

    pip install -e ".[outofcore,bench]"
    python -m freshdata.benchmarks.run_benchmarks \
        --sizes 10k,100k,1m,10m,25m --engines pandas,polars,duckdb

Workload: `strategy="conservative", fix_dtypes=False` — the native (out-of-core)
subset: snake_case rename, whitespace trim, sentinel→null, empty column/row
drops, and full-row dedup. Timing for **pandas** includes the in-memory Parquet
load (that is its nature); **polars** and **duckdb** receive the Parquet *path*
and stream it. Peak RSS is the process resident-memory increase during the run.
Dataset: `freshdata.benchmarks._data_gen.generate_parquet` (seed 42; 10 numeric
+ 5 categorical + 2 PII + 2 text columns, 15% nulls, 5% sentinels).

Machine: Apple M2 (Mac14,2), 16 GB RAM, macOS 24.5.0.
Versions: Python 3.12.13, pandas 2.3.3, polars 1.42.1, duckdb 1.5.4, numpy 2.4.6.
Raw output: `results/benchmark_20260712_064336.{json,md}` (committed).

| Engine | Rows | File (MB) | Wall (s) | Peak RSS (MB) | Throughput (rows/s) | Trust |
|--------|-----:|----------:|---------:|--------------:|--------------------:|------:|
| pandas | 10,000 | 1.2 | 0.13 | 23 | 77,549 | 95.0 |
| polars | 10,000 | 1.2 | 0.14 | 68 | 72,703 | 95.0 |
| duckdb | 10,000 | 1.2 | 0.36 | 93 | 27,497 | 95.0 |
| pandas | 100,000 | 11.2 | 0.81 | 110 | 122,840 | 95.1 |
| polars | 100,000 | 11.2 | 0.27 | 198 | 368,825 | 95.1 |
| duckdb | 100,000 | 11.2 | 1.82 | 83 | 55,066 | 95.1 |
| pandas | 1,000,000 | 112.2 | 6.02 | 1,046 | 166,131 | 95.0 |
| polars | 1,000,000 | 112.2 | 1.93 | 1,018 | 517,391 | 95.0 |
| duckdb | 1,000,000 | 112.2 | 5.89 | 200 | 169,684 | 95.0 |
| pandas | 10,000,000 | 1,121.9 | 89.55 | 2,149 | 111,667 | 95.0 |
| polars | 10,000,000 | 1,121.9 | 41.25 | 5,895 | 242,398 | 95.0 |
| duckdb | 10,000,000 | 1,121.9 | 75.76 | 3,478 | 131,993 | 95.0 |
| pandas | 25,000,000 | 2,804.7 | 269.89 | 5,654 | 92,631 | 95.0 |
| polars | 25,000,000 | 2,804.7 | 225.68 | 8,984 | 110,778 | 95.0 |
| duckdb | 25,000,000 | 2,804.7 | 270.99 | 8,245 | 92,256 | 95.0 |

## Reading the numbers

- **Parity holds at scale.** The Data Trust Score is identical across all three
  backends at every size, 10k through 25M (95.0 / 95.1) — the Polars and DuckDB
  engines produce the same cleaned data and the same `CleanReport` as pandas.
- **Throughput:** Polars is the fastest at every size ≥100k (2–3× pandas at
  1M–10M; 10M rows in 41s vs 90s). DuckDB beats pandas at 1M–10M.
- **Memory at moderate scale:** DuckDB cleans 1M rows in **200 MB** vs pandas'
  **1,046 MB** (5× less) by streaming the Parquet scan.
- **The 25M caveat (honest):** on this 16 GB machine, the gaps *narrow* at 25M —
  polars 226s vs pandas 270s (1.2×, down from 2.2× at 10M), and polars/duckdb
  peak RSS is *higher* than pandas. Two run-specific reasons: (1) the harness
  converts the full result to pandas to compute the Trust Score, transiently
  doubling the result in memory; (2) with `memory_limit_gb=8` and results this
  large, the whole machine is under memory pressure and DuckDB has no
  instruction to spill earlier. Lowering `EngineConfig(memory_limit_gb=...)`
  forces spill-to-disk and trades wall time for footprint — that, plus the
  native-handle output formats (`output_format="duckdb"`/`"polars-lazy"`, which
  skip result materialization entirely — see `benchmarks/bench_outofcore.py`),
  is the lever for genuinely larger-than-RAM data. These numbers are the
  *materializing* path measured honestly, not the best case.

## 100M / 1B rows

The harness is parametrised up to `1b`. Those sizes need tens-to-hundreds of GB
of synthetic Parquet and were not generated here; run them on a box with the
disk to spare (`--sizes 100m,1b`). pandas is expected to OOM well before 1B rows
while the streaming/spilling backends complete. **No claims are made beyond the
25M evidence above.**
