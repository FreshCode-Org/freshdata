---
title: Streaming / micro-batch cleaning
description: >-
  Clean datasets larger than memory with freshdata's StreamingCleaner: bounded running
  statistics, per-batch trust scoring and schema-drift detection, and a stable-memory
  100M-row out-of-core benchmark.
keywords: streaming data cleaning, out-of-core data quality, micro-batch cleaning, pandas chunked cleaning, kafka data quality, arrow flight, reservoir median, welford streaming
---

# Streaming / micro-batch cleaning

`fd.StreamingCleaner` cleans an **unbounded stream of DataFrame batches in constant
memory**. It reuses the same in-memory pipeline for representation repair, the same
[`CleanReport`](cleaning-engine.md) and `Action` audit records, and the same 0-100
[Data Trust Score](feature-overview.md) you already know from `fd.clean` — but the
*statistical* decisions (which value to impute) are driven by **bounded running
statistics** maintained across batches instead of one batch in isolation.

!!! info "Micro-batch, not row-by-row"
    Streaming mode is **micro-batch**: you feed it batches (a chunked CSV reader, a
    Kafka poll, an Arrow `RecordBatchReader`), and it cleans and emits one batch at a
    time. It is *not* a true row-by-row real-time engine, and it never concatenates the
    stream.

## Why it stays flat in memory

Every per-column accumulator is **O(1)** or **O(capacity)** — never O(rows):

| Statistic | Algorithm | Memory |
|---|---|---|
| mean / variance / std | Welford (Chan's parallel batch update) | 3 floats |
| median / quantiles | reservoir sampling (Vitter Algorithm R) | `quantile_reservoir_size` floats |
| mode / top-k categories | Space-Saving summary | `max_categories` keys |
| datetime min/max/order | running scalars | O(1) |

Feeding 100M rows costs the same memory as feeding 100k. Nothing retains the batch
after the running stats are updated.

## Quickstart

```python
import freshdata as fd
import pandas as pd

cleaner = fd.StreamingCleaner(
    target_column="churn",
    id_columns=("customer_id",),
    window_size=100_000,
    warmup_batches=3,
    strategy="balanced",
)

batches = pd.read_csv("events.csv", chunksize=100_000)   # never loads the full file
for cleaned_batch, report in cleaner.clean_batches(batches):
    write(cleaned_batch)
    log(report.to_dict())

final_report = cleaner.finalize()
```

Or drive it one micro-batch at a time:

```python
cleaned_batch, report = cleaner.clean_batch(df_or_arrow_or_polars)
state = cleaner.state_          # JSON-friendly snapshot of the running statistics
```

## Warmup vs. stable phase

The cleaner imputes in two phases so it never fills from statistics it does not yet have:

1. **Warmup** (the first `warmup_batches` batches). Only safe representation-level
   repairs run — column normalization, whitespace stripping, sentinel normalization,
   safe dtype repair, within-batch duplicate handling. Statistical imputation is
   **deferred** and every deferral is recorded as an auditable `Action`.
2. **Stable.** Missing values are imputed from the running/window statistics:
    - **numeric** — running median for skewed/outlier-bearing columns, running mean when
      the distribution is approximately symmetric;
    - **categorical** — the running mode when it is confidently dominant, otherwise a
      visible `"Unknown"` / `"Missing"` sentinel;
    - **datetime** — forward/backward fill **only** within a usably ordered column;
      otherwise the gap is preserved rather than inventing timestamps;
    - **high-missing** columns are preserved with a warning (same balanced-mode behavior
      as `fd.clean`).

The leakage-aware safety gates are identical to the in-memory cleaner: **ID columns are
never imputed, the target column is never modified, free-text is never force-filled**,
and domain-sensitive outliers are not blindly removed.

## Per-batch report fields

When a report comes from streaming mode, `report.streaming` (and the `"streaming"` key
in `report.to_dict()`) carries:

| Field | Meaning |
|---|---|
| `batch_id` | 1-based index of this batch |
| `rows_in_batch` / `rows_seen_total` | rows in this batch / cumulative across the stream |
| `batch_trust_score` | trust score of this cleaned batch (0-100) |
| `rolling_trust_score` | row-weighted trust over the recent window |
| `cumulative_trust_score` | row-weighted trust over the whole stream |
| `schema_drift_detected` | whether drift was flagged for this batch |
| `warmup_phase` | whether this batch was still in warmup |
| `trust_gate_passed` | `fail_under_trust` result for this batch |

Normal (non-streaming) `CleanReport`s are unchanged — no `streaming` key is emitted, so
existing serialization is fully backward compatible.

## Schema & distribution drift

Each batch is compared against the locked schema baseline and the running state, cheaply
(no extra pass over history). Drift surfaces as both a `drift` `Action` and a warning:

- a new column appears, or a baseline column disappears;
- a column's dtype changes;
- its missing ratio jumps sharply (`drift_missing_jump`);
- its cardinality explodes (`drift_cardinality_factor`);
- its numeric mean shifts past `drift_zscore` σ from the running mean.

## Input formats

`clean_batches` accepts any iterable of:

- **pandas** `DataFrame`;
- **PyArrow** `Table` / `RecordBatch` (when `pyarrow` is installed) — also via
  `cleaner.clean_arrow_batches(reader)`;
- **polars** `DataFrame` / `LazyFrame` (when `polars` is installed).

### Optional source connectors

These live behind extras so the base install stays dependency-free. If the dependency is
missing, the helper raises a clear `ImportError` naming the extra.

```bash
pip install "freshdata-cleaner[kafka]"     # kafka-python
pip install "freshdata-cleaner[flight]"    # pyarrow with the flight module
```

```python
for cleaned, report in cleaner.clean_kafka(
    topic="events", bootstrap_servers="localhost:9092",
    value_format="json", batch_size=10_000,
):
    ...

for cleaned, report in cleaner.clean_arrow_flight(
    "grpc://localhost:8815", batch_size=100_000,
):
    ...
```

## CLI

The `freshdata` CLI gains streaming subcommands. Input is read batch-by-batch
(`read_csv(chunksize=)` / `ParquetFile.iter_batches`) and output is written
batch-by-batch, so a 100M-row file is never held in memory. With `--report`, a JSON
report is written per batch plus a final `summary.json`; `--fail-under-trust` sets the
process exit code.

```bash
# Chunked CSV/Parquet
freshdata stream events.csv --batch-size 100000 -o out.parquet --report reports/ \
    --target-column churn --id-columns customer_id --fail-under-trust 80

# Kafka topic
freshdata stream-kafka --topic events --bootstrap-servers localhost:9092 \
    --batch-size 10000 --report reports/

# Stable-memory benchmark
freshdata benchmark-stream --rows 100000000 --batch-size 100000 --cols 20 \
    --report benchmark.json
```

## The 100M-row stable-memory benchmark

`benchmarks/bench_streaming.py` is an **out-of-core / streaming** test, *not* a
single-DataFrame test. It generates synthetic rows lazily, one batch at a time, feeds
them through a `StreamingCleaner`, and tracks peak RSS (via `psutil`, falling back to the
`resource` module).

```bash
python benchmarks/bench_streaming.py --rows 100000000 --batch-size 100000 --cols 20
```

It reports total rows processed, batch size, columns, elapsed time, rows/sec, baseline
and peak memory, steady-state memory growth, and the final cumulative trust score.
**Acceptance** is that steady-state memory does not grow with rows — peak memory stays
bounded relative to the *batch* size, not the dataset size — and it fails loudly if
fewer rows are processed than requested.

!!! warning "Honest limitations"
    - Streaming mode is **micro-batch**, not true row-by-row real time.
    - **Global** cross-batch duplicate detection is limited: by default duplicates are
      scoped within a batch; enabling `global_duplicates` uses a *bounded recent-window*
      that can miss duplicates older than the window.
    - Medians/quantiles are **approximate** (reservoir-sampled), and the top-k category
      summary is approximate when a column saturates `max_categories`.
    - Kafka and Arrow Flight are **optional** integrations.
    - The enterprise-scale "clean ≥100M rows out-of-core with stable memory" claim
      depends on the benchmark above passing in **your** environment.
