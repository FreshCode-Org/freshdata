# Scalable execution backends

`freshdata` is pandas-first, but the same clean can run on **Polars**, **DuckDB**,
**Spark**, or the optional **FreshCore** native engine. Every backend produces
the **same `CleanReport` audit contract** — identical action schema
(`step`, `column`, `count`, `rationale`, `risk`, `confidence`) — so downstream
consumers (`compliance`, `integrations`, trust scoring) work unchanged.

```python
import freshdata as fd

# in-memory pandas (default, unchanged)
clean = fd.clean(df)

# scale-out engines, result materialized to a pandas frame
clean = fd.clean("data.parquet", engine="duckdb", output_format="pandas")
clean = fd.clean(polars_df, engine="polars")
clean = fd.clean(spark_df,  engine="spark")          # or engine="auto"
clean = fd.clean(df,        engine="freshcore")      # optional native extension

# honest out-of-core: keep a native, un-materialized handle (you decide when to pull rows)
rel = fd.clean("data.parquet", engine="duckdb", output_format="duckdb")        # DuckDBPyRelation
lf  = fd.clean(polars_df,      engine="polars", output_format="polars-lazy")   # pl.LazyFrame
```

### What "out-of-core" honestly means here

DuckDB and Polars can **spill to disk during the cleaning pipeline**, but the
*output format* decides whether the cleaned result is then pulled fully into
memory:

| `output_format` | Returns | Materializes whole result? |
|-----------------|---------|----------------------------|
| `"pandas"` (default) | `pandas.DataFrame` | **Yes** — `fetchdf()` / `collect()` |
| `"polars"` / `"arrow"` / `"spark"` | eager frame / Arrow table / Spark frame | **Yes** |
| `"duckdb"` | `DuckDBPyRelation` (un-fetched) | **No** — you call `.fetchdf()`/`.arrow()` |
| `"polars-lazy"` | `pl.LazyFrame` (un-collected) | **No** — you call `.collect()` |

Only the `"duckdb"` and `"polars-lazy"` handles are larger-than-RAM safe end to
end: nothing is fetched/collected until *you* ask. When a clean returns a native
handle, `report.materialized` is `False` and `report.summary()` says so plainly.
If the requested strategy needs the pandas decision engine (e.g. `balanced`/
`aggressive` imputation, dtype heuristics), the backend transparently falls back
to pandas — that fallback is recorded in `report.fallback_events`, and the result
is materialized. Use `strategy="conservative"` (deterministic representation
repair + structural reduction) to keep the native handle.

**Semantic cleaning stays native too.** When `semantic_mode` is enabled on a
Polars/DuckDB engine, the semantic stage runs over a *natively extracted*
distinct table (a bounded `GROUP BY`) and maps repairs back with `replace`/SQL —
the full frame is never pulled into pandas just to inspect values. See the
[native-engine semantic notes](limitations.md#native-engine-semantic-cleaning)
for the representation edges (e.g. partially-mapped boolean columns).

The `StreamingCleaner` micro-batch path (see *Streaming*) is the other genuinely
out-of-core route: rows are processed one bounded batch at a time and never
concatenated.

The **pandas backend is the reference implementation.** Native backends reproduce
the deterministic subset directly; anything outside it is delegated to pandas and
recorded in `report.fallback_events`. Every native step records a
`report.backend_differences` entry when its statistics (e.g. quantile
interpolation) can differ from the pandas reference.

## Selecting a backend

`engine="auto"` resolves a concrete backend from the input:

| Input                         | `auto` picks |
|-------------------------------|--------------|
| Spark `DataFrame`             | `spark`      |
| `.parquet` / `.csv` path      | `duckdb`     |
| Polars `DataFrame`/`LazyFrame`| `polars`     |
| Arrow `Table` / `RecordBatch` | `polars` (else `duckdb`) |
| DuckDB relation               | `duckdb`     |
| pandas `DataFrame`            | sized: `pandas` → `polars` → `duckdb` |

`EngineConfig` controls execution (never *what* is cleaned):

```python
from freshdata.execution import EngineConfig

cfg = EngineConfig(engine="duckdb", memory_limit_gb=4, temp_directory="/tmp/spill")
cfg = EngineConfig(engine="spark", spark_shuffle_partitions=200, output_format="spark")
```

PySpark is an **optional dependency** (`pip install 'freshdata-cleaner[spark]'`) and
also needs a JVM at runtime. Importing `freshdata` never imports pyspark.

FreshCore is also optional. Install the Python package normally, then build the
native extension from the repo checkout:

```bash
pip install -e ".[dev,freshcore]"
maturin develop --manifest-path crates/freshcore/Cargo.toml --features extension-module
python benchmarks/bench_freshcore.py --rows 10000 100000 --workload full
```

If `engine="freshcore"` is requested but the native module is not installed,
FreshData delegates to the pandas reference pipeline and records the reason in
`report.fallback_events`.

## Backend support matrix

`native` = run by the backend itself; `fallback` = delegated to the pandas
reference (output identical, recorded in `report.fallback_events`); `unsupported`
= not applicable to that engine.

| Step (config)                          | pandas | polars | duckdb | spark | freshcore |
|----------------------------------------|--------|--------|--------|-------|-----------|
| `column_names` (snake_case rename)     | native | native | native | native | native |
| `strip_whitespace`                     | native | native | native | native | native |
| `normalize_case` (`string_case`)       | native | fallback | fallback | fallback | native |
| `normalize_sentinels`                  | native | native | native | native | native |
| `drop_empty_columns` / `drop_empty_rows` | native | native | native | native | native |
| `drop_duplicates` (full-row, keep first/last) | native | native | native | native | native |
| `impute` = mean / median / mode / auto | native | native | native | native | native |
| `impute_method="missforest"` / `impute_strategy` | native | fallback | fallback | fallback | fallback |
| `outliers` with `outlier_method="iqr"`/`"zscore"` (clip/flag) | native | native | native | native | native |
| `outliers` with `outlier_method="isolation_forest"` | native | fallback | fallback | fallback | fallback |
| `outliers` with `outlier_method="auto"` (skew-based) | native | fallback | fallback | fallback | fallback |
| `drop_duplicates` with a `duplicate_subset` | native | fallback | fallback | fallback | fallback |
| `duplicate_keep` = `drop` / `aggregate` | native | fallback | fallback | fallback | fallback |
| `fix_dtypes` (sampled heuristics)      | native | fallback | fallback | fallback | partial native |
| `drop_constant_columns`                | native | fallback | fallback | fallback | fallback |
| `optimize_memory` (downcasting)        | native | fallback | fallback | fallback | fallback |
| Decision engine (`strategy="balanced"`/`"aggressive"`) | native | fallback | fallback | fallback | fallback |
| Missing-indicator columns (`missing_indicators`) | engine-only | fallback | fallback | fallback | fallback |
| `output_format`                        | pandas | pandas/polars/arrow | pandas/arrow | spark/pandas | pandas/polars/arrow |

Notes:

- **Imputation counts** are exact across backends (the number of filled cells is
  unambiguous); the fill *value* for `median`/`mode` can differ slightly because
  each engine uses its own quantile interpolation / tie-breaking. Polars and
  DuckDB use linear-interpolated quantiles matching pandas; Spark uses
  `approxQuantile`. Such divergences are recorded in `report.backend_differences`.
- **Outlier counts** match the pandas reference where the quantile statistics
  match (Polars/DuckDB linear interpolation); Spark may flag a different count.
- A non-default pandas index (e.g. a `DatetimeIndex`) forces a pandas fallback,
  since native frames carry no index.
- FreshCore v1 is a cleaning-first native engine, not an out-of-core engine. It
  supports pandas-compatible materialized outputs and records per-stage timings
  in `report.stage_timings`.

## Arrow interoperability

Arrow `Table` and `RecordBatch` are first-class inputs. DuckDB scans Arrow
natively (zero-copy) and Polars uses `from_arrow`, so no pandas materialization
happens on the way in. Round-trip Arrow in → Arrow out:

```python
table = fd.clean(arrow_table, engine="duckdb", output_format="arrow")
```

## Command line

```bash
freshdata clean input.parquet --engine spark
freshdata clean input.parquet --engine duckdb --memory-limit-gb 4
freshdata clean input.csv --engine polars
```

Non-pandas `--engine` values run the scalable path: the file is read by the
backend (DuckDB/Polars scan in place; Spark uses its own readers), and the
cleaned frame plus a `CleanReport` summary are emitted. `--report report.json`
writes the full report (including `backend`, `fallback_events`, and
`backend_differences`).
