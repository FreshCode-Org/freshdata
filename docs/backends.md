# Scalable execution backends

`freshdata` is pandas-first, but the same clean can run on **Polars**, **DuckDB**,
or **Spark** for larger-than-RAM or distributed workloads. Every backend produces
the **same `CleanReport` audit contract** — identical action schema
(`step`, `column`, `count`, `rationale`, `risk`, `confidence`) — so downstream
consumers (`compliance`, `integrations`, trust scoring) work unchanged.

```python
import freshdata as fd

# in-memory pandas (default, unchanged)
clean = fd.clean(df)

# out-of-core / distributed
clean = fd.clean("data.parquet", engine="duckdb", output_format="pandas")
clean = fd.clean(polars_df, engine="polars")
clean = fd.clean(spark_df,  engine="spark")          # or engine="auto"
```

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

## Backend support matrix

`native` = run by the backend itself; `fallback` = delegated to the pandas
reference (output identical, recorded in `report.fallback_events`); `unsupported`
= not applicable to that engine.

| Step (config)                          | pandas | polars | duckdb | spark |
|----------------------------------------|--------|--------|--------|-------|
| `column_names` (snake_case rename)     | native | native | native | native |
| `strip_whitespace`                     | native | native | native | native |
| `normalize_sentinels`                  | native | native | native | native |
| `drop_empty_columns` / `drop_empty_rows` | native | native | native | native |
| `drop_duplicates` (full-row, keep first/last) | native | native | native | native |
| `impute` = mean / median / mode / auto | native | native | native | native |
| `outliers` with `outlier_method="iqr"`/`"zscore"` (clip/flag) | native | native | native | native |
| `outliers` with `outlier_method="isolation_forest"` | native | fallback | fallback | fallback |
| `outliers` with `outlier_method="auto"` (skew-based) | native | fallback | fallback | fallback |
| `drop_duplicates` with a `duplicate_subset` | native | fallback | fallback | fallback |
| `duplicate_keep` = `drop` / `aggregate` | native | fallback | fallback | fallback |
| `fix_dtypes` (sampled heuristics)      | native | fallback | fallback | fallback |
| `drop_constant_columns`                | native | fallback | fallback | fallback |
| `optimize_memory` (downcasting)        | native | fallback | fallback | fallback |
| Decision engine (`strategy="balanced"`/`"aggressive"`) | native | fallback | fallback | fallback |
| Missing-indicator columns (`missing_indicators`) | engine-only | fallback | fallback | fallback |
| `output_format`                        | pandas | pandas/polars/arrow | pandas/arrow | spark/pandas |

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
