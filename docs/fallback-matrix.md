# Backend fallback matrix

**The single most important row is the first one.** The default
`strategy="balanced"` runs the accuracy-first decision engine, which is
evaluated by the pandas backend — so with default options, **every native
engine delegates the whole pipeline to pandas** (recorded on
`report.fallback_events`). The fully native path is
`strategy="conservative"` with `fix_dtypes=False`.

This table is transcribed from the single source of truth,
`PlanGenerator.fallback_reason()` in `src/freshdata/execution/_plan.py`, plus
the input checks in `pandas_ingest_fallback_reason()`
(`src/freshdata/execution/_ingest.py`) — if you change either, change this page.
FreshCore also runs its own config and data checks in
`FreshCoreEngine._unsupported_reason()`
(`src/freshdata/execution/backends/_freshcore.py`); keep those rows in sync too.

| Operation / config | polars | duckdb | spark | freshcore | Why the fallback exists |
|---|---|---|---|---|---|
| `strategy="balanced"` / `"aggressive"` (default) | pandas | pandas | pandas | pandas | data-dependent decision engine; porting it per backend would fork its accuracy behaviour |
| column rename / whitespace / sentinels | native | native | native | native | — |
| empty row/column removal | native | native | native | native | — |
| full-row dedup (`keep="first"/"last"`) | native | native | native | native | streaming polars dedup drops row order (disclosed); `streaming_dedup=False` restores it. Spark keeps the first/last row in the input DataFrame's partition order |
| **subset dedup** (`duplicate_subset=`) | **native** | pandas | pandas | pandas | keep semantics are order-sensitive; Polars reproduces them via order-preserving `unique` (eager, not streaming — disclosed) |
| dedup `keep="drop"/"aggregate"` | pandas | pandas | pandas | pandas | group-wise resolution isn't expressed natively yet |
| detection-only dedup (`drop_duplicates=False`) with `duplicate_ratio_action="error"` | native | native | native | native (pandas with native modules that don't report `duplicates_detected`) | the escalation needs the duplicate-row count at the pandas dedup stage; FreshCore counts it natively, but modules built before that count existed would never raise |
| global impute mean/median/mode | native | native | native | native | — |
| impute `mode`/`auto` with missing values in a nullable `boolean` column | native | native | native | pandas | FreshCore v1 kernels do not impute boolean columns |
| impute `mode`/`auto` when `fix_dtypes` casts a text column (e.g. `"yes"`/`"no"`) to boolean and it still has missing values | native | native | native | pandas | same kernel gap as above, but the column only becomes boolean inside the native run, so the adapter checks the native result (`fallback_step="impute"`) and reruns on pandas |
| per-column `impute_strategy` | pandas | pandas | pandas | pandas | unimplemented natively (no fundamental blocker) |
| `impute="missforest"` | pandas | pandas | pandas | pandas | scikit-learn model |
| outliers `iqr` / `zscore` | native | native | native | native | — |
| outliers when a float column holds `±inf` | native | native | native | pandas | FreshCore v1 fences don't exclude non-finite values, so they flag or clip nothing |
| outliers when `fix_dtypes` casts a text column holding `"inf"`/`"-inf"` to float | native | native | native | pandas | same fence gap as above, but the infinity only appears inside the native run, so the adapter checks the native result (`fallback_step="outliers"`) and reruns on pandas |
| `outlier_action="auto"` / model methods | pandas | pandas | pandas | pandas | data-dependent / model-based selection |
| `fix_dtypes=True` (default) | pandas | pandas | pandas | partial | sampled heuristics on the pandas reference; FreshCore casts bool/numeric natively, defers datetimes |
| `drop_constant_columns` | pandas | pandas | pandas | pandas | needs a data scan before planning (two-phase plan not built) |
| `optimize_memory` | pandas | pandas | pandas | pandas | pandas-specific downcasting — meaningless for other outputs, by design |
| semantic cleaning | native-distinct | native-distinct | pandas | pandas | polars/duckdb run it over a natively extracted distinct table; non-default semantic backends force pandas |
| pandas input with a mixed-type object column (e.g. numbers and strings) | pandas | pandas | — | pandas | native ingestion would reject the column (polars) or cast every value to text (duckdb) |
| pandas input with duplicate column labels | pandas | pandas | — | pandas | native frames need unique column names; the pandas pipeline deduplicates them (`"x", "x"` → `"x", "x_2"`) |
| pandas input whose column labels collide once stringified (e.g. `1` and `"1"`) | native | native | — | pandas | FreshCore names columns by `str(label)`, so one column would overwrite the other; distinct non-string labels (e.g. `0`, `1`) stay native and come back unchanged |
| pandas input with datetime / timedelta / categorical columns | native | native | — | pandas | FreshCore v1 carries only float, bool and string columns, so these dtypes would come back as strings |
| pandas input with period or interval columns | pandas | pandas | — | pandas | DuckDB rejects both dtypes outright; Polars ingests a period as its raw int64 ordinal (`2020-01` → `600`) and an interval as a `{left, right}` struct |
| pandas input with a nanosecond `timedelta64[ns]` or a timezone-aware datetime column | native | pandas | — | pandas | DuckDB stores `INTERVAL` in microseconds (`5ns` → `0`) and returns `TIMESTAMP WITH TIME ZONE` in the session time zone, at microsecond resolution |
| pandas input with an integer column holding a value beyond ±2\*\*53 | native | native | — | pandas | FreshCore v1 carries numbers as float64, which cannot represent such integers exactly; other integer columns are cast back to their input dtype (or reported in `backend_differences` when the result is no longer integral) |
| contracts / validation / memory / profile replay | pandas | pandas | pandas | pandas | in-memory reference features (see [limitations](limitations.md)) |

“pandas” means the **whole pipeline** runs on the pandas reference (fallbacks
are all-or-nothing per run, not per step) and is recorded as a
`fallback_event` with the exact reason.

## Refusing the fallback: `fallback_policy`

You never have to *discover* a fallback after the fact:

```python
import freshdata as fd

# strict out-of-core guarantee: raise BEFORE any pandas materialization
fd.clean("big.parquet", engine="duckdb", fallback_policy="error",
         strategy="conservative", fix_dtypes=False)

# or just be told about it
fd.clean(df, engine="polars", fallback_policy="warn")   # FallbackWarning
```

- `"allow"` (default): fallback runs, recorded on `report.fallback_events`.
- `"warn"`: additionally emits `fd.FallbackWarning`.
- `"error"`: raises `fd.FallbackError` *before* the pandas pipeline runs —
  the message names the exact trigger and the native escape hatch.
- CLI: `freshdata clean data.parquet --engine duckdb --fallback-policy error`.
- Preview without running anything: `fd.plan(df, engine="duckdb").fallback_reason`.

## What the report tells you afterwards

Every engine run stamps:

- `report.requested_backend` — what you asked for (including `"auto"`);
- `report.backend` — what actually executed;
- `report.fallback_events` — each delegation with its reason;
- `report.rows_materialized` — rows pulled into memory for the returned
  result (`None` for native handles: nothing was pulled);
- `report.peak_memory` — process peak RSS in bytes (`None` on Windows);
- `report.materialized` — `False` when you received an un-collected
  `LazyFrame` / un-fetched DuckDB relation
  (`output_format="polars-lazy"/"duckdb"`).
