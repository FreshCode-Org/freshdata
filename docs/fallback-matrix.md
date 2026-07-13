# Backend fallback matrix

**The single most important row is the first one.** The default
`strategy="balanced"` runs the accuracy-first decision engine, which is
evaluated by the pandas backend — so with default options, **every native
engine delegates the whole pipeline to pandas** (recorded on
`report.fallback_events`). The fully native path is
`strategy="conservative"` with `fix_dtypes=False`.

This table is transcribed from the single source of truth,
`PlanGenerator.fallback_reason()` in `src/freshdata/execution/_plan.py` —
if you change that function, change this page.

| Operation / config | polars | duckdb | spark | freshcore | Why the fallback exists |
|---|---|---|---|---|---|
| `strategy="balanced"` / `"aggressive"` (default) | pandas | pandas | pandas | pandas | data-dependent decision engine; porting it per backend would fork its accuracy behaviour |
| column rename / whitespace / sentinels | native | native | native | native | — |
| empty row/column removal | native | native | native | native | — |
| full-row dedup (`keep="first"/"last"`) | native | native | native | native | streaming polars dedup drops row order (disclosed); `streaming_dedup=False` restores it |
| **subset dedup** (`duplicate_subset=`) | **native** | pandas | pandas | pandas | keep semantics are order-sensitive; Polars reproduces them via order-preserving `unique` (eager, not streaming — disclosed) |
| dedup `keep="drop"/"aggregate"` | pandas | pandas | pandas | pandas | group-wise resolution isn't expressed natively yet |
| global impute mean/median/mode | native | native | native | native | — |
| per-column `impute_strategy` | pandas | pandas | pandas | pandas | unimplemented natively (no fundamental blocker) |
| `impute="missforest"` | pandas | pandas | pandas | pandas | scikit-learn model |
| outliers `iqr` / `zscore` | native | native | native | native | — |
| `outlier_action="auto"` / model methods | pandas | pandas | pandas | pandas | data-dependent / model-based selection |
| `fix_dtypes=True` (default) | pandas | pandas | pandas | partial | sampled heuristics on the pandas reference; FreshCore casts bool/numeric natively, defers datetimes |
| `drop_constant_columns` | pandas | pandas | pandas | pandas | needs a data scan before planning (two-phase plan not built) |
| `optimize_memory` | pandas | pandas | pandas | pandas | pandas-specific downcasting — meaningless for other outputs, by design |
| semantic cleaning | native-distinct | native-distinct | pandas | pandas | polars/duckdb run it over a natively extracted distinct table; non-default semantic backends force pandas |
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
