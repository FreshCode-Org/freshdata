---
title: Honest limitations
description: >-
  Which freshdata operations are pandas-only, streaming-safe, materialize, need
  an optional extra, or are experimental.
keywords: freshdata limitations, out-of-core, streaming, pandas materialization
---

# Honest limitations

freshdata aims to never claim more than it does. This page is the single source
of truth for what each path actually guarantees.

## Materialization vs out-of-core

| Path | Behaviour |
|------|-----------|
| `fd.clean(df)` (default pandas) | In-memory; the whole frame is in RAM. |
| `engine="polars"` / `"duckdb"` / `"spark"`, `output_format="pandas"` (default) | Scales out **during** the pipeline (spill-to-disk), then **materializes** the result into a pandas frame. |
| `output_format="duckdb"` | Returns an un-fetched `DuckDBPyRelation` — **not materialized**; you call `.fetchdf()`. |
| `output_format="polars-lazy"` | Returns an uncollected `LazyFrame` — **not materialized**; you call `.collect()`. |
| `StreamingCleaner` / `fd.clean_timeseries(..., stream=...)` | Genuinely out-of-core: bounded micro-batches, never concatenated. |

`report.materialized` is `False` whenever a native handle is returned, and
`report.summary()` says so. If a strategy needs the pandas decision engine
(`balanced` / `aggressive` imputation, dtype heuristics), the native backends
transparently fall back to pandas — recorded in `report.fallback_events` — and
the result is materialized. Use `strategy="conservative"` to keep the native
handle.

## Streaming-safe vs materializing steps

- **Streaming-safe** (Polars/DuckDB native): column rename, whitespace/sentinel
  normalization, empty column/row drops, full-row dedup.
- Under streaming, exact full-row dedup does **not** preserve original row order
  (disclosed in `report.backend_differences`). Set
  `EngineConfig(streaming_dedup=False)` to preserve order — which materializes.
- The accuracy-first **decision engine**, heuristic dtype repair, and opt-in
  impute/outliers run on pandas (materialized).

## pandas-only operations

`contract=` gates, `memory=` replay, `compare_to_baseline(key=...)` key-level
diffs, `fd.lint_text_encoding`, `fd.evaluate_quality_debt`, and the compliance
report generators operate on in-memory pandas frames.

## Requires an optional extra

| Feature | Extra |
|---------|-------|
| Polars / DuckDB / Spark backends | `polars` / `duckdb` / `spark` |
| Interactive HTML upgrades (tables, charts) | `viz` / `notebook` |
| NER-based PII + format-preserving encryption | `privacy` |
| dbt / Great Expectations / orchestration exporters | `integrations` (or `dbt`, `dagster`, `airflow`) |
| YAML domain packs & CLI | `domains` / `cli` |

The base renderers, regex PII detection, entity resolution (pandas fallback),
and cleaning memory need **none** of these.

## Experimental / evolving

- `output_format="duckdb"` / `"polars-lazy"` native handles are new; the exact
  handle type follows the installed DuckDB/Polars version.
- Quality-debt escalation heuristics and the dirty-join confidence scoring are
  tuned conservatively and may change between minor versions.
- `fd.lint_text_encoding` is heuristic; treat "auto-repair-safe" as advisory and
  review before bulk-applying.
