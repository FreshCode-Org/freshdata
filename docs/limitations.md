---
title: Honest limitations
description: >-
  What each freshdata path actually guarantees — memory behaviour, which steps
  run natively vs. materialize, optional extras, calibration honesty, and the
  edges that are still evolving.
keywords: freshdata limitations, out-of-core, streaming, pandas materialization, calibration, ECE
---

# Honest limitations

freshdata's one rule is that it never claims more than it does. This page is the
single source of truth for what each path guarantees, where it materializes, and
which edges are still sharp. If something here contradicts a headline number
elsewhere, this page wins — please [open an issue](https://github.com/FreshCode-Org/freshdata/issues).

The short version:

- **Correctness first, scale second.** The accuracy-first decision engine is
  pandas. Native backends (Polars/DuckDB/Spark) and streaming scale the *safe*
  steps out-of-core and transparently fall back to pandas — always recorded —
  when a step needs the decision engine.
- **Nothing is guessed silently.** Ambiguous repairs become suggestions or
  flags, never quiet edits. `strict=True` turns ambiguity into a hard error.
- **No LLM, no cloud, no per-cell inference.** The only network call in the
  package is the explicit `fd.models.pull(...)`; cleaning never touches the wire.

---

## Memory: what materializes and what doesn't

| Path | Behaviour |
|------|-----------|
| `fd.clean(df)` (default pandas) | In-memory; the whole frame is in RAM. |
| `engine="polars"` / `"duckdb"` / `"spark"`, `output_format="pandas"` (default) | Scales out **during** the pipeline (spill-to-disk), then **materializes** the result into a pandas frame. |
| `output_format="duckdb"` | Returns an un-fetched `DuckDBPyRelation` — **not materialized**; you call `.fetchdf()`. |
| `output_format="polars-lazy"` | Returns an uncollected `LazyFrame` — the **result** is not materialized (you call `.collect()`), but pipeline stages currently collect intermediates eagerly, so peak memory *during* cleaning is comparable to eager output. DuckDB is the lower-peak-memory native path today (measured: `python benchmarks/bench_outofcore.py`). |
| `StreamingCleaner` / `fd.clean_timeseries(..., stream=...)` | Genuinely out-of-core: bounded micro-batches, running statistics, never concatenated. |

`report.materialized` is `False` whenever a native handle is returned, and
`report.summary()` says so. If a strategy needs the pandas decision engine
(`balanced` / `aggressive` imputation, dtype heuristics), the native backends
transparently fall back to pandas — recorded in `report.fallback_events` — and
the result is materialized. To keep the native handle use
`strategy="conservative"` **and** `fix_dtypes=False` — dtype fixing uses
sampled pandas heuristics and forces the fallback even under `conservative`
(measure it yourself: `python benchmarks/bench_outofcore.py`).
Streaming holds memory flat by design: bounded reservoirs and counters,
a recent-window (not global) dedup, and no cross-batch concatenation.

---

## Which steps run natively vs. on pandas

- **Streaming-safe** (Polars/DuckDB native): column rename, whitespace/sentinel
  normalization, empty column/row drops, full-row dedup.
- Under streaming native dedup, exact full-row dedup does **not** preserve
  original row order (disclosed in `report.backend_differences`). Set
  `EngineConfig(streaming_dedup=False)` to preserve order — which materializes.
- The accuracy-first **decision engine**, heuristic dtype repair, and opt-in
  impute/outliers run on pandas (materialized).
- **pandas-only features:** `contract=` gates, `fd.validate(suite=...)`
  validation suites (non-pandas inputs are materialized — recorded on
  `ValidationResult.execution`), `memory=` replay,
  `compare_to_baseline(key=...)` key-level diffs, `fd.lint_text_encoding`,
  `fd.evaluate_quality_debt`, and the compliance-report generators all operate
  on in-memory pandas frames.
- Native-engine users can make an unrequested pandas materialization
  impossible with `fallback_policy="error"` — see the
  [fallback matrix](fallback-matrix.md).

### Native-engine semantic cleaning

The semantic stage reasons only about a column's **distinct values**, so on a
Polars/DuckDB engine it runs over a *natively extracted* distinct table (a
`GROUP BY`, bounded by `semantic_max_distinct_values`) rather than pulling the
whole frame into pandas. The distinct table is scored through the same gate as
the pandas reference path, and accepted repairs are mapped back natively with
`replace`/SQL `CASE`. This keeps the scale path out-of-core and honest — with
edges:

- **Representation, not correctness.** Native engines have no `object` dtype. A
  *partially*-mapped boolean column that pandas returns as a mixed `[True, False,
  "unknown"]` object column is kept as a canonical string column
  (`["true", "false", "unknown"]`) on native engines. The **same cells are
  repaired**; only the storage dtype differs. Fully-coercible columns (all values
  mapped) tighten to `Boolean`/`Int64`/`Float64` and match pandas exactly.
- **Lazy frames + non-string targets.** On an un-collected `LazyFrame`, a repair
  that would target a non-string column is left unchanged and disclosed in
  `report.fallback_events` (experts target string columns, so this is rare).
- **Non-default backends.** The native distinct path serves the default
  deterministic backend. When `semantic_backends` also includes `memory` /
  `profile` / `embedding`, or a learned `profile=` is supplied, the whole clean
  routes through pandas with a recorded semantic fallback so results stay
  byte-identical to the reference path.

---

## Context policies and protected columns

The context compiler, semantic experts (email / phone / reference lists /
numbers / dates), repair plans, and the protected-column guard are all
**deterministic, offline** code paths — no LLM, no ONNX runtime, no learned
component, no network.

- **Only the tier-0 context language.** `context=` understands the documented
  sentence patterns (uniqueness, protection, formats, allowed values, imputation
  confidence, ranges, dedup keys). Arbitrary natural language is surfaced as
  *unparsed*, never guessed at; `strict=True` makes that a hard error.
- **Ambiguous repairs are never auto-applied.** `bob[at]gmail.com`, a phone
  number with the wrong digit count, a typo close to two allowed values, an
  ambiguous `01/02/2026` date — these become suggestions or flags in the
  report/plan, not silent changes.
- **Phone normalization ships for `region="IN"` only** today; other regions
  compile into the policy but produce no value repairs yet.
- **Hard byte-identity for context-protected columns** (a `protected` rule or
  `mutable=False`). Legacy `preserve_columns` keeps its historical meaning —
  never dropped, but representation repair (whitespace, dtypes) still applies —
  unless the column is also context-protected. `fd.apply_plan` additionally
  guards `preserve_columns`, `target_column`, and id columns.
- **Streaming compiles the policy once.** A stream has no single frame, so
  `StreamingCleaner(context=...)` compiles the policy against the **first batch**
  and applies the same protected columns and constraints to every batch — it is
  never recompiled, so a batch missing a column can't make the policy drift.
  Protected columns are excluded from the streaming imputer too, so their missing
  cells are preserved rather than filled. Inspect `cleaner.policy_`.
- **Undo is cell-scoped.** Row drops, aggregations, and column drops are not
  reversible from the undo log and are never marked as such.

---

## Models and calibration honesty

- **The default install is model-free.** Everything in this section applies only
  after `pip install "freshdata-cleaner[semantic]"` *and* an explicit
  `fd.models.pull(...)` (or air-gapped file placement). Nothing is ever
  downloaded during cleaning.
- **Official model artifacts are not hosted yet.** `fd.models.pull` raises a
  clear `ModelNotPublishedError` until they are; the air-gapped path and the
  `FRESHDATA_MODEL_URL_BASE` mirror override work today. Checksums are pinned as
  artifacts publish; unpinned manual placements load as *unverified*.
- **Embedding proposals are evidence, not authority.** They pass the same gate as
  deterministic proposals, are calibrated conservatively (pure similarity
  clustering is capped below the auto threshold — suggest-only by default), and
  ambiguous matches produce no proposal at all.
- **Calibration is honest, not magical.** Out of the box, freshdata ships a
  deterministic isotonic table that is identity for deterministic/memory
  proposals and conservative for embedding ones. On CleanBench this reaches an
  **expected calibration error (ECE) of ~0.038**, which clears the ≤ 0.05 target
  but **not** the stricter ≤ 0.03 tier. The stricter tier requires the trained
  [`calib-v1`](model-cards.md) artifact, which is **not published yet** — so the
  strict-ECE gate is a known, disclosed gap, not a silent failure (see
  [Benchmarks](benchmarks.md)). The ">95% confidence" imputation clause remains
  mostly a polite refusal: honestly calibrated confidence rarely clears 0.95
  outside near-deterministic cases, and freshdata preserves rather than guesses.
- **No LLM, no cloud, no per-cell inference** — structurally: backends see
  distinct values only.

---

## Benchmarks: what the numbers mean

CleanBench headline results are committed to the repo and reproducible with
`python -m benchmarks.cleanbench ... --reproduce-headline` /
`--verify-results` (see [Benchmarks](benchmarks.md)). Read them with these
caveats:

- **Release evals run on synthetic and template-derived corpora** reviewed by
  maintainers, plus a set of curated real public datasets. Strong scores there
  are necessary, not sufficient, evidence for arbitrary real-world data.
- **The headline run uses the deterministic calibration**, not `calib-v1`
  (`trained calib-v1 used: no`). The strict-ECE row is expected to fail until
  that artifact ships; every other headline gate passes.
- **Performance gates compare against a same-machine pinned baseline.** On a
  fresh machine the first run bootstraps the baseline and the perf gate is
  informational until then.

---

## Requires an optional extra

| Feature | Extra |
|---------|-------|
| Polars / DuckDB / Spark backends | `polars` / `duckdb` / `spark` |
| Interactive HTML upgrades (tables, charts) | `viz` / `notebook` |
| NER-based PII + format-preserving encryption | `privacy` |
| dbt / Great Expectations / orchestration exporters | `integrations` (or `dbt`, `dagster`, `airflow`) |
| YAML domain packs & CLI | `domains` / `cli` |
| Optional learned models | `semantic` (+ `ml` for the tensor-framework build) |

The base renderers, regex PII detection, entity resolution (pandas fallback),
context policies, and cleaning memory need **none** of these.

---

## Experimental / evolving

- `output_format="duckdb"` / `"polars-lazy"` native handles are new; the exact
  handle type follows the installed DuckDB/Polars version.
- Quality-debt escalation heuristics and the dirty-join confidence scoring are
  tuned conservatively and may change between minor versions.
- `fd.lint_text_encoding` is heuristic; treat "auto-repair-safe" as advisory and
  review before bulk-applying.
- The Spark backend requires a JVM and `pyspark`; it is exercised in native
  parity tests but is not part of the default CI matrix.

---

## Development-time training pipeline

These apply only to the dev-time `training/` package, never to `import freshdata`:

- The optional encoder contrastive-distillation stage requires a tensor
  framework; without it the stage records `skipped` and the Phase-3 baseline
  encoder is retained (its safety gates still run).
- Dev artifact builds without the `onnx` package export portable weight JSON
  instead of `.onnx` graphs; release builds require `onnx` and fail without it.
