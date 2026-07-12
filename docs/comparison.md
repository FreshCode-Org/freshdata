# Honest comparison: pandas, PyJanitor, Great Expectations, OpenRefine

Every cell is one of **win / loss / tie / unproven**, and every *win* links to
committed evidence in this repository. "Unproven" means we believe something
but have not measured it — we say so instead of claiming it. Losses are
stated plainly. Cleaning (mutation) and validation (read-only checks) are
scored separately; exact deduplication and entity resolution are scored
separately.

Evidence sources referenced below:

- `src/freshdata/benchmarks/RESULTS.md` — out-of-core benchmark (10k → 25M
  rows; wall, peak RSS, throughput, trust-score parity; machine + versions +
  exact command recorded).
- `benchmarks/er_results.json` + `benchmarks/data/er_labelled_5k.csv` —
  labelled entity-resolution accuracy (P/R/F1, FP/FN, false merges,
  blocking reduction), generator committed (`benchmarks/gen_er_labelled.py`).
- `benchmarks/cleanbench/results/` — CleanBench cell-repair quality vs
  pandas / PyJanitor / Great Expectations baselines (`benchmarks/baselines/`).
- `tests/` — feature claims cite the test that exercises them.

## vs pandas

| Criterion | Verdict | Evidence / why |
|---|---|---|
| Ubiquity, ecosystem, general dataframe ops | **loss** | pandas is the substrate; freshdata is built on it |
| One-call structural cleaning with audit trail | **win** | pandas has no cleaning pipeline or action log; `tests/test_api.py`, CleanBench T1 vs the pandas baseline (`benchmarks/baselines/pandas_baseline.py`, results in `benchmarks/cleanbench/results/`) |
| Cleaning throughput at 10M+ rows | **win** | Polars backend 2–3× pandas wall time at 10M rows with identical output (trust-score parity) — `src/freshdata/benchmarks/RESULTS.md` |
| Peak memory at moderate scale (1M rows, materializing) | **win** | DuckDB 200 MB vs pandas 1,046 MB — `RESULTS.md` |
| Peak memory at 25M rows, materializing output | **loss** | measured honestly: polars/duckdb peak RSS *exceeds* pandas at 25M on a 16 GB machine (Trust-Score conversion + memory pressure — see the 25M caveat in `RESULTS.md`); the native-handle path (`output_format="duckdb"`) avoids result materialization (`benchmarks/bench_outofcore.py`) but is not what this row measures |
| Raw single-primitive speed (one `fillna`, one `drop_duplicates`) | **loss** | freshdata adds profiling/reporting around each step; a single pandas primitive is thinner. We do not benchmark our pipeline against a lone primitive — that comparison would be rigged in our favour on features and against us on time |
| Missing-value recall/precision beyond `fillna` | **win** | role-aware decision engine measured on CleanBench T2/T3 vs the pandas baseline (`benchmarks/cleanbench/results/latest.md`) |
| Exact deduplication semantics | **tie** | same semantics (freshdata delegates to/reproduces pandas; parity tests `tests/test_execution/test_fallback_policy.py`) |

## vs PyJanitor

| Criterion | Verdict | Evidence / why |
|---|---|---|
| Breadth of convenience verbs (finance, chemistry, engineering accessors) | **loss** | PyJanitor ships many domain accessors freshdata has no equivalent for |
| Fluent chaining API | **tie** | `fd.pipeline()` (immutable, serializable, engine-aware — `tests/test_pipeline_builder.py`); PyJanitor chains directly on DataFrames, which some prefer |
| Cleaning quality on messy data | **win** | CleanBench T1–T4 vs the committed PyJanitor baseline (`benchmarks/baselines/`, results in `benchmarks/cleanbench/results/latest.md`) |
| Audit trail / reports / risk levels | **win** | PyJanitor mutates without an action log; every freshdata action carries rationale/risk/confidence (`tests/test_api.py`) |
| Execution beyond pandas (Polars/DuckDB/Spark, out-of-core, 25M rows) | **win** | PyJanitor is pandas-only; `RESULTS.md` (10k–25M), `docs/fallback-matrix.md` for exactly what is native |
| Simplicity of a quick one-liner on a small frame | **tie** | `df.clean_names()` vs `fd.clean(df)` — both one line |

## vs Great Expectations

| Criterion | Verdict | Evidence / why |
|---|---|---|
| Expectation vocabulary breadth | **loss** | GX ships hundreds of expectations; freshdata's suite vocabulary is the common core (~20 rule types — `docs/validation.md`) |
| Suite/checkpoint stores, Data Docs, orchestration ecosystem | **loss** | GX has a mature deployment ecosystem; freshdata suites are plain JSON + a CLI |
| Native validation execution on Spark/SQL engines | **loss** | GX executes expectations on Spark/SQLAlchemy backends; freshdata validation materializes to pandas (disclosed on `ValidationResult.execution`) |
| Setup overhead to first validation | **win** | zero scaffolding: one import, one `ValidationSuite`, exit-code CLI (`tests/test_validation_suite.py`); GX requires context/datasource configuration |
| Failure tolerance (`mostly`), severity levels, unexpected-value samples | **tie** | both support them — `tests/test_validation_suite.py::test_mostly_*` |
| Integrated cleaning + repair after validation | **win** | GX validates only; freshdata validates *and* repairs with plan/approve/apply and protected-column guarantees (`tests/test_repairplan*.py`) |
| Baseline drift monitoring (PSI/KS) in core | **win** | `fd.compare_to_baseline` / `monitor_contract` with PSI/KS thresholds (`tests/test_enterprise_contracts.py`); GX core focuses on expectations, not statistical drift baselines |
| Interop | **tie** | freshdata exports GX suites (`fd.export_gx_suite`, `tests/test_expectations_preflight.py`) — cooperation, not replacement |

## vs OpenRefine

| Criterion | Verdict | Evidence / why |
|---|---|---|
| Interactive, human-in-the-loop curation UI | **loss** | OpenRefine is a GUI built for exactly that; freshdata is a library |
| Reconciliation against external services (Wikidata etc.) | **loss** | not a freshdata capability |
| Reproducible, headless, CI-runnable pipelines | **win** | freshdata runs are code + serializable plans/suites/pipelines with deterministic outputs (`tests/test_pipeline_builder.py`, reproducibility workflow); OpenRefine's operation history replays inside OpenRefine, not headless CI |
| Entity resolution with blocking at scale | **win** | blocking rules + `max_pairs` safety gate + review queue + golden records (`tests/test_enterprise_entity_resolution*.py`); OpenRefine clusters one column in memory |
| Measured linkage accuracy | **win** (self), **unproven** (head-to-head) | committed labelled P/R/F1/false-merge results (`benchmarks/er_results.json`); a head-to-head vs OpenRefine clustering is not automated here because OpenRefine's clustering runs interactively — we do not claim to beat it |
| Fuzzy clustering ergonomics for one messy column | **tie** | `cluster_column` covers key-collision-style clustering; OpenRefine's interactive refine loop is better for exploration |
| Scale beyond RAM | **win** | DuckDB/Polars streaming with committed RSS evidence up to 25M rows (`RESULTS.md`); OpenRefine loads the project into memory |

## What FreshData still cannot claim

- Splink-class **probabilistic** record linkage: scoring is a rule-weighted
  normalized average — no m/u-probabilities, no EM. Named accordingly
  everywhere; see the ER module docstring.
- GX-scale validation vocabulary or a validation deployment ecosystem.
- Native (non-pandas) validation execution.
- 100M+/1B-row committed benchmark evidence: the harness has the size keys,
  but the largest committed run is 25M rows (see `RESULTS.md` for machine
  limits).
- Any superiority on a workload we haven't benchmarked — the tables above
  link every "win" to its evidence; everything else is "unproven".

## Verdict (computed from the tables)

Wins cluster on: **audited automated cleaning, scale (out-of-core execution
with disclosed fallbacks), integrated validate→plan→repair, reproducible
pipelines, and labelled ER evidence**. Losses cluster on: **ecosystem size
(pandas, GX), vocabulary breadth (GX, PyJanitor verbs), and interactive
curation (OpenRefine)**.

So: choose FreshData when you want an automated, audited, reproducible
cleaning+validation pipeline that scales past pandas. Choose GX for a
validation-platform deployment, OpenRefine for hands-on curation, PyJanitor
for its convenience-verb breadth on small frames — and pandas is under all
of it either way. FreshData "leads" only in the first column, and this page
is the boundary of that claim.
