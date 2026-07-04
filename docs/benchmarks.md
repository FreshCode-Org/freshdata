---
title: Benchmarks
description: >-
  freshdata performance benchmarks — throughput on real and synthetic datasets,
  the reproducible nine-metric harness, and how to run them locally.
keywords: data cleaning performance, pandas cleaning speed, freshdata benchmarks
---

# Benchmarks

`freshdata` is built on vectorized pandas/NumPy with one-pass engine caching
(correlation matrix, column contexts). No C extension is required.

## Typical throughput

Measured on a modern laptop (see `tests/fixtures/perf/baselines.json`):

| Dataset size | Balanced | Aggressive |
|---|---|---|
| 500 rows | < 0.5 s | < 1 s |
| 3,000 rows | < 2.5 s | < 6 s |
| 29k rows (full AQI) | < 5 s | KNN gated |

The aggressive bottleneck is KNN imputation on large frames, which is why KNN is
gated to aggressive mode only.

MissForest-style imputation is also benchmarked separately because it is
opt-in, scikit-learn-backed, and intentionally slower than the default engine.
Use `python benchmarks/bench_missforest.py` after installing
`freshdata-cleaner[ml]` to compare median/mode, aggressive KNN, and MissForest
on mixed-type synthetic data.

## The Benchmark Release harness

`benchmarks/bench.py` is a reproducible, schema-stable harness that measures
**nine standardized metrics** against an enterprise fixture library, with pandas
and pyjanitor baselines. It calls FreshData exactly as a user would; it never
modifies library internals.

```bash
pip install -e ".[dev]" jsonschema

python benchmarks/bench.py fixtures     # (optional) write fixture CSVs to disk
python benchmarks/bench.py run          # all fixtures, 10k rows, write results/<run_id>/
python benchmarks/bench.py report       # render report.md + report.json for the latest run
python benchmarks/bench.py compare --fixture crm --size 100000   # FreshData vs baselines
python benchmarks/bench.py single --fixture crm --size 100000 --metric time
```

Makefile shortcuts: `make benchmark`, `make benchmark-ci`, `make benchmark-report`,
`make benchmark-fixtures`, `make benchmark-test`. Results land in
`results/<run_id>/<fixture>/<size>.json` in the schema in
`benchmarks/results_schema.py`, so runs diff cleanly across versions.

> The legacy quick-bench (`python benchmarks/bench_quick.py --fixtures --compare`)
> is preserved for ad-hoc throughput checks on the `tests/fixtures/` corpus.

## Calibration metrics (CleanBench Phase 3)

The CleanBench mini-suite (`benchmarks/cleanbench/`, run in CI by
`tests/test_cleanbench.py`) gates the semantic layer's *confidence honesty*
alongside the Phase-2 safety gates:

| metric | definition | Phase-3 gate | long-run target |
|---|---|---|---|
| protected-column violation rate | any diff in protected columns | **= 0, absolute** | = 0 |
| false modification rate | already-correct cells changed anyway | ≤ 0.1% | trend to 0 |
| expected calibration error (ECE) | equal-width-bin gap between confidence and accuracy over semantic proposals | ≤ 0.05 | ≤ 0.03 |
| precision @ confidence ≥ 0.95 | share of high-confidence proposals that match ground truth | ≥ 0.98 | ≥ 0.99 |
| coverage @ precision 0.98 | largest share of proposals acceptable at that precision (abstention quality) | reported | grows per phase |
| ambiguous auto-applies | embedding merges applied without margin/allowed-values evidence | **= 0** | = 0 |

Pairs are extracted with `cleanbench.confidence_outcomes(report, truth,
corrupted)` — every semantic proposal's calibrated confidence against whether
its repair matches the fixture's ground truth. The Phase-3 fixture runs the
full embedding path with a deterministic stub encoder, so the gates hold with
no model files and no network.

## Strategic-report scaling benchmarks

`benchmarks/bench_report.py` covers five reproducible scaling cases, each
measured for `strategy="balanced"` vs `strategy="aggressive"` where applicable.
It generates its own synthetic fixtures and writes
`benchmarks/results/report_bench.json`.

```bash
pip install -e ".[bench]"          # pyarrow + psutil

python benchmarks/bench_report.py --all                 # everything
python benchmarks/bench_report.py csv_ingest --mb 100   # ~100 MB CSV ingest + clean
python benchmarks/bench_report.py profile --rows 1000000     # 1M-row mixed-schema profile
python benchmarks/bench_report.py nullfill --rows 10000000   # 10M-row null-fill / flag pass
python benchmarks/bench_report.py import_time           # cold `import freshdata`
python benchmarks/bench_report.py memory --rows 1000000      # peak-RSS of a full clean
```

### Results — not yet measured

These numbers are **environment-specific and are not committed**. Run the
commands above and read `benchmarks/results/report_bench.json`. Fill the table
in with the hardware/software you actually ran on; do not copy numbers from
elsewhere.

- **Hardware:** _(CPU, cores, RAM — fill in)_
- **Software:** _(OS, Python version, pandas/pyarrow/polars/duckdb versions — fill in)_

| Benchmark | Balanced | Aggressive |
|---|---|---|
| 100 MB CSV ingest + clean | _not yet measured_ | _not yet measured_ |
| 1M-row profile | _not yet measured_ | _n/a (read-only)_ |
| 10M-row null-fill / flag | _not yet measured_ | _not yet measured_ |
| Import time (`import freshdata`) | _not yet measured_ | _n/a_ |
| Peak memory @ 1M rows | _not yet measured_ | _not yet measured_ |

### What each metric measures (and why)

| # | Metric | Definition | Why it matters |
|---|---|---|---|
| 1 | **Wall-clock** | p50/p95 seconds for `fd.clean(df, return_report=True)` over 5 repeats, I/O excluded. | Raw throughput on real-shaped data. |
| 2 | **Peak memory (MB)** | Peak/delta `tracemalloc` allocation during the run. | Footprint, esp. the wide-schema report-generation stress case. |
| 3 | **Repair fidelity (%)** | Cell-level match to the gold oracle; family-level post-conditions (`DEFECT_MANIFEST`) for named fixtures. | Does FreshData actually fix what it claims to? |
| 4 | **False-repair rate (%)** | % of cells that must **not** change (id/target/free-text traps) that changed anyway. | The core safety contract. Must be **0**. |
| 5 | **Preservation rate (%)** | % of protected values identical before/after. | The id/target/text invariant under load. Must be **100** on non-null ids. |
| 6 | **Authored-code reduction** | FreshData call lines vs the pandas/pyjanitor baseline lines for the same defect set. | Expressiveness (reported separately from timing). |
| 7 | **Diagnosis speed** | Latency of `report.summary()` / `.to_frame()` / `.to_dict()` (+ enterprise `quality.to_markdown()` / `lineage.emit()`). | Time from report to explanation. |
| 8 | **Trust-score usefulness** | Strict monotonic decrease of the 0–100 trust score as defect rate rises (0→60%). | The trust score must be a meaningful signal. |
| 9 | **Export completeness** | All report fields populated; every repaired action carries before/after, risk, confidence (rationale for engine decisions). | A repair you cannot explain or export is not auditable. |

### Current results — 10k-row scale, balanced mode

Refreshed by the CI benchmark workflow; values below are a reference local run
(freshdata 1.0.0). Re-render with `bench.py run && bench.py report`.

| fixture | n_rows | n_cols | p50 s | p95 s | peak MB | repair % | false-repair % | preserve % | trust | monotonic | export % |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|--:|
| crm | 10,200 | 40 | 0.93 | 0.93 | 8.2 | 100.0 | 0.0 | 100.0 | 93.8 | ✅ | 100.0 |
| finance | 10,200 | 60 | 1.16 | 1.16 | 10.0 | 100.0 | 0.0 | 100.0 | 99.5 | ✅ | 100.0 |
| event_log | 10,000 | 25 | 0.35 | 0.35 | 6.4 | 100.0 | 0.0 | 100.0 | 99.7 | ✅ | 100.0 |
| wide_schema | 10,000 | 100 | 2.28 | 2.29 | 14.6 | 100.0 | 0.0 | 100.0 | 96.0 | ✅ | 100.0 |
| provenance | 10,000 | 18 | 0.29 | 0.30 | 7.5 | 100.0 | 0.0 | 100.0 | 99.8 | ✅ | 100.0 |
| gold | 10,200 | 7 | 0.14 | 0.14 | 2.7 | 100.0 | 0.0 | 100.0 | 98.3 | ✅ | 100.0 |

**Authored-code reduction (Metric 6):** FreshData = 3 lines; pandas baseline = 26
lines (**88.5%** reduction); pyjanitor baseline = 20 lines (**85.0%** reduction).

The false-repair rate is **0.0%** and preservation **100.0%** on every fixture —
the id/target/free-text invariant holds under load, not just on toy examples.

## Competitor differentiation

See [`benchmarks/competitor_analysis.md`](https://github.com/FreshCode-Org/freshdata/blob/main/benchmarks/competitor_analysis.md):
a curated, factual table covering pandas, pyjanitor, Great Expectations, Soda,
dbt, AWS Glue Data Quality, Google Dataplex, OpenRefine, Dedupe,
ydata-profiling, sweetviz and cleanlab. Capability claims come from each tool's
public docs; FreshData advantages appear only where a harness metric confirms
them.

## Adding a fixture or baseline

See [`docs/fixtures.md`](fixtures.md) for fixture schemas and the
`DEFECT_MANIFEST` structure, and `benchmarks/README.md` for the baseline
contract (`run(df)` + `AUTHORED_LINES`, lazy imports, graceful skip).

## Reproduce the legacy corpus

```bash
python benchmarks/bench_quick.py --fixtures --compare   # tests/fixtures corpus, side by side
python benchmarks/bench_quick.py --online --compare     # cached online datasets
```

Every fixture in `tests/fixtures/` is run under `conservative`, `balanced`, and
`aggressive` strategies in CI, plus 50 curated real public datasets. Reproduce
the quality/efficiency matrix on your own data with:

```python
import freshdata as fd
print(fd.compare_clean(df))
```
