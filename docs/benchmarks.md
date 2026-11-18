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

## Performance and scalability investigation

The [performance investigation](performance-investigation.md) documents the
deterministic mixed-schema harness, isolated subprocess execution, versioned JSON
evidence, profiling, and variability-aware comparison methodology. Baseline
measurement is in progress, so the investigation does not yet claim a slowdown or
improvement.

```bash
pip install -e ".[dev,bench,ml]"
make performance-ci
make performance-baseline
make performance-profile
make performance-report
```

Raw cases are written under `benchmarks/results/performance/` and remain ignored.
Compact `*-summary.json` and `*-report.md` evidence may be committed after it is
validated. The weekly/manual GitHub Actions workflow runs the large matrix and
uploads the full result directory, including failed, timed-out, or exhausted cases.

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

## Entity-resolution accuracy (labelled)

Measured on the committed labelled dataset
`benchmarks/data/er_labelled_5k.csv` (5,000 rows; generator:
`benchmarks/gen_er_labelled.py --rows 5000 --seed 7` — typos, reordered names,
abbreviations, blanked fields, transliteration, shared households, deliberate
near-collisions). Method: **rule-weighted linkage** (normalized weighted
average — no EM/Fellegi–Sunter, no Splink-parity claim). Raw output:
`benchmarks/er_results.json`.

| Configuration | Precision | Recall | F1 | FP | FN | False-merged clusters |
|---|---:|---:|---:|---:|---:|---:|
| pandas exact `drop_duplicates` (baseline) | 1.000 | 0.054 | 0.103 | 0 | 2,539 | 0 |
| `null_policy=penalize, mode=balanced` | 0.999 | 0.300 | 0.462 | 1 | 1,878 | 1 |
| `null_policy=neutral, mode=balanced` | 0.994 | 0.456 | 0.626 | 8 | 1,459 | 4 |
| `null_policy=neutral, mode=precision` | 0.991 | 0.332 | 0.497 | 8 | 1,793 | 4 |
| `null_policy=neutral, mode=recall` | 0.978 | 0.780 | 0.868 | 47 | 591 | 27 |

Blocking pruned 99.96% of the 12.5M candidate pairs in every configuration.
Honest readings:

- exact dedup recovers **5.4%** of true duplicates on this data — that gap is
  what fuzzy linkage exists to close;
- `null_policy="neutral"` is the single biggest accuracy lever here (recall
  0.30 → 0.46 at ~equal precision): blanked fields stop counting as
  disagreement;
- on near-collision-heavy data, `mode=precision` did **not** beat balanced
  precision — the surviving false positives are near-identical distinct
  people who score above any reasonable threshold. Review queues, not
  thresholds, are the tool for that failure mode;
- recall is bounded by blocking: pairs no blocking rule reaches are never
  scored. Add rules to raise the ceiling, at candidate-pair cost;
- no head-to-head vs OpenRefine/recordlinkage/Splink is published here — see
  [comparison](comparison.md) for what is and isn't claimed.

## Competitor differentiation

See [`benchmarks/competitor_analysis.md`](https://github.com/FreshCode-Org/freshdata/blob/main/benchmarks/competitor_analysis.md):
a curated, factual table covering pandas, pyjanitor, Great Expectations, Soda,
dbt, AWS Glue Data Quality, Google Dataplex, OpenRefine, Dedupe,
ydata-profiling, sweetviz and cleanlab. Capability claims come from each tool's
public docs; FreshData advantages appear only where a harness metric confirms
them.

