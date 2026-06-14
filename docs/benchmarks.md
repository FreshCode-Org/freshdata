---
title: Benchmarks
description: >-
  freshdata performance benchmarks — throughput on real and synthetic datasets,
  and how to reproduce them locally.
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

## Reproduce locally

```bash
python benchmarks/bench.py --fixtures --compare   # all local fixtures, side by side
python benchmarks/bench.py --online --compare     # cached online datasets
python benchmarks/bench.py                         # synthetic data
```

Optional large-file benchmark (29k-row AQI.csv, not committed):

```bash
export FRESHDATA_AQI_PATH=/path/to/AQI.csv
pytest -m large
```

## Validated scenarios

Every fixture in `tests/fixtures/` is run under `conservative`, `balanced`, and
`aggressive` strategies in CI, plus 50 curated real public datasets (10 Tier-1
anchors with golden snapshots, 40 Tier-2 smoke tests). Reproduce the
quality/efficiency matrix on your own data with:

```python
import freshdata as fd
print(fd.compare_clean(df))
```
