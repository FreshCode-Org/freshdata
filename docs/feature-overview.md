---
title: Feature overview
description: >-
  A complete overview of freshdata's features — automated cleaning, profiling,
  explainable reports, the enterprise governance layer, and Polars support.
keywords: freshdata features, data cleaning features, pii masking, data trust score, openlineage, polars data cleaning
---

# Feature overview

## Core

| Feature | Description |
|---|---|
| Automated cleaning | `fd.clean(df)` handles missing values, outliers, duplicates, dtype repair, and column names in one call. |
| Decision engine | Per-column actions chosen from inferred role + explicit threshold rules. |
| Explainable reports | Every action carries a rationale, risk level, and confidence score. |
| Profiling | `fd.profile(df)` — read-only data-quality insight using the same inference as `clean`. |
| Plans & comparisons | `fd.suggest_plan`, `fd.compare_plans`, `fd.compare_clean`, `fd.explain_clean`. |
| Safe defaults | Targets, IDs, and free-text columns are protected from leakage and corruption. |
| Typed & tested | `py.typed`, 800+ tests, 95%+ coverage, mypy-clean. |
| pandas-first | Pure pandas + NumPy core; no heavy dependencies required. |

## The enterprise layer

`freshdata.enterprise` adds opt-in governance and data-quality capabilities. It
accepts and returns **either pandas or Polars** — running Polars-native fast
paths when available and falling back to vectorized pandas otherwise. Optional
dependencies stay lazy, so a plain `import freshdata` is unaffected.

| Capability | API |
|---|---|
| Full enterprise pipeline | `clean_enterprise(df, *, enterprise=…)` → `EnterpriseResult` |
| Data Trust Score (0–100) | `compute_trust_score(df)` → completeness / validity / uniqueness / consistency |
| Fuzzy value clustering | `merge_clusters(df, cols)` / `cluster_column(df, col)` |
| PII masking | `mask_dataframe(df, rules)` — hash / redact / partial / regex-scrub / drop |
| Semantic validation | `run_semantic_validation(df, configs)` — reference / regex / API checks |
| Lineage | `LineageTracker` / `schema_of` — OpenLineage-compatible metadata |
| Label-noise (ML) | `detect_label_issues` / `detect_outliers` — optional Cleanlab wrappers |
| Batch CLI | `freshdata clean | trust | profile` with quality-gate exit codes |

```python
from freshdata.enterprise import clean_enterprise, EnterpriseConfig, ClusterConfig

ec = EnterpriseConfig(enable_clustering=True, clustering=ClusterConfig(columns=("vendor",)), fail_under_trust=80)
result = clean_enterprise(df, enterprise=ec)
print(result.quality.to_markdown())
assert result.passed_gate
```

## Compliance reports

The `freshdata.compliance` subpackage turns a `CleanReport` into a regulatory
audit artifact, mapping freshdata's transformations onto named control
frameworks — 21 CFR Part 11, GDPR (Art. 30/17), ALCOA+, SOX-404, and HIPAA Safe
Harbor. The generators are purely additive and report-only. See the
[compliance reports guide](compliance.md).

## Orchestration integrations

Run freshdata's clean + trust gate inside Dagster, Airflow, or dbt and warn / fail /
skip a pipeline on low data quality. See the
[orchestration integrations guide](integrations.md).

## Polars support

```python
import polars as pl
import freshdata as fd

cleaned = fd.clean(pl_df)   # returns a pl.DataFrame when the input is Polars
```

Install with `pip install "freshdata-cleaner[polars]"`.
