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
| Typed & tested | `py.typed`, 1,200+ tests, 93% coverage gate enforced in CI, mypy-clean. |
| pandas-first | Pure pandas + NumPy core; no heavy dependencies required. |

## The enterprise layer

`freshdata.enterprise` adds opt-in governance and data-quality capabilities. It
accepts and returns **either pandas or Polars** — running Polars-native fast
paths when available and falling back to vectorized pandas otherwise. Optional
dependencies stay lazy, so a plain `import freshdata` is unaffected.

| Capability | API |
|---|---|
| Full enterprise pipeline | `clean_enterprise(df, *, enterprise=…)` → `EnterpriseResult` |
| Data Trust Score (0–100) | `compute_trust_score(df)` → completeness / validity / uniqueness / consistency (uniqueness is unknown and left out of the overall when a column holds unhashable values such as lists) |
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

### `freshdata clean --config` files

`--config` takes a JSON or YAML object with two optional sections, `clean` and
`enterprise`. An unknown section or key, including a typo, stops the run before
any data is read: a one-line error with a "did you mean" hint, exit 1.

```yaml
clean:
  strategy: balanced
enterprise:
  fail_under_trust: 80
  masking:
    - {name: pii, columns: [email], strategy: hash}
  enable_privacy_detection: true
  privacy: {min_score: 0.6}
```

`clean` accepts any `CleanConfig` option. `enterprise` accepts these keys:

| Key | Value | Builds |
|---|---|---|
| `actor` | string or null | `EnterpriseConfig.actor` |
| `fail_under_trust` | number from 0 to 100, or null | the trust gate; `--fail-under-trust` overrides it |
| `enable_masking`, `enable_clustering`, `enable_validation`, `enable_lineage`, `enable_privacy_detection`, `enable_entity_resolution` | `true` or `false` | the toggle of the same name |
| `masking` | list of objects | one `MaskingRule` each; `--mask` adds more |
| `semantic` | list of objects | one `SemanticValidatorConfig` each |
| `clustering` | object | `ClusterConfig`; `--cluster` replaces it |
| `trust_weights` | object | `TrustScoreWeights` |
| `lineage` | object | `LineageConfig` |
| `privacy` | object | `PIIDetectionConfig`, applied when `enable_privacy_detection` is true |
| `k_anonymity` | object | `KAnonymityConfig` |
| `entity_resolution` | object, with `blocking_rules` and `comparisons` as lists of objects | `EntityResolutionConfig` (with `BlockingRule` and `ComparisonLevel`), applied when `enable_entity_resolution` is true |

Nested objects take the field names of the class they build, and unknown names
are rejected the same way. Three `EnterpriseConfig` fields do nothing in
`freshdata clean`: `enable_contracts` and `drift` (the command takes no baseline or
data contract to check against) and `anonymization` (no pipeline applies it; use
`masking`, or `privacy` with `enable_privacy_detection`). They are accepted and
ignored when null or set to their default (`enable_contracts: false`, `drift: {}`
or an object of `DriftConfig` defaults, `anonymization: []`). Any other value is
rejected with that explanation.

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
