---
title: API reference
description: >-
  Complete freshdata API reference — clean, profile, suggest_plan,
  compare_plans, compare_clean, explain_clean, Cleaner, CleanConfig, and reports.
keywords: freshdata api, fd.clean, CleanConfig, CleanReport, pandas cleaning api
---

# API reference

Auto-generated from the source docstrings. Everything below is available as a
top-level attribute of `freshdata` (e.g. `import freshdata as fd; fd.clean(...)`).

## Cleaning

::: freshdata.clean

::: freshdata.Cleaner

## Profiling & inspection

::: freshdata.profile

::: freshdata.infer_roles

::: freshdata.explain_clean

## Planning & comparison

::: freshdata.suggest_plan

::: freshdata.compare_plans

::: freshdata.compare_clean

## Configuration

::: freshdata.CleanConfig

## Reports & results

::: freshdata.CleanReport

::: freshdata.Action

::: freshdata.CleanPlan

::: freshdata.ColumnPlan

::: freshdata.Profile

::: freshdata.ColumnProfile

::: freshdata.ExplainReport

## Enterprise layer

The `freshdata.enterprise` subpackage is documented in the
[feature overview](feature-overview.md). Import its symbols lazily:

```python
from freshdata.enterprise import clean_enterprise, EnterpriseConfig
```

## Compliance

The `freshdata.compliance` subpackage maps a `CleanReport` onto regulatory control
frameworks; it is documented in the [compliance reports guide](compliance.md).
Import its symbols lazily:

```python
from freshdata.compliance import generate_compliance_report, ComplianceConfig
```

## Integrations

The `freshdata.integrations` subpackage runs the clean + trust gate inside Dagster,
Airflow, and dbt; it is documented in the
[orchestration integrations guide](integrations.md). The framework-agnostic core is
always importable:

```python
from freshdata.integrations import evaluate_trust_gate, TrustGateResult
```
