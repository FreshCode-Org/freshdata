---
title: Orchestration integrations
description: >-
  Run freshdata's clean + trust gate inside Dagster, Airflow, and dbt — warn, fail,
  or skip a pipeline on low data quality.
keywords: dagster data quality, airflow data quality operator, dbt data quality gate, freshdata trust gate, data quality orchestration
---

# Orchestration integrations

`freshdata.integrations` lets you run freshdata's clean step and a **trust gate**
inside your orchestrator: clean a DataFrame, score the cleaned result with the 0-100
[Data Trust Score](feature-overview.md), and **warn / fail / skip** when it falls
below a threshold.

The framework-agnostic core, [`evaluate_trust_gate`](#the-trust-gate), is always
importable and has no extra dependencies. Each orchestrator adapter is an opt-in
extra:

```bash
pip install "freshdata[dagster]"      # or [airflow], [dbt], or [integrations] for all
```

Each adapter module imports cleanly even when its framework is absent; the framework
is required only when you actually use the adapter.

## The trust gate

```python
from freshdata.integrations import evaluate_trust_gate

cleaned, result = evaluate_trust_gate(
    df,
    trust_score_threshold=80.0,   # minimum acceptable 0-100 score
    on_low_score="fail",          # "warn" (default), "fail", or "skip"
    publish_full_report=True,     # attach the clean report (+ compliance, if installed)
)
result.passed            # bool: trust_score >= threshold
result.trust_score       # 0-100
result.high_risk_count   # number of high-risk cleaning actions
result.as_metadata()     # flat {"freshdata/...": value} dict for dashboards
```

`evaluate_trust_gate` never raises on a low score — it returns a `TrustGateResult`
whose `should_fail` / `should_skip` flags let each adapter react in its own terms.
When `publish_full_report=True` and [`freshdata.compliance`](compliance.md) is
installed, a SOX-404 compliance bundle is folded into `result.report_dict`
automatically; otherwise that step is a silent no-op.

## Dagster

```python
from dagster import asset
from freshdata.integrations.dagster import freshdata_asset_check, FreshDataResource

@asset
def orders(): ...

check_orders = freshdata_asset_check(
    asset=orders, trust_score_threshold=80.0, on_low_score="fail"
)
```

The check returns an `AssetCheckResult` with `freshdata/*` metadata. Severity is
`ERROR` when `on_low_score="fail"` and `WARN` otherwise, so a strict gate surfaces as
an error in the Dagster UI while a soft gate stays a warning. `FreshDataResource` is a
`ConfigurableResource` bundling the same options for use inside assets/ops via
`resource.gate(df)`.

## Airflow

```python
from freshdata.integrations.airflow import FreshDataCleanOperator

clean_orders = FreshDataCleanOperator(
    task_id="clean_orders",
    input_task_id="extract_orders",   # pulls the DataFrame from this task's XCom
    trust_score_threshold=80.0,
    on_low_score="fail",              # raises AirflowException; "skip" -> AirflowSkipException
)
```

The operator pulls the upstream DataFrame from XCom, cleans and gates it, and pushes
the cleaned frame plus the gate result (under `<output_xcom_key>__gate`) back to
XCom. A failing gate raises `AirflowException` (`on_low_score="fail"`) or
`AirflowSkipException` (`"skip"`); `"warn"` logs and continues.

## dbt

Gate every model after `dbt run` — typically as a CI step — with the `dbt-gate`
console script (installed by the `dbt` extra):

```bash
export FRESHDATA_WAREHOUSE_CONN="postgresql://user:pass@host/db"
dbt-gate --manifest target/manifest.json --threshold 80 --fail
```

It parses dbt's `manifest.json`, reads each model's materialized table via SQLAlchemy,
gates it, and exits non-zero (with `--fail`) if any model is below the threshold. For
a single model — or to write per-model `<model>_audit.json` files — use
`FreshDataDbtTransform`:

```python
from freshdata.integrations.dbt import FreshDataDbtTransform

result = FreshDataDbtTransform(
    model_name="analytics.orders",
    output_dir="target/freshdata",
    trust_score_threshold=80.0,
    fail_on_low_score=True,
).run()
```

A bundled Jinja macro, `freshdata_trust_gate`, documents the recommended `on-run-end`
invocation; see `freshdata/integrations/dbt/macros/freshdata_trust_gate.sql`.

## On a low score

`on_low_score` controls how each adapter reacts when `trust_score < threshold`:

| Value | Dagster | Airflow | dbt (`dbt-gate`) |
| --- | --- | --- | --- |
| `"warn"` (default) | `AssetCheckResult(passed=False)`, `WARN` | logs, task succeeds | reported, exit 0 |
| `"fail"` | `AssetCheckResult(passed=False)`, `ERROR` | raises `AirflowException` | exit 1 with `--fail` |
| `"skip"` | `AssetCheckResult(passed=False)`, `WARN` | raises `AirflowSkipException` | reported |
