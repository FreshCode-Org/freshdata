---
title: How to add data cleaning to an ETL pipeline
description: Automate data hygiene and quality gating inside Airflow, Prefect, and Dagster pipelines.
keywords: etl data cleaning pipeline, airflow data quality gate, python etl data cleaning, dagster data quality
---

# How to add data cleaning to an ETL pipeline

* **Search Intent**: Data engineers wanting to embed automated data cleaning and quality gates into scheduled ETL/ELT pipelines (Airflow, Prefect, Dagster).
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

In enterprise data pipelines, dirty partner data or third-party webhooks frequently cause daily batch jobs to fail unexpectedly. Data engineers need:
1. An automated cleaning layer at ingest boundaries.
2. A quantifiable **Trust Gate** that evaluates data quality on a 0–100 scale and halts downstream loads if anomalies exceed tolerable limits.

---

## Code

```bash
pip install "freshdata-cleaner[airflow]"
```

```python
import pandas as pd
from freshdata.integrations._core import evaluate_trust_gate

# 1. Dirty data extracted from upstream storage
raw_batch = pd.DataFrame({
    "order_id": ["O-1", "O-2", "O-3", "O-4"],
    "amount": ["$120.00", "N/A", "$85.50", "$340.00"],
    "customer_rating": [5, 4, 999, 5],  # 999 is an invalid outlier
})

# 2. Clean and evaluate Trust Gate
cleaned_batch, gate_result = evaluate_trust_gate(
    raw_batch,
    trust_score_threshold=80.0,
    on_low_score="warn",  # Options: "warn", "fail", "skip"
)

print(f"Trust Gate Passed: {gate_result.passed}")
print(f"Trust Score:       {gate_result.trust_score:.1f} / 100.0 (Grade {gate_result.grade})")
print(f"Gate Message:      {gate_result.message}")

if gate_result.passed:
    # Load clean data to data warehouse (BigQuery / Snowflake)
    print("Writing clean data to data warehouse...")
```

---

## Output

```text
Trust Gate Passed: True
Trust Score:       97.5 / 100.0 (Grade A)
Gate Message:      freshdata trust gate passed: score 97.5 (grade A) vs threshold 80.0; 0 high-risk action(s); 4 -> 4 rows.
Writing clean data to data warehouse...
```

---

## Explanation

* **`evaluate_trust_gate`**: Cleans the input DataFrame and computes an objective Trust Score based on the severity and proportion of anomalies detected.
* **Orchestration Integration**: If data corruption is unacceptable, `on_low_score="fail"` raises an exception, halting downstream DAG tasks and alerting on-call engineers before bad data lands in Snowflake or BigQuery.
* **Native Airflow Operator**: For Airflow users, FreshData provides `FreshDataCleanOperator` to manage XCom passing and task state natively.

---

## Next Steps

* See [Airflow Integration Recipe](https://github.com/FreshCode-Org/freshdata/blob/main/examples/integrations/airflow_task.py).
* Learn about [Orchestration Integrations](../integrations.md).
