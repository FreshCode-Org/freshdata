"""examples/integrations/airflow_task.py — Apache Airflow & Pipeline Orchestration.

Problem:
--------
In production data engineering pipelines (Airflow, Prefect, Dagster), dirty data
often causes downstream SQL loads or ML training tasks to fail silently or crash.
Data teams need:
1. Automated cleaning at pipeline boundaries.
2. A strict **Trust Gate** that scores data quality and halts or warns if incoming
   data corruption exceeds tolerable thresholds.

Installation:
-------------
pip install "freshdata-cleaner[airflow]"

Expected Result:
----------------
- FreshData cleans incoming frames and evaluates a deterministic Trust Score (0–100).
- If the trust score satisfies the threshold, cleaned data flows downstream.
- If data corruption is unacceptable, the task fails or warns with full audit context.
"""

from __future__ import annotations

import pandas as pd

from freshdata.integrations._core import evaluate_trust_gate


def simulate_airflow_python_task() -> None:
    print("=== Integration: Pipeline Orchestration & Trust Gate ===")

    # 1. Simulating an upstream task pulling raw data from an external partner
    raw_partner_data = pd.DataFrame({
        "account_id": ["ACC-001", "ACC-002", "ACC-003", "ACC-004"],
        "balance": ["$1,500.00", "N/A", "$3,200.75", "$450.00"],
        "credit_score": [720, 680, 810, -50],  # -50 is an invalid outlier
        "status": [" active ", "active", "pending", "active"],
    })
    print("\n[1] Upstream Raw Partner Data:")
    print(raw_partner_data)

    # 2. Evaluate Trust Gate (used inside Airflow operators or custom pipeline tasks)
    print("\n[2] Executing FreshData Trust Gate (Threshold: 80.0)...")
    cleaned_df, gate_result = evaluate_trust_gate(
        raw_partner_data,
        trust_score_threshold=80.0,
        on_low_score="warn",
    )

    print(f"\nTrust Gate Passed: {gate_result.passed}")
    print(f"Trust Score:       {gate_result.trust_score:.1f} / 100.0 (Grade: {gate_result.grade})")
    print(f"Gate Message:      {gate_result.message}")

    print("\n[3] Cleaned DataFrame Ready for Warehouse Load:")
    print(cleaned_df)

    # 3. Airflow DAG Operator Definition Pattern (Reference):
    print("\n[4] Airflow DAG Operator Pattern (for use in dags/):")
    dag_sample = '''
from airflow import DAG
from freshdata.integrations.airflow import FreshDataCleanOperator

with DAG(dag_id="partner_ingest_pipeline", schedule="@daily") as dag:
    clean_task = FreshDataCleanOperator(
        task_id="clean_and_gate_partner_data",
        input_task_id="extract_partner_s3",
        trust_score_threshold=85.0,
        on_low_score="fail",  # halts downstream DAG if quality drops
    )
'''
    print(dag_sample)


if __name__ == "__main__":
    simulate_airflow_python_task()
