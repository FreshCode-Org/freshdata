---
title: How to audit automated data cleaning in Python
description: Generate complete, explainable audit trails and machine-readable logs for automated data cleaning.
keywords: data cleaning audit trail, explainable data cleaning python, audit dataframe changes, data lineage log
---

# How to audit automated data cleaning in Python

* **Search Intent**: Data governance and compliance engineers looking to verify, audit, and explain automated data cleaning decisions to regulators, stakeholders, and reviewers.
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

Automated data cleaning is often treated as a "black box". Regulated industries (finance, healthcare, insurance) and safety-critical ML pipelines cannot deploy automated tools without answering:
1. What exact cells were altered?
2. Why was that action chosen over alternatives?
3. What was the risk level and model confidence in each decision?
4. What cells were deliberately left untouched and require human review?

---

## Code

```python
import json
import pandas as pd
import freshdata as fd

df = pd.DataFrame({
    "patient_id": ["P-101", "P-102", "P-103"],
    "systolic_bp": [120, -10, 140],           # -10 is an impossible outlier
    "lab_result": ["Normal", "N/A", "Normal"], # "N/A" sentinel
})

# Clean with full audit reporting
cleaned, report = fd.clean(
    df,
    preserve_columns=["patient_id"],
    return_report=True,
)

# 1. Print human-readable summary
print("=== Human-Readable Summary ===")
print(report.summary())

# 2. Inspect structured Action objects
print("\n=== Structured Actions ===")
for action in report.actions:
    print(f"Step: {action.step} | Column: {action.column} | Risk: {action.risk} | Confidence: {action.confidence}")
    print(f"  Rationale: {action.rationale or action.description}")

# 3. Export to JSON for enterprise SIEM or compliance storage
audit_json = report.to_json()
print("\n=== Audit JSON Preview ===")
print(audit_json[:300] + "...")
```

---

## Output

```text
=== Human-Readable Summary ===
freshdata clean report
  rows:    3 -> 3 (+0)
  columns: 3 -> 4 (+1)
  missing: 0 -> 1 cell(s)
  actions (3):
    - [normalize_sentinels] 'lab_result': replaced sentinel "N/A" with missing
    - [missing] 'lab_result': preserved 1 missing value(s)
    - [outliers] 'systolic_bp': flagged 1 outlier(s) in new column 'systolic_bp_outlier'

=== Structured Actions ===
Step: normalize_sentinels | Column: lab_result | Risk: low | Confidence: 1.0
  Rationale: replaced sentinel strings ("N/A", "-", "", …) with missing
Step: missing | Column: lab_result | Risk: medium | Confidence: 0.6
  Rationale: preserved 1 missing value(s)
Step: outliers | Column: systolic_bp | Risk: low | Confidence: 0.5
  Rationale: flagged 1 outlier(s) (method=iqr, factor=1.5) in new column 'systolic_bp_outlier'
```

---

## Explanation

Every execution of `fd.clean(..., return_report=True)` generates an immutable `CleanReport`:
* **`report.actions`**: List of individual `Action` dataclasses capturing `step`, `column`, `description`, `count`, `rationale`, `risk` (`low`, `medium`, `high`), and `confidence` ($0.0 \le c \le 1.0$).
* **`report.to_json()`**: Generates a schema-compliant JSON payload that can be archived into S3, Splunk, Datadog, or data catalogs for regulatory audits.
* **`report.recommendations`**: Highlights ambiguous cases where human review is advised.

---

## Next Steps

* See [Flagship Example 01: CSV Cleaning with Audit Trail](https://github.com/FreshCode-Org/freshdata/blob/main/examples/01_csv_cleaning.py).
* Learn about [Compliance Report Generation](../compliance.md).
