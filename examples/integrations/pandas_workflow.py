"""examples/integrations/pandas_workflow.py — Seamless pandas Workflow Integration.

Problem:
--------
Existing data teams have extensive pandas-based data transformation pipelines.
Rewriting whole pipelines in a new framework is costly and risky. Data teams need
a drop-in cleaning step that accepts a pandas DataFrame, returns a standard pandas
DataFrame, and preserves all downstream indexing and slicing behavior.

Installation:
-------------
pip install "freshdata-cleaner"
"""

from __future__ import annotations

import pandas as pd

import freshdata as fd


def main() -> None:
    print("=== Integration: Drop-in pandas Cleaning Pipeline ===")

    # 1. Existing pandas pipeline ingest
    df = pd.DataFrame({
        "Department": [" Engineering ", "Sales", "marketing", "Sales", " Engineering "],
        "Employee_ID": ["EMP-01", "EMP-02", "EMP-03", "EMP-04", "EMP-05"],
        "Salary": ["$125,000", "$95,000", "N/A", "$105,000", "$140,000"],
        "Performance_Score": [4.5, 3.8, 4.2, -1.0, 5.0],
    })
    print("\n[1] Raw pandas DataFrame:")
    print(df)

    # 2. FreshData clean step (seamlessly integrated into method chain or function)
    cleaned_df, report = fd.clean(
        df,
        preserve_columns=["Employee_ID"],
        return_report=True,
    )

    print("\n[2] Cleaned pandas DataFrame:")
    print(cleaned_df)
    assert isinstance(cleaned_df, pd.DataFrame), "Output must be a standard pandas DataFrame"

    # 3. Downstream pandas aggregation
    print("\n[3] Downstream pandas Aggregation:")
    summary = cleaned_df.groupby("department")["salary"].mean().reset_index()
    print(summary)

    print("\n[4] Audit Trail Rationale:")
    for a in report.actions:
        print(f"  - [{a.step}] {a.column}: {a.description}")


if __name__ == "__main__":
    main()
