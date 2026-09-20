"""01_csv_cleaning.py — End-to-End Messy CSV Cleaning with Audit Trail.

Problem:
--------
Raw CSV exports from operational databases, web forms, and third-party systems
consistently contain messy formatting: leading/trailing whitespace, dirty
currency symbols, mixed datetime formats, multiple sentinel strings for missing
data ("N/A", "null", "-", "missing"), and unstructured column names. Manually
patching these issues with ad-hoc pandas code requires dozens of lines and
often introduces silent errors.

Dataset:
--------
A realistic in-memory CSV export of customer billing records with formatting
glitches, dirty sentinels, and type inconsistencies.

Installation:
-------------
pip install "freshdata-cleaner[cli]"

Expected Result:
----------------
Columns are standardized to snake_case, whitespace is trimmed, sentinels are
normalized to proper NaNs, currency values are converted to clean floats, dates
are parsed, and a complete audit report is produced with zero cell changes
left unexplained.
"""

from __future__ import annotations

import io

import pandas as pd

import freshdata as fd

# 1. Raw messy CSV content
RAW_CSV = """Customer ID, Full Name ,Age,Signup Date,Annual Spend,Status
C-101,  Alice Smith  ,29,2023-01-15,$1200.50,active
C-102,Bob Jones,34,2023/02/20,$450.00,pending
C-103,Charlie Brown,-4,N/A,$3100.00,active
C-104,  Diana Prince,42,2023-04-10,missing,active
C-105,Evan Wright,165,not_a_date,$890.25,cancelled
"""


def main() -> None:
    print("=== FreshData Flagship Example 01: CSV Cleaning ===")

    # 2. Read raw CSV
    df = pd.read_csv(io.StringIO(RAW_CSV))
    print("\n[1] Raw Input DataFrame:")
    print(df)

    # 3. Clean the DataFrame in one call with full audit reporting
    # We specify customer_id as protected to ensure primary keys are never mutated
    cleaned_df, report = fd.clean(
        df,
        preserve_columns=["customer_id"],
        return_report=True,
    )

    print("\n[2] Cleaned DataFrame:")
    print(cleaned_df)

    print("\n[3] Audit Report Summary:")
    print(report.summary())

    print("\n[4] Structured Action Log:")
    for action in report.actions:
        print(
            f"  - Step: {action.step:<18} | Column: {str(action.column):<14} | "
            f"Risk: {action.risk:<6} | Confidence: {action.confidence:.2f} | "
            f"Description: {action.description}"
        )

    # 5. Export audit report to machine-readable JSON
    audit_dict = report.to_dict()
    print(f"\n[5] Machine-Readable Audit Output: {len(audit_dict['actions'])} actions recorded")
    print(f"    Cells changed: {audit_dict.get('cells_changed', 'N/A')}")
    print(f"    Execution time: {audit_dict.get('duration_seconds', 0):.4f}s")


if __name__ == "__main__":
    main()
