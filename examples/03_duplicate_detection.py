"""03_duplicate_detection.py — Duplicate Row Detection and Safe Resolution.

Problem:
--------
Blindly running `df.drop_duplicates()` in automated data pipelines is dangerous.
In many real-world systems (e.g. retail orders, audit logs, sensor telemetry),
identical rows can represent valid, distinct events rather than technical
glitches. Furthermore, if a faulty database join creates an unexpected 40%
duplicate ratio, silently dropping rows masks an upstream outage.

Dataset:
--------
A realistic transaction dataset containing both valid repeated customer orders
and accidentally duplicated webhook payloads.

Installation:
-------------
pip install freshdata-cleaner

Expected Result:
----------------
FreshData detects duplicate rows and evaluates the duplicate ratio:
- By default, duplicates are reported and warned upon, but NOT silently removed.
- When opt-in `drop_duplicates=True` is supplied, FreshData safely removes
  redundant rows according to `duplicate_keep` semantics and records the
  exact count in the audit trail.
- If the duplicate ratio exceeds safe thresholds, an alert is raised.
"""

from __future__ import annotations

import pandas as pd

import freshdata as fd


def create_order_data() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["ORD-101", "ORD-102", "ORD-102", "ORD-103", "ORD-104", "ORD-104"],
        "customer_id": ["C-1", "C-2", "C-2", "C-3", "C-4", "C-4"],
        "amount": [49.99, 120.00, 120.00, 15.50, 89.00, 89.00],
        "status": ["completed", "completed", "completed", "shipped", "pending", "pending"],
    })


def main() -> None:
    print("=== FreshData Flagship Example 03: Duplicate Detection & Resolution ===")
    df = create_order_data()

    print(f"\n[1] Raw DataFrame ({len(df)} rows):")
    print(df)

    # 1. Safe default behavior: FreshData warns and reports, but does NOT silently drop
    print("\n--- [A] Default Mode: Detection & Reporting Only (Safe Default) ---")
    cleaned_default, report_default = fd.clean(df, return_report=True)
    print(f"Rows after default clean: {len(cleaned_default)} (preserves all rows)")
    if report_default.warnings:
        print("Maintainer Warnings:")
        for w in report_default.warnings:
            print(f"  ! {w}")

    # 2. Opt-in resolution: Explicitly instruct FreshData to resolve duplicates
    print("\n--- [B] Opt-in Mode: Controlled Duplicate Resolution ---")
    cleaned_resolved, report_resolved = fd.clean(
        df,
        drop_duplicates=True,
        duplicate_keep="first",
        return_report=True,
    )
    dups = report_resolved.duplicates_removed
    print(f"Rows after resolution: {len(cleaned_resolved)} (removed {dups} duplicates)")
    print("\nResolved DataFrame:")
    print(cleaned_resolved)

    print("\nAudit Report Actions:")
    for a in report_resolved.actions:
        if "duplicate" in a.step or "duplicate" in a.description.lower():
            print(f"  - [{a.step}] {a.description} (risk: {a.risk}, count: {a.count})")


if __name__ == "__main__":
    main()
