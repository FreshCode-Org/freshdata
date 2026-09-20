"""04_outlier_handling.py — Non-Destructive Outlier Flagging vs Controlled Capping.

Problem:
--------
Automated data-cleaning pipelines frequently corrupt data when handling outliers:
1. Blindly dropping outlier rows destroys valid extreme events (e.g. Black Friday
   spending spikes or rare medical conditions) and causes row-alignment errors.
2. Blindly winsorizing/capping distorts variance and ruins statistical tests.

Dataset:
--------
300 transaction records with a core normal distribution (mean 100, std 15) and
10 injected extreme anomalies (mean 1200, std 50).

Installation:
-------------
pip install freshdata-cleaner

Expected Result:
----------------
- By default (under strategy="balanced"), FreshData flags outliers non-destructively
  in a dedicated boolean column (`amount_outlier`) with zero row loss.
- Controlled alternatives (`outlier_action="cap"` or `"drop"`) can be explicitly
  specified when required, with every decision documented in the audit report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import freshdata as fd


def generate_transactions(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    amounts = rng.normal(100, 15, n)
    # Inject 10 extreme outliers
    amounts[:10] = rng.normal(1_200, 50, 10)
    quantities = rng.integers(1, 6, n)
    return pd.DataFrame({
        "transaction_id": [f"TX-{10000 + i}" for i in range(n)],
        "amount": np.round(amounts, 2),
        "quantity": quantities,
    })


def main() -> None:
    print("=== FreshData Flagship Example 04: Non-Destructive Outlier Handling ===")
    df = generate_transactions()

    print(f"\n[1] Raw Dataset Summary ({len(df)} rows):")
    print(f"    Min amount: ${df['amount'].min():.2f}")
    print(f"    Max amount: ${df['amount'].max():.2f}")
    print(f"    Mean amount: ${df['amount'].mean():.2f}")

    # Approach 1: Non-destructive flagging (FreshData Default)
    print("\n--- [A] Default Mode: Non-Destructive Flagging ---")
    flagged_df, report_flagged = fd.clean(
        df,
        preserve_columns=["transaction_id"],
        return_report=True,
    )
    print(f"Cleaned DataFrame rows: {len(flagged_df)} (Zero row loss)")
    print("New indicator column added: 'amount_outlier'")
    num_outliers = int(flagged_df["amount_outlier"].sum())
    print(f"Number of outliers flagged: {num_outliers}")
    print("\nSample flagged rows:")
    print(flagged_df[flagged_df["amount_outlier"]].head(3))

    # Approach 2: Controlled capping (Winsorization)
    print("\n--- [B] Controlled Capping (Winsorization) ---")
    capped_df, report_capped = fd.clean(
        df,
        outlier_action="cap",
        preserve_columns=["transaction_id"],
        return_report=True,
    )
    max_capped = capped_df['amount'].max()
    max_orig = df['amount'].max()
    print(f"Capped Max amount: ${max_capped:.2f} (original was ${max_orig:.2f})")

    # Approach 3: Explicit row removal (when explicitly requested)
    print("\n--- [C] Explicit Row Removal ---")
    dropped_df, report_dropped = fd.clean(
        df,
        outlier_action="remove",
        preserve_columns=["transaction_id"],
        return_report=True,
    )
    print(f"Rows after removal: {len(dropped_df)} (removed {len(df) - len(dropped_df)} rows)")

    print("\nAudit Report Rationale (from Default Mode):")
    for a in report_flagged.actions:
        if a.step == "outliers":
            print(f"  - Column '{a.column}': {a.description}")
            print(f"    Risk: {a.risk} | Confidence: {a.confidence}")


if __name__ == "__main__":
    main()
