"""Outlier detection and handling with freshdata.

In balanced mode freshdata *flags* outliers (adds a `<col>_outlier` column)
rather than silently removing or capping them. Run:

    python examples/02_outliers.py
"""

import numpy as np
import pandas as pd

import freshdata as fd


def make_data(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.normal(100, 20, 300)
    amount[:10] = rng.normal(1_000, 50, 10)  # 10 obvious outliers
    return pd.DataFrame({"amount": amount, "qty": rng.integers(1, 5, 300)})


def main() -> None:
    df = make_data()

    # Default (balanced) — flag outliers, keep the rows
    flagged, report = fd.clean(df, return_report=True)
    print(report.summary())
    if "amount_outlier" in flagged.columns:
        print(f"\nFlagged {int(flagged['amount_outlier'].sum())} outliers in 'amount'")

    # Explicit removal instead of flagging
    removed = fd.clean(df, outlier_action="remove", verbose=False)
    print(f"Rows after outlier_action='remove': {len(removed)} (from {len(df)})")

    # Explicit capping (winsorize) — honored even under the balanced default
    capped = fd.clean(df, outlier_action="cap", verbose=False)
    print(f"amount max after outlier_action='cap': {capped['amount'].max():.1f} "
          f"(was {df['amount'].max():.1f})")


if __name__ == "__main__":
    main()
