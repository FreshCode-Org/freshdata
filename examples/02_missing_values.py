"""02_missing_values.py — Role-Aware Missing Value Handling.

Problem:
--------
Naive data cleaning tools apply a single global rule for missing values
(e.g. fillna with 0, mean, or mode across all columns). This practice damages
data: imputing customer IDs creates phantom users, imputing target labels leaks
ground truth into ML models, and mean-imputing highly skewed financial columns
distorts distributions.

Dataset:
--------
A synthetic dataset of 250 customer profiles with distinct feature roles:
- customer_id: primary key identifier
- age: symmetric numeric feature
- income: skewed financial metric
- tier: categorical status
- churned: ML prediction target

Installation:
-------------
pip install "freshdata-cleaner[ml]"

Expected Result:
----------------
FreshData inspects each column's inferred role and distribution:
- Identifiers and target labels are preserved (never imputed).
- Low-skew numerics receive median imputation.
- Categoricals receive mode imputation or separate missing category.
- Every decision is accompanied by a rationale and confidence score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import freshdata as fd


def generate_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 250
    df = pd.DataFrame({
        "customer_id": [f"USR-{1000 + i}" for i in range(n)],
        "age": np.round(rng.normal(38, 10, n)),
        "annual_income": np.round(rng.lognormal(10.5, 0.6, n)),
        "tier": rng.choice(["Bronze", "Silver", "Gold"], n, p=[0.5, 0.3, 0.2]),
        "churned": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })

    # Inject realistic missing patterns
    df.loc[rng.choice(n, 5, replace=False), "customer_id"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "age"] = np.nan
    df.loc[rng.choice(n, 25, replace=False), "annual_income"] = np.nan
    df.loc[rng.choice(n, 18, replace=False), "tier"] = np.nan
    df.loc[rng.choice(n, 6, replace=False), "churned"] = np.nan

    return df


def main() -> None:
    print("=== FreshData Flagship Example 02: Role-Aware Missing Values ===")
    df = generate_dataset()

    print("\n[1] Missing values before cleaning:")
    print(df.isna().sum())

    # Clean with balanced strategy and protected target column
    cleaned, report = fd.clean(
        df,
        target_column="churned",
        strategy="balanced",
        return_report=True,
    )

    print("\n[2] Missing values after cleaning:")
    print(cleaned.isna().sum())

    print("\n[3] Actions taken by decision engine:")
    for a in report.actions:
        if a.step == "missing":
            print(f"  - Column '{a.column}': {a.description}")
            rat = a.rationale or "Standard role-based policy"
            print(f"    Rationale: {rat} | Risk: {a.risk} | Confidence: {a.confidence}")

    print("\n[4] Summary:")
    print(report.summary())


if __name__ == "__main__":
    main()
