"""05_ml_preprocessing.py — ML-Ready Preprocessing with Zero Target Leakage.

Problem:
--------
In machine learning pipelines, improper data preprocessing is the #1 cause of
data leakage and model degradation:
1. Target leakage: Naive imputation across the entire DataFrame uses target labels
   or imputes missing labels, contaminating ground truth.
2. Silent corruption: Arbitrary scaling or imputation across identifier columns
   distorts feature splits in tree-based and linear models.

Dataset:
--------
600 customer subscription records with feature columns (tenure, monthly spend,
subscription tier) and a binary churn label (`churn`).

Installation:
-------------
pip install "freshdata-cleaner[ml]"

Expected Result:
----------------
FreshData prepares the feature set for ML modeling:
- Target column (`churn`) is strictly protected and never imputed or modified.
- Missing values in features are imputed using role-aware strategies (median for
  numeric spend, mode for categorical tier).
- Identifier column (`customer_id`) is recognized and kept pristine.
- Output is directly consumable by scikit-learn estimators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import freshdata as fd


def generate_churn_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    tenure = rng.integers(1, 72, n)
    monthly_charges = np.round(rng.normal(65, 20, n), 2)
    plan = rng.choice(["basic", "pro", "enterprise"], n, p=[0.5, 0.3, 0.2])
    # Churn propensity correlated with tenure and high charges
    churn_prob = 1.0 / (1.0 + np.exp(-(-1.5 - 0.05 * tenure + 0.04 * monthly_charges)))
    churn = (rng.random(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{1000 + i}" for i in range(n)],
        "tenure_months": tenure.astype(float),
        "monthly_charges": monthly_charges,
        "plan": plan,
        "churn": churn,
    })

    # Inject missingness into features only
    df.loc[rng.choice(n, 45, replace=False), "monthly_charges"] = np.nan
    df.loc[rng.choice(n, 20, replace=False), "plan"] = np.nan
    return df


def main() -> None:
    print("=== FreshData Flagship Example 05: Machine Learning Preprocessing ===")
    raw_df = generate_churn_dataset()

    print(f"\n[1] Raw Dataset ({len(raw_df)} samples):")
    print(raw_df.head(4))
    print(f"\nMissing values before cleaning:\n{raw_df.isna().sum()}")

    # 1. Clean with explicit target protection to guarantee zero leakage
    print("\n--- [2] Cleaning with Target Protection ---")
    cleaned_df, report = fd.clean(
        raw_df,
        target_column="churn",
        preserve_columns=["customer_id"],
        strategy="balanced",
        return_report=True,
    )

    # Invariant assertion: target column must never be modified
    assert cleaned_df["churn"].equals(raw_df["churn"]), "Invariant violation: target was modified!"
    print("Target integrity verified: 0% mutation on ground truth labels.")

    print("\nMissing values after cleaning:")
    print(cleaned_df.isna().sum())

    # 2. Prepare feature matrix and split
    print("\n--- [3] Training scikit-learn Classifier ---")
    feature_cols = [
        c for c in cleaned_df.columns
        if c not in ("customer_id", "churn") and not c.endswith("_outlier")
    ]
    X = pd.get_dummies(cleaned_df[feature_cols], drop_first=True)
    y = cleaned_df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"Random Forest Train Accuracy: {train_acc:.3f}")
    print(f"Random Forest Test Accuracy:  {test_acc:.3f}")

    print("\n[4] FreshData Audit Summary:")
    print(report.summary())


if __name__ == "__main__":
    main()
