"""End-to-end machine-learning preprocessing pipeline with freshdata.

Clean -> encode -> split -> train, with the target protected from leakage. Run:

    pip install "freshdata-cleaner[ml]"
    python examples/05_ml_pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import freshdata as fd


def make_data(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    tenure = rng.integers(1, 72, n)
    monthly = rng.normal(70, 25, n)
    churn = ((tenure < 12) & (monthly > 60)).astype(int)
    df = pd.DataFrame(
        {
            "customer_id": range(n),
            "tenure_months": tenure.astype(float),
            "monthly_charges": monthly,
            "plan": rng.choice(["basic", "pro", "enterprise"], n),
            "churn": churn,
        }
    )
    df.loc[rng.choice(n, 50, replace=False), "monthly_charges"] = np.nan
    return df


def main() -> None:
    raw = make_data()

    # 1. Clean — protect id + target so they are never imputed/modified
    clean_df, report = fd.clean(
        raw, id_columns=("customer_id",), target_column="churn", return_report=True
    )
    print(report.summary())
    assert clean_df["churn"].equals(raw["churn"]), "target must be untouched"

    # 2. Encode + split + train on AI-ready data
    features = clean_df.drop(columns=["customer_id", "churn"])
    X = pd.get_dummies(features)
    y = clean_df["churn"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)
    print(f"\nTest accuracy: {model.score(X_te, y_te):.3f}")


if __name__ == "__main__":
    main()
