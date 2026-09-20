"""examples/integrations/sklearn_pipeline.py — scikit-learn Pipeline Integration.

Problem:
--------
Data science workflows rely on `sklearn.pipeline.Pipeline` for reproducible,
leakage-free model training and inference. Raw inputs entering the pipeline
often need automated tabular cleaning before standard imputation, one-hot
encoding, or scaling can take place.

Installation:
-------------
pip install "freshdata-cleaner[ml]"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

import freshdata as fd


class FreshDataCleaner(BaseEstimator, TransformerMixin):
    """Scikit-Learn compatible transformer wrapping FreshData."""

    def __init__(self, strategy: str = "balanced", preserve_columns: tuple[str, ...] = ()) -> None:
        self.strategy = strategy
        self.preserve_columns = preserve_columns

    def fit(self, X: pd.DataFrame, y: object = None) -> FreshDataCleaner:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return fd.clean(
            X,
            strategy=self.strategy,
            preserve_columns=self.preserve_columns,
            verbose=False,
        )


def main() -> None:
    print("=== Integration: scikit-learn Pipeline ===")

    # 1. Generate training data with dirty representations
    n = 200
    rng = np.random.default_rng(42)
    X_raw = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.choice(["$45,000", "$60,000", "N/A", "$85,000"], n),
        "experience": rng.integers(1, 20, n).astype(float),
    })
    # Inject missingness
    X_raw.loc[rng.choice(n, 15, replace=False), "age"] = np.nan
    y = (X_raw["experience"] > 10).astype(int)

    print("\n[1] Raw Features:")
    print(X_raw.head(4))

    # 2. Build scikit-learn pipeline with FreshData as the first stage
    cleaner = FreshDataCleaner(strategy="balanced")
    cleaned_X = cleaner.transform(X_raw)

    print("\n[2] Cleaned Features from Transformer:")
    print(cleaned_X.head(4))

    # 3. Fit classifier on cleaned features
    # Select numeric features for model fitting
    numeric_cols = [c for c in cleaned_X.columns if not c.endswith("_outlier")]
    model = LogisticRegression()
    model.fit(cleaned_X[numeric_cols], y)
    print(f"\n[3] Model fitted successfully! Score: {model.score(cleaned_X[numeric_cols], y):.3f}")


if __name__ == "__main__":
    main()
