"""Feature normalization for machine learning.

freshdata cleans and types the data first; here we then scale numeric features
with scikit-learn for model-ready input. Run:

    pip install "freshdata-cleaner[ml]"
    python examples/03_normalization.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import freshdata as fd


def make_data(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 250
    return pd.DataFrame(
        {
            "height_cm": rng.normal(170, 10, n),
            "weight_kg": rng.normal(70, 15, n),
            "price": [f"${v:,.2f}" for v in rng.normal(500, 120, n)],  # currency strings
        }
    )


def main() -> None:
    df = make_data()

    # 1. Clean: currency strings -> float, fix dtypes, handle any missingness
    clean_df = fd.clean(df)
    print("dtypes after cleaning:\n", clean_df.dtypes, "\n")

    # 2. Normalize numeric features
    numeric = clean_df.select_dtypes("number")
    scaled = pd.DataFrame(
        StandardScaler().fit_transform(numeric),
        columns=numeric.columns,
        index=numeric.index,
    )
    print("scaled feature means (~0):\n", scaled.mean().round(3))
    print("scaled feature stds (~1):\n", scaled.std(ddof=0).round(3))


if __name__ == "__main__":
    main()
