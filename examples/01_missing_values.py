"""Smart, role-aware missing-value handling with freshdata.

freshdata chooses an imputation per column based on its inferred role and
missing ratio — and never imputes identifiers or targets. Run:

    python examples/01_missing_values.py
"""

import numpy as np
import pandas as pd

import freshdata as fd


def make_data(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),          # id — never imputed
            "age": rng.normal(40, 12, n),            # low-missing numeric
            "income": rng.normal(60_000, 15_000, n), # medium-missing numeric
            "segment": rng.choice(["A", "B", "C"], n),
            "notes": rng.choice(["", "vip", "late payer"], n),  # free text
        }
    )
    # Inject missingness
    df.loc[rng.choice(n, 8, replace=False), "age"] = np.nan        # ~4%
    df.loc[rng.choice(n, 40, replace=False), "income"] = np.nan    # ~20%
    df.loc[rng.choice(n, 30, replace=False), "segment"] = np.nan   # ~15%
    return df


def main() -> None:
    df = make_data()
    print(f"Before: {df.isna().sum().sum()} missing cells\n")

    cleaned, report = fd.clean(
        df,
        id_columns=("customer_id",),
        preserve_columns=("notes",),
        return_report=True,
    )

    print(report.summary())
    print(f"\nAfter: {cleaned.isna().sum().sum()} missing cells in protected/free-text only")


if __name__ == "__main__":
    main()
