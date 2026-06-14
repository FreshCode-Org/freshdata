"""Cleaning a large synthetic dataset, with timing.

Demonstrates throughput on a 100k-row frame and reusing a configured Cleaner.
Run:

    python examples/06_large_dataset.py
"""

import time

import numpy as np
import pandas as pd

import freshdata as fd


def make_data(n: int = 100_000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "id": range(n),
            "value": rng.normal(0, 1, n),
            "category": rng.choice(["x", "y", "z"], n),
            "amount": rng.gamma(2.0, 50.0, n),
        }
    )
    df.loc[rng.choice(n, n // 20, replace=False), "value"] = np.nan
    # Add some duplicate rows
    return pd.concat([df, df.iloc[:500]], ignore_index=True)


def main() -> None:
    df = make_data()
    print(f"Input: {len(df):,} rows x {df.shape[1]} columns")

    start = time.perf_counter()
    cleaned, report = fd.clean(df, id_columns=("id",), return_report=True)
    elapsed = time.perf_counter() - start

    print(f"Cleaned in {elapsed:.2f}s")
    print(f"Rows: {len(df):,} -> {len(cleaned):,}")
    print(report.summary().splitlines()[0])


if __name__ == "__main__":
    main()
