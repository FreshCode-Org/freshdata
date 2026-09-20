"""examples/integrations/polars_workflow.py — Native Polars Cleaning Workflow.

Problem:
--------
Polars is rapidly becoming the standard dataframe engine for high-performance
data analytics and feature engineering. However, Polars users still face messy
CSV/Parquet ingestion issues: un-normalized sentinels ("N/A", "null", "-"),
dirty column headers, and unparsed dtypes. Converting back and forth to pandas
manually defeats the developer ergonomics.

Installation:
-------------
pip install "freshdata-cleaner[polars]"
"""

from __future__ import annotations

import polars as pl

import freshdata as fd


def main() -> None:
    print("=== Integration: Native Polars Cleaning Workflow ===")

    # 1. Create a native Polars DataFrame with typical dirty inputs
    pl_df = pl.DataFrame({
        "User ID": ["P-101", "P-102", "P-103", "P-104", "P-105"],
        "City": [" New York ", "London", "N/A", "Tokyo", "Berlin "],
        "Latency_ms": ["45.2", "120.5", "null", "85.0", "-"],
        "Errors": [0, 1, 0, 0, 5],
    })

    print("\n[1] Input Polars DataFrame:")
    print(pl_df)
    print(f"Type: {type(pl_df)}")

    # 2. Clean directly using FreshData (preserves Polars DataFrame type)
    cleaned_pl = fd.clean(
        pl_df,
        preserve_columns=["User ID"],
    )

    print("\n[2] Cleaned Polars DataFrame:")
    print(cleaned_pl)
    print(f"Type: {type(cleaned_pl)}")
    assert isinstance(cleaned_pl, pl.DataFrame), "Output must be a Polars DataFrame"

    # 3. Downstream native Polars query expression
    print("\n[3] Downstream Polars Query Expression:")
    result = (
        cleaned_pl
        .filter(pl.col("latency_ms").is_not_null())
        .select([
            pl.col("city"),
            pl.col("latency_ms"),
            (pl.col("latency_ms") * 1.1).alias("adjusted_latency"),
        ])
    )
    print(result)


if __name__ == "__main__":
    main()
