"""Dropping freshdata into an existing pandas workflow.

freshdata returns a normal DataFrame and never mutates the input, so it slots
into any pandas pipeline. Run:

    python examples/07_pandas_integration.py
"""

import pandas as pd

import freshdata as fd


def main() -> None:
    df = pd.DataFrame(
        {
            "Order ID": ["1001", "1002", "1002", "1003"],
            "Amount($)": ["$19.99", "$5.00", "$5.00", "n/a"],
            "Status": [" shipped ", "PENDING", "PENDING", "shipped"],
        }
    )

    # Original is untouched
    cleaned = fd.clean(df)
    assert df["Status"].tolist() == [" shipped ", "PENDING", "PENDING", "shipped"]

    # Continue with ordinary pandas on the cleaned frame
    result = (
        cleaned.groupby("status", dropna=False)["amount"]
        .agg(["count", "sum"])
        .reset_index()
    )
    print("Cleaned columns:", list(cleaned.columns))
    print(result)


if __name__ == "__main__":
    main()
