"""Recipe: validate a messy DataFrame with pandera, repair it with freshdata,
then validate again to see what's fixed and what still needs manual repair.
"""

import pandas as pd
import pandera.pandas as pa

import freshdata as fd


def main() -> None:
    df = pd.DataFrame(
        {
            "Age": [25, 41, None, 33],
            "Amount($)": ["$19.99", "$5.00", "$8.25", "n/a"],
            "Status": [" shipped ", "PENDING", "shipped", "pending"],
        }
    )

    raw_schema = pa.DataFrameSchema(
        {
            "Age": pa.Column(float, nullable=False),
            "Amount($)": pa.Column(float, nullable=False),
            "Status": pa.Column(str, nullable=False),
        }
    )

    print("=== Step 1: validate the raw frame ===")
    try:
        raw_schema.validate(df)
    except pa.errors.SchemaError as exc:
        print(f"Validation failed, as expected:\n{exc}\n")

    cleaned = fd.clean(df)

    clean_schema = pa.DataFrameSchema(
        {
            "age": pa.Column(float, nullable=False),
            "amount": pa.Column(float, nullable=False),
            "status": pa.Column(str, pa.Check.isin(["shipped", "pending"]), nullable=False),
        }
    )

    print("=== Step 2: validate the freshdata-cleaned frame ===")
    try:
        clean_schema.validate(cleaned)
        print("Schema check passed after cleaning:")
        print(cleaned)
    except pa.errors.SchemaError as exc:
        print(f"Still failing after cleaning (freshdata renamed/typed the columns, "
              f"but doesn't impute nulls or normalize casing):\n{exc}\n")

    print("=== Step 3: repair the remaining issues, then re-validate ===")
    repaired = cleaned.copy()
    repaired["age"] = repaired["age"].fillna(repaired["age"].median())
    repaired["amount"] = repaired["amount"].fillna(repaired["amount"].median())
    repaired["status"] = repaired["status"].str.lower()

    clean_schema.validate(repaired)
    print("Schema check passed after repair:")
    print(repaired)


if __name__ == "__main__":
    main()
