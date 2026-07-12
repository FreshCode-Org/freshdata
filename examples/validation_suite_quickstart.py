"""Declarative validation with fd.ValidationSuite — validate, never mutate.

Run: python examples/validation_suite_quickstart.py
"""

from __future__ import annotations

import pandas as pd

import freshdata as fd

df = pd.DataFrame(
    {
        "customer_id": ["c1", "c2", "c3", "c3"],  # duplicate id
        "age": [34, -2, 51, 29],  # negative age
        "email": ["a@x.com", "bad-email", "c@z.net", "d@w.io"],
        "signup": pd.to_datetime(["2021-03-01", "2022-01-15", "2020-07-01", "2023-02-01"]),
        "last_seen": pd.to_datetime(["2022-03-01", "2021-06-15", "2024-01-01", "2024-06-01"]),
    }
)

suite = fd.ValidationSuite(
    name="customers",
    rules=[
        fd.ColumnRule("customer_id", nullable=False, unique=True),
        fd.ColumnRule("age", min_value=0, max_value=120),
        # mostly=0.75: tolerate up to 25% bad emails as a warning, not a failure
        fd.ColumnRule("email", regex=r"[^@]+@[^@]+\.[^@]+", mostly=0.75),
        fd.ColumnRule("signup", min_datetime="2019-01-01", max_datetime="2026-01-01"),
    ],
    cross_column=[fd.CrossColumnRule("signup", "<=", "last_seen")],
    min_rows=1,
)

result = fd.validate(df, suite=suite)
print(f"passed: {result.passed} ({result.n_errors} errors, {result.n_warnings} warnings)")
for finding in result.report.findings:
    print(f"  [{finding.status}] {finding.check_id}: {finding.message}")

# Suites are versioned, serializable artifacts — commit them next to your data.
suite.save("customers_suite.json")
print("\nsaved to customers_suite.json; validate in CI with:")
print("  freshdata validate data.csv --suite customers_suite.json")

# Existing DataContract users migrate with one call:
contract = suite.to_contract()
same_suite = fd.ValidationSuite.from_contract(contract)
assert same_suite.to_contract().to_dict() == contract.to_dict()

# result.raise_if_failed() raises fd.ValidationError — CI-friendly.
