---
title: How to automatically clean a pandas DataFrame
description: Learn how to clean messy pandas DataFrames in Python with one call using FreshData.
keywords: automatic pandas data cleaning, clean dataframe python, pandas clean data, fix dirty pandas dataframe
---

# How to automatically clean a pandas DataFrame

* **Search Intent**: Developers searching for an automated, reliable way to clean messy pandas DataFrames (whitespace, sentinel strings, mixed dtypes, dirty headers) without writing dozens of brittle custom helper functions.
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

When loading raw tabular data into pandas from CSV, Excel, or SQL databases, developers face a barrage of common data defects:
1. Column names have irregular spacing, mixed casing, and special characters (`" Full Name "`, `"Order ID#"`).
2. Missing values are masked as string sentinels (`"N/A"`, `"null"`, `"-"`, `"missing"`).
3. Text cells contain accidental leading or trailing whitespace.
4. Numeric and date columns are parsed as generic `object` dtypes due to formatting artifacts (`"$1,200.50"`).

Manually writing separate `str.strip()`, `replace()`, `to_numeric()`, and `rename()` steps is tedious, difficult to maintain, and prone to silent data bugs.

---

## Code

```bash
pip install freshdata-cleaner
```

```python
import pandas as pd
import freshdata as fd

# 1. Create a messy pandas DataFrame
raw_df = pd.DataFrame({
    " Customer ID ": ["C-101", "C-102", "C-103", "C-104"],
    "Full Name": ["  Alice Smith  ", "Bob Jones", "Charlie Brown", "Diana Prince"],
    "Signup Date": ["2023-01-15", "2023/02/20", "N/A", "2023-04-10"],
    "Annual Spend": ["$1,200.50", "$450.00", "missing", "$890.25"],
    "Status": ["active", "pending", "-", "active"],
})

# 2. Clean in one line with explainable report
cleaned_df, report = fd.clean(
    raw_df,
    preserve_columns=[" Customer ID "],
    return_report=True,
)

print("Cleaned DataFrame:")
print(cleaned_df)

print("\nAudit Summary:")
print(report.summary())
```

---

## Output

```text
Cleaned DataFrame:
  customer_id     full_name signup_date  annual_spend   status
0       C-101   Alice Smith  2023-01-15       1200.50   active
1       C-102     Bob Jones  2023-02-20        450.00  pending
2       C-103 Charlie Brown         NaN           NaN      NaN
3       C-104  Diana Prince  2023-04-10        890.25   active

Audit Summary:
freshdata clean report
  rows:    4 -> 4 (+0)
  columns: 5 -> 5 (+0)
  missing: 0 -> 3 cell(s)
  actions (6):
    - [column_names] normalized 5 headers to snake_case
    - [strip_whitespace] trimmed surrounding whitespace in 'full_name'
    - [normalize_sentinels] replaced sentinels ("N/A", "missing", "-") with missing
    - [fix_dtypes] converted 'annual_spend' from object to Float64
    - [fix_dtypes] parsed 'signup_date' to datetime64[ns]
    - [missing] preserved missing values for manual inspection
```

---

## Explanation

`fd.clean(df)` executes an ordered, non-destructive cleaning pipeline:
1. **Representation Repair**: Standardizes column names into consistent `snake_case`, strips whitespace from text cells, and detects recognized sentinel values (`"N/A"`, `"-"`, `"null"`), converting them to true `np.nan`.
2. **Type Inference**: Scans text columns for currency symbols, numeric strings, and ISO dates, converting them to proper numeric (`Float64`, `Int64`) or `datetime64[ns]` dtypes.
3. **Safety Protection**: Notice how `customer_id` is protected against alteration. Every single modification is logged in the `CleanReport` with a human-readable description and confidence score.

---

## Next Steps

* Learn more about [Decision-Preserving Workflows](../decision-workflow.md).
* Explore the [GitHub Repository](https://github.com/FreshCode-Org/freshdata).
