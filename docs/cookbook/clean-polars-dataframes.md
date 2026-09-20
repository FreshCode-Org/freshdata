---
title: How to clean Polars DataFrames in Python
description: Clean Polars DataFrames natively with FreshData — Polars in, Polars out with zero boilerplate.
keywords: polars clean dataframe, polars data cleaning, polars data quality, polars missing values
---

# How to clean Polars DataFrames in Python

* **Search Intent**: Polars developers looking for an automated data cleaning solution that accepts Polars DataFrames and returns Polars DataFrames without manual conversion boilerplate.
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

Polars is renowned for lightning-fast performance and memory efficiency. However, when ingesting raw data from external CSV or Parquet files, Polars users still encounter messy real-world artifacts:
1. Sentinel strings (`"N/A"`, `"-"`, `"null"`) that prevent columns from casting to numeric or date types.
2. Accidental surrounding whitespace in text columns.
3. Inconsistent column casing (`"User Name"`, `"Order_Date"`).

Converting Polars DataFrames to pandas just to clean them is slow, consumes extra memory, and breaks pipeline type signatures.

---

## Code

```bash
pip install "freshdata-cleaner[polars]"
```

```python
import polars as pl
import freshdata as fd

# 1. Create a native Polars DataFrame
pl_df = pl.DataFrame({
    " User ID ": ["P-01", "P-02", "P-03", "P-04"],
    "City": [" New York ", "London", "N/A", "Tokyo "],
    "Score": ["92.5", "88.0", "null", "95.2"],
})

# 2. Clean directly with FreshData
cleaned_pl = fd.clean(
    pl_df,
    preserve_columns=[" User ID "],
)

print(cleaned_pl)
print(f"Output type: {type(cleaned_pl)}")
assert isinstance(cleaned_pl, pl.DataFrame), "Output must be a native Polars DataFrame!"
```

---

## Output

```text
shape: (4, 3)
┌─────────┬──────────┬───────┐
│ user_id ┆ city     ┆ score │
│ ---     ┆ ---      ┆ ---   │
│ str     ┆ str      ┆ f64   │
╞═════════╪══════════╪═══════╡
│ P-01    ┆ New York ┆ 92.5  │
│ P-02    ┆ London   ┆ 88.0  │
│ P-03    ┆ null     ┆ null  │
│ P-04    ┆ Tokyo    ┆ 95.2  │
└─────────┴──────────┴───────┘
Output type: <class 'polars.dataframe.frame.DataFrame'>
```

---

## Explanation

FreshData includes a dedicated **Polars adapter**:
* **Preserves Native Types**: Polars DataFrames passed into `fd.clean()` return native `polars.DataFrame` instances.
* **Zero Overhead Interchange**: Uses Arrow C Data interface and PyArrow for near-zero-copy data exchange.
* **Automatic Repair**: Column names are standardized to `snake_case`, whitespace is stripped from Polars String columns, sentinels (`"N/A"`, `"null"`) are converted to true nulls, and numeric strings are cast to proper float/integer types.

---

## Next Steps

* See [Polars Integration Recipe](https://github.com/FreshCode-Org/freshdata/blob/main/examples/integrations/polars_workflow.py).
* Learn about [Out-of-Core Execution Backends](../backends.md).
