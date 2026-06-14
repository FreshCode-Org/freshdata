---
title: Quickstart
description: >-
  Get started cleaning pandas DataFrames with freshdata — clean in one call,
  read the audit report, and preview the engine's decisions before applying them.
keywords: pandas clean dataframe, quickstart data cleaning python, automated preprocessing tutorial
---

# Quickstart

## Clean a DataFrame

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("messy_export.csv")

cleaned = fd.clean(df)   # sensible, explainable defaults
```

`fd.clean` returns a new, cleaned DataFrame and **never mutates your input**
(unless you pass `preserve_original=False`).

## Get the audit trail

```python
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())
```

```text
freshdata clean report
  rows:    525 -> 500 (-25)
  columns: 7 -> 6 (-1)
  missing: 421 -> 0 cell(s)
  time:    0.017s
  actions (7):
    - [drop_duplicates] dropped 25 duplicate row(s) (4.8% of rows, keep='first')
    - [missing] 'age': filled 12 missing value(s) with median (39.6846)
    - [outliers] 'amount': flagged 15 outlier(s) in new column 'amount_outlier'
  review (1):
    ? column 'mostly_gone' preserved at 60.0% missing in balanced mode
```

The report is also machine-readable:

```python
report.to_frame()   # one row per decision, as a DataFrame
report.to_dict()    # JSON-friendly for logging / dashboards
```

## Preview before cleaning

```python
# Read-only data-quality report
print(fd.profile(df))

# The exact plan clean() would run
print(fd.suggest_plan(df).summary())

# Compare strategies side by side
print(fd.compare_plans(df))
```

## Protect important columns

```python
cleaned = fd.clean(
    df,
    target_column="churn",        # never modified (prevents leakage)
    id_columns=("customer_id",),  # never imputed
    preserve_columns=("notes",),  # never dropped
    return_report=True,
)
```

## Reuse a configured pipeline

```python
cleaner = fd.Cleaner(target_column="churn", strategy="balanced")
for path in paths:
    out = cleaner.clean(pd.read_csv(path))
    log.info(cleaner.report_.summary())
```

## Choose a strategy

| strategy | behavior |
|---|---|
| `"balanced"` *(default)* | accuracy-first; preserves high-missing columns, flags outliers |
| `"aggressive"` | maximal scrubbing: KNN imputation, column drops, winsorization |
| `"conservative"` | representation repair only (names, sentinels, dtypes, dupes) |

```python
fd.clean(df, strategy="aggressive")
```

Next: learn exactly how those decisions are made in the
[cleaning engine guide](cleaning-engine.md).
