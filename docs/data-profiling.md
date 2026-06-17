---
title: Data profiling
description: >-
  Inspect data quality before cleaning with freshdata's profiler — missingness,
  dtype issues, duplicates, and a faithful dry-run cleaning plan for EDA.
keywords: data profiling python, pandas eda, data quality report, exploratory data analysis
---

# Data profiling

`fd.profile(df)` inspects a DataFrame **without changing anything**. Because it
runs the *same* inference code as `clean`, its suggestions are a faithful preview
of what cleaning would do — ideal for exploratory data analysis (EDA) and
data-quality checks.

```python
import freshdata as fd

print(fd.profile(df))
```

```text
freshdata profile — 5 rows x 6 columns, 1.5 KB
  missing cells: 6 (20.0%)   duplicate rows: 1
  column        dtype    missing  issues
   First Name   object       20%  20.0% missing; 1 value(s) with surrounding whitespace; 1 sentinel value(s) meaning missing
  AGE           object         -  1 sentinel value(s) meaning missing; would convert to Int64
  Joined Date   object         -  1 sentinel value(s) meaning missing; would convert to datetime64[ns]
  Active        object         -  would convert to bool
  Salary($)     object         -  1 sentinel value(s) meaning missing; would convert to float64
  empty         object      100%  100.0% missing; constant column
```

## Attach a dry-run plan

```python
profile = fd.profile(df, include_plan=True)
print(profile.plan.summary())   # primary model chosen per column
```

## Programmatic access

```python
profile = fd.profile(df)
profile.to_dict()         # JSON-friendly summary
for col in profile.columns:
    print(col.name, col.dtype, col.missing_pct, col.issues)
```

## Inferring column roles

```python
print(fd.infer_roles(df))   # id / target / datetime / text / categorical / numeric
```

## Explaining a clean

```python
explanation = fd.explain_clean(df, strategy="balanced")
print(explanation.summary())
print(explanation.roles)
```

See the [API reference](api-reference.md) for `Profile` and `ColumnProfile`
fields.

## Compare with ydata-profiling

`fd.profile(df)` gives a **repair-focused** preview: it tells you what cleaning
actions would likely change your DataFrame, while leaving the data untouched.

`ydata_profiling.ProfileReport` (optional dependency) is useful for a fuller
EDA-style report with charts and correlation views.

```python
import pandas as pd
import freshdata as fd

sample = pd.DataFrame(
    {
        "customer_id": [1, 2, None, 4, 5],
        "revenue": [120.0, -999.0, 340.0, None, 560.0],
        "signup_date": ["2026-01-05", "2026/03/12", "invalid", "2026-06-01", None],
        "churned": ["yes", "no", "yes", None, "no"],
    }
)

print("Freshdata profile preview:")
fresh_profile = fd.profile(sample)
print(fresh_profile)
print("Freshdata plan summary:")
print(fresh_profile.plan.summary())
```

```python
# Optional: ydata-profiling for richer exploratory context.
# Install only if your project already uses it.
# pip install ydata-profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(sample, title="raw-data quick check", minimal=True)
profile.to_file("quickstart_profile.html")
```

### When to use which

- Use `fd.profile` when you need a lightweight, dry-run signal for **what
  `freshdata` would clean**.
- Use `ydata_profiling` when you need deeper EDA artifacts (distribution and
  correlation views) before deciding a cleaning strategy.
- You can run both on the same raw frame: profile first for a repair plan, then use
  a full profiling report for additional manual exploration.
