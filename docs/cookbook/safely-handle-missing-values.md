---
title: How to safely handle missing values in Python
description: Safely impute missing data in pandas without corrupting IDs, target labels, or skewed distributions.
keywords: missing value cleaning python, safe fillna pandas, role-aware imputation, data cleaning missing values
---

# How to safely handle missing values in Python

* **Search Intent**: Developers looking to handle `NaN` and missing values in tabular data safely without blindly filling everything with mean or zero.
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

A common beginner mistake in data engineering is running a blanket imputation:

```python
# DANGEROUS: Blind imputation corrupts data
df.fillna(df.mean(), inplace=True)
```

This causes three catastrophic issues:
1. **Corrupting Identifiers**: Filling missing `user_id` values creates duplicate or synthetic primary keys that break database joins.
2. **Label Leakage**: Imputing target labels in training data teaches machine learning models to predict imputed assumptions rather than real patterns.
3. **Distorting Skewed Data**: Mean-imputing highly skewed financial metrics (e.g. annual salary or transaction amount) shifts variance and damages downstream statistics.

---

## Code

```python
import numpy as np
import pandas as pd
import freshdata as fd

# 1. Dataset with distinct column roles
df = pd.DataFrame({
    "account_id": [101, 102, np.nan, 104, 105],  # Identifier -> NEVER impute
    "age": [25.0, 30.0, np.nan, 45.0, 32.0],     # Normal numeric -> median
    "churn": [0, 1, 0, np.nan, 0],               # Target label -> NEVER impute
    "tier": ["Gold", "Silver", np.nan, "Gold", "Bronze"],  # Categorical -> mode / sentinel
})

# 2. Clean with role-aware missing handling
cleaned, report = fd.clean(
    df,
    target_column="churn",
    strategy="balanced",
    return_report=True,
)

print("Cleaned DataFrame:")
print(cleaned)
```

---

## Output

```text
Cleaned DataFrame:
   account_id   age  churn      tier  age_outlier
0       101.0  25.0    0.0      Gold        False
1       102.0  30.0    1.0    Silver        False
2         NaN  31.0    0.0   Missing        False
3       104.0  45.0    NaN      Gold        False
4       105.0  32.0    0.0    Bronze        False

freshdata clean report
  actions (4):
    - [missing] 'account_id': preserved 1 missing value(s) (identifier-like column)
    - [missing] 'churn': preserved 1 missing value(s) (target label column)
    - [missing] 'age': filled 1 missing value(s) with median (31.0)
    - [missing] 'tier': filled 1 missing value(s) with sentinel "Missing"
  warnings (1):
    ! target column 'churn' has 1 missing value(s); rows without a label usually need to be dropped manually
```

---

## Explanation

FreshData uses a **role-aware decision engine**:
* **Identifiers**: Columns with high uniqueness or naming patterns (`id`, `uuid`, `key`) are recognized as identifiers. Imputation is refused because inventing primary keys violates relational integrity.
* **Target Columns**: Columns specified via `target_column="churn"` are strictly protected against mutation to prevent data leakage.
* **Skew-Aware Numeric Imputation**: Symmetric distributions use the mean, while skewed or outlier-bearing features automatically select the median.
* **Categoricals**: If there is a dominant category, mode imputation is applied; otherwise, an explicit `"Missing"` category keeps data gaps transparent.

---

## Next Steps

* See [Flagship Example 02: Missing Values](https://github.com/FreshCode-Org/freshdata/blob/main/examples/02_missing_values.py).
* Learn about [FreshData Cleaning Engine Rules](../cleaning-engine.md).
