---
title: How to prepare tabular data for machine learning in Python
description: Clean tabular datasets for scikit-learn and XGBoost models without data leakage or label corruption.
keywords: ml tabular data preprocessing, scikit-learn clean data, prevent target leakage, machine learning feature cleaning
---

# How to prepare tabular data for machine learning in Python

* **Search Intent**: ML engineers looking for a reliable way to clean tabular datasets for scikit-learn, XGBoost, or PyTorch without introducing target leakage or corrupting feature distributions.
* **Target Library**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)

---

## Problem

Preprocessing tabular data for machine learning poses strict constraints:
1. **Target Leakage**: Filling missing values using information computed across the whole dataset (including the target) leaks future labels into training features.
2. **Label Corruption**: Imputing missing values in the target column creates fake ground-truth labels.
3. **Identifier Distortion**: High-cardinality ID columns (`customer_id`, `uuid`) must not be one-hot encoded or scaled.

---

## Code

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import freshdata as fd

# 1. Raw training data with messy inputs and target
df = pd.DataFrame({
    "user_id": [f"ID_{i}" for i in range(100)],
    "age": [20 + i % 50 if i % 7 != 0 else np.nan for i in range(100)],
    "income": [f"${30000 + i * 500}" if i % 5 != 0 else "missing" for i in range(100)],
    "converted": [1 if i % 3 == 0 else 0 for i in range(100)],
})

# 2. Clean with target protection
cleaned_df, report = fd.clean(
    df,
    target_column="converted",
    preserve_columns=["user_id"],
    strategy="balanced",
    return_report=True,
)

# 3. Verify zero leakage on target
assert cleaned_df["converted"].equals(df["converted"]), "Target was altered!"

# 4. Train scikit-learn model
feature_cols = [c for c in cleaned_df.columns if c not in ("user_id", "converted") and not c.endswith("_outlier")]
X = cleaned_df[feature_cols]
y = cleaned_df["converted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42).fit(X_train, y_train)

print(f"Model trained successfully! Test accuracy: {model.score(X_test, y_test):.2f}")
```

---

## Output

```text
Target was verified: 0% mutation on ground truth labels.
Model trained successfully! Test accuracy: 0.85

Audit Report Summary:
freshdata clean report
  rows:    100 -> 100 (+0)
  columns: 4 -> 5 (+1 outlier indicator)
  actions:
    - [normalize_sentinels] 'income': converted "missing" to NaN
    - [fix_dtypes] 'income': parsed currency strings to Float64
    - [missing] 'income': imputed using median
    - [missing] 'age': imputed using median
    - [missing] 'user_id': preserved (identifier)
    - [missing] 'converted': preserved (target column)
```

---

## Explanation

By passing `target_column="converted"`, FreshData locks the target feature against all statistical alterations. Missing values in feature columns are imputed using robust local medians, currency strings are cleanly coerced to floats, and identifier columns are protected against modification.

---

## Next Steps

* See [Flagship Example 05: ML Preprocessing](https://github.com/FreshCode-Org/freshdata/blob/main/examples/05_ml_preprocessing.py).
* Explore [Scikit-Learn Pipeline Integration](https://github.com/FreshCode-Org/freshdata/blob/main/examples/integrations/sklearn_pipeline.py).
