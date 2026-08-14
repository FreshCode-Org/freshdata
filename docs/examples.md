---
title: Examples
description: >-
  Runnable freshdata examples — missing-value cleaning, outlier handling, feature
  normalization, data profiling, ML preprocessing pipelines, and CSV automation.
keywords: data cleaning examples, pandas preprocessing example, ml preprocessing pipeline, csv cleaning automation
---

# Examples

Every script in the [`examples/`](https://github.com/FreshCode-Org/freshdata/tree/main/examples)
directory is self-contained and runnable (`python examples/<name>.py`). The
[`notebooks/`](https://github.com/FreshCode-Org/freshdata/tree/main/notebooks)
directory has narrated Jupyter walkthroughs.

| Example | What it shows |
|---|---|
| `01_missing_values.py` | Smart, role-aware missing-value imputation |
| `02_outliers.py` | Outlier detection and flagging vs removal |
| `03_normalization.py` | Feature normalization for ML |
| `04_profiling.py` | Read-only data profiling and EDA |
| `05_ml_pipeline.py` | End-to-end ML preprocessing with scikit-learn |
| `06_large_dataset.py` | Cleaning a large synthetic dataset, with timing |
| `07_pandas_integration.py` | Dropping freshdata into an existing pandas workflow |
| `09_pandera_recipe.py` | Validating with pandera before and after freshdata cleaning |
| `10_pyjanitor_interop.py` | Combining PyJanitor transforms with FreshData quality repair |
| `11_great_expectations_recipe.py` | Repair string-typed data, then run an optional Great Expectations checkpoint |

## Missing-value cleaning

```python
import pandas as pd
import freshdata as fd

df = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "age": [34, None, 41, None, 29],
    "segment": ["A", "B", None, "A", None],
})

cleaned, report = fd.clean(df, id_columns=("customer_id",), return_report=True)
print(report.summary())
```

## ML preprocessing pipeline

```python
import pandas as pd
import freshdata as fd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

raw = pd.read_csv("customers.csv")
clean_df, report = fd.clean(raw, target_column="churn", return_report=True)
assert not report.warnings

X = pd.get_dummies(clean_df.drop(columns="churn"))
y = clean_df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
model = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
print("accuracy:", model.score(X_te, y_te))
```

## CSV automation

```python
from pathlib import Path
import pandas as pd
import freshdata as fd

cleaner = fd.Cleaner(strategy="balanced")
for path in Path("inbox").glob("*.csv"):
    out = cleaner.clean(pd.read_csv(path))
    out.to_csv(Path("clean") / path.name, index=False)
    print(path.name, "→", cleaner.report_.summary().splitlines()[0])
```

## PyJanitor interoperability

FreshData and [PyJanitor](https://pyjanitor-devs.github.io/pyjanitor/) solve
different parts of a pandas workflow. Use PyJanitor for explicit reshaping and
method-style transformations; use FreshData for evidence-based quality
detection, conservative repair, and an auditable report.

The runnable [`10_pyjanitor_interop.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/10_pyjanitor_interop.py)
example demonstrates both useful orderings on one small inline DataFrame:

- **PyJanitor then FreshData:** normalize the input shape first, then detect and
  repair quality issues in the resulting columns.
- **FreshData then PyJanitor:** clean and record the quality decisions first,
  then add an explicit presentation transform to the cleaned result.

PyJanitor remains an optional dependency. FreshData 2.0 supports pandas 1.5–2.x;
install the compatible PyJanitor 0.31 line to run the example:

```bash
pip install "pyjanitor<0.32"
python examples/10_pyjanitor_interop.py
```

## Great Expectations: repair, then validate

FreshData and Great Expectations have complementary roles: FreshData repairs
representational problems and records the changes, while Great Expectations
checks the result against an explicit data contract. Great Expectations remains
an optional dependency:

```bash
pip install great-expectations
python examples/11_great_expectations_recipe.py
```

The recipe runs the same checkpoint before and after `fd.clean()`. The raw
currency and boolean strings fail the typed contract; after FreshData converts
them to `float64` and `bool`, the checkpoint passes. The checkpoint validates
the result but does not modify the DataFrame.
