---
title: Choose and audit missing-value imputation
description: Compare simple and MissForest imputation, configure column rules, protect declared roles, and inspect actual repairs.
---

# Choose and audit missing-value imputation

Missing values need a decision: retain the gap, fill it from the observed
column, or estimate it from other features. An imputed value is an estimate,
not a recovered observation. Start with the simplest strategy that fits your
data and inspect its effects before using the result downstream.

## Choose a strategy

| `impute` value | Behavior | Tradeoff |
| --- | --- | --- |
| `"mean"` | Fill numeric columns with their observed mean. Non-numeric columns are skipped. | Sensitive to extreme values; repeated fills reduce variation. |
| `"median"` | Fill numeric columns with their observed median. Non-numeric columns are skipped. | More resistant to extreme values, but still a constant estimate. |
| `"mode"` | Fill from the most frequent observed value, including text columns. | Can reinforce the majority category; inspect ties and rare categories. |
| `"auto"` | Use median for numeric columns and mode for other columns. | A simple per-column rule, not model selection. |
| `"missforest"` | Use iterative random-forest regression/classification with other columns as predictors. | Requires optional ML dependencies and enough usable data; can fall back to simple imputation with an audit reason. |

Simple imputation uses only the observed values of the column being filled.
FreshData's MissForest-inspired implementation initializes gaps, fits a model
for each eligible column, and iterates up to `missforest_max_iter`, stopping
earlier if it converges. More complexity does not guarantee better estimates.
Compare choices on representative data using validation suited to your task.

`strategy` controls the broader cleaning engine; `impute` and `impute_strategy`
are explicit missing-value overrides:

| Combination | What to expect |
| --- | --- |
| `strategy="conservative"`, `impute=None` | Representation repairs; no decision-engine imputation. |
| `strategy="balanced"`, `impute=None` | Accuracy-first decisions based on column roles, missingness, and available evidence. Gaps may remain. |
| `strategy="aggressive"`, `impute=None` | More active filling, column drops, and capping where the engine's guards allow them. |
| Any strategy with an explicit `impute` choice | Apply that override to eligible columns; declared ID/target protection still applies. |
| Any strategy with `impute_strategy={...}` | Rules for the named columns override the global `impute` choice. |

The examples use `"conservative"` to make explicit imputation easy to inspect.
Other strategies can make additional decisions; none guarantees that every
gap will be filled. See the [cleaning engine](../cleaning-engine.md).

## Apply global and per-column rules

Fill age with its median and group with its mode. Deliberately request fills
for declared ID and target columns to verify that those requests are ignored
and audited. One age and one group are missing at `row-c`.

```python
import pandas as pd
import freshdata as fd

frame = pd.DataFrame(
    {
        "record_key": ["A", None, "C", "D", "E"],
        "age": [20.0, 30.0, None, 40.0, 90.0],
        "group": ["red", "blue", None, "red", "red"],
        "outcome": [1.0, 0.0, 1.0, None, 0.0],
    },
    index=["row-a", "row-b", "row-c", "row-d", "row-e"],
)

cleaned, report = fd.clean(
    frame,
    strategy="conservative",
    impute="median",                  # global numeric rule
    impute_strategy={
        "group": "mode",              # overrides the global rule
        "record_key": "mode",         # ignored: declared identifier
        "outcome": "mean",            # ignored: declared target
    },
    id_columns=("record_key",),
    target_column="outcome",
    fix_dtypes=False,
    drop_empty_rows=False,
    drop_empty_columns=False,
    return_report=True,
    verbose=False,
)

assert cleaned.loc["row-c", "age"] == 35.0
assert cleaned.loc["row-c", "group"] == "red"
pd.testing.assert_series_equal(cleaned["record_key"], frame["record_key"])
pd.testing.assert_series_equal(cleaned["outcome"], frame["outcome"])
print(cleaned[["age", "group"]])
print(report.warnings)
```

The warnings identify ignored rules for `record_key` and `outcome`. Declare
important roles explicitly: explicit simple imputation protects `id_columns`
and `target_column`, so do not rely on name heuristics to protect arbitrary
keys or labels. Context-protected columns also remain untouched. All-missing
columns have no observed fill value, and numeric rules do not apply to text;
those gaps can remain.

Per-column names resolve after column-name normalization. For `"Customer Age"`,
use the normalized name `"customer_age"` in `impute_strategy`. Configure these
options through `fd.clean` keyword arguments or `fd.CleanConfig`.

## Inspect the audit and the changed cells

Continue with `frame`, `cleaned`, and `report` from the previous example:

```python
for action in report.actions:
    if action.step == "impute":
        print(action.column, action.count, action.description, action.metadata)

# Actions report a column and affected count, not every cell's location.
# This example keeps the index, columns, and row order, so the frames align.
pd.testing.assert_index_equal(cleaned.index, frame.index)
assert all(column in cleaned.columns for column in frame.columns)
filled = frame.isna() & cleaned[frame.columns].notna()
repairs = [
    {
        "row_position": position,
        "row_label": frame.index[position],
        "column": column,
        "value": cleaned[column].iloc[position],
    }
    for column in frame.columns
    for position in range(len(frame))
    if filled[column].iloc[position]
]
print(repairs)
assert [(item["row_label"], item["column"]) for item in repairs] == [
    ("row-c", "age"),
    ("row-c", "group"),
]

# Persist the structured aggregate audit when you need a portable record.
audit_json = report.to_json()
```

The filled cells are age `35.0` and group `"red"` at `row-c`; ID and target gaps
remain. Imputation actions report one affected cell per fill and zero-count
notes for protected columns. `CleanReport` also captures warnings and overall
before/after missing counts. `columns_imputed` is a decision-engine summary;
use the actual `impute` actions for explicit overrides.

This comparison identifies missing-to-present repairs. If your plan drops or
reorders rows or renames columns, align the frames using a stable record
identity first. It is not a history of every non-missing value change. See
[audit automated cleaning](audit-automated-data-cleaning.md) and the
[audit schema](../audit-trail.md) for supported report fields.

## Try model-based imputation

Install the optional ML dependencies:

```bash
pip install "freshdata-cleaner[ml]"
```

This separate example has 80 rows and related numeric/categorical predictors.
It fills eight missing charges with a small forest and at most two iterations.
No identifier or target column is included in this model example.

```python
import pandas as pd
import freshdata as fd

training = pd.DataFrame(
    {
        "hours": [float(i % 12) for i in range(80)],
        "charges": [10 + 2 * (i % 12) + 0.5 * (i % 5) for i in range(80)],
        "kind": ["standard", "priority"] * 40,
    }
)
training.loc[training.index[::11], "charges"] = float("nan")

estimated, model_report = fd.clean(
    training,
    strategy="conservative",
    impute_strategy={"charges": "missforest"},
    missforest_n_estimators=10,
    missforest_max_iter=2,
    missforest_random_state=42,
    missforest_add_indicators=False,
    fix_dtypes=False,
    return_report=True,
    verbose=False,
)

assert estimated["charges"].notna().all()
for action in model_report.actions:
    if action.step == "impute" and action.column == "charges":
        print(action.count, action.metadata)
        assert action.count == 8
```

Inspect `selected_model_type`, `iterations`, `converged`, and `fallback_reason`
in action metadata. With ML dependencies installed, this example reports
`selected_model_type="regressor"` and `fallback_reason=None`. Reaching the
iteration limit is recorded separately from convergence. Numerical estimates
and model diagnostics can vary with dependency versions.

`missforest_min_rows_for_model` defaults to 50. Smaller inputs take a recorded
simple-imputation fallback. Use the earlier five-row `frame`:

```python
small, small_report = fd.clean(
    frame,
    strategy="conservative",
    impute_strategy={"age": "missforest"},
    id_columns=("record_key",),
    target_column="outcome",
    missforest_add_indicators=False,
    fix_dtypes=False,
    drop_empty_rows=False,
    drop_empty_columns=False,
    return_report=True,
    verbose=False,
)

assert small.loc["row-c", "age"] == 35.0
fallback_actions = [
    action for action in small_report.actions
    if action.step == "impute" and action.column == "age"
]
assert len(fallback_actions) == 1
print(fallback_actions[0].metadata["fallback_reason"])
assert "fewer rows" in fallback_actions[0].metadata["fallback_reason"]
```

Missing ML dependencies, insufficient predictors, and ineligible columns can
also lead to fallback or preservation; inspect the audit rather than assuming
a requested model ran. Declared IDs and targets are not imputed, but role
protection is not a train/test leakage guarantee. Cleaning learns from the
supplied frame; isolate training data and validate your downstream workflow
before estimating values in evaluation data.
