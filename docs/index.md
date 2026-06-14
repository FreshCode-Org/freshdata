---
title: freshdata — automated, explainable DataFrame cleaning for Python
description: >-
  freshdata is an automated data-cleaning library for Python and pandas. Smart
  missing-value handling, outlier detection, duplicate resolution, dtype repair,
  and AI-ready preprocessing in one explainable call.
keywords: automated dataframe cleaning, pandas preprocessing, data cleaning python, missing value handling, machine learning preprocessing, data quality automation
---

# freshdata

**Automated DataFrame cleaning for pandas — explainable, safe, and production-ready.**

`freshdata` turns a messy CSV, Excel, or SQL export into analysis- and ML-ready
data in a single call — and tells you exactly what it changed and **why**.

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())
```

It is **not** a `fillna` wrapper. A rule-based decision engine profiles every
column — missing ratio, dtype, skewness, cardinality, inferred role — and chooses
the right action per column, logging a rationale, a risk level, and a confidence
score for each decision.

## Why freshdata

- **Automated DataFrame cleaning** in one call: missing values, outliers,
  duplicates, dtype repair, and column-name normalization.
- **Explainable** — every decision is logged; if a `NaN` survives, the report
  says why.
- **Safe** — never imputes an identifier, modifies a target/label column,
  force-fills free text, or removes outliers blindly.
- **AI-ready preprocessing** — leakage-aware, typed output for scikit-learn,
  XGBoost, and any ML pipeline.
- **pandas-first, Polars-optional**, fully typed, 800+ tests, 95%+ coverage.

## Install

```bash
pip install freshdata-cleaner
```

See [Installation](installation.md) for optional extras (`ml`, `enterprise`, `all`).

## Next steps

<div class="grid cards" markdown>

- :material-rocket-launch: **[Quickstart](quickstart.md)** — clean your first DataFrame.
- :material-cog: **[Cleaning engine](cleaning-engine.md)** — how decisions are made.
- :material-chart-box: **[Data profiling](data-profiling.md)** — inspect before you clean.
- :material-book-open-variant: **[API reference](api-reference.md)** — every function and class.
- :material-flask: **[Examples](examples.md)** — runnable end-to-end recipes.
- :material-help-circle: **[FAQ](faq.md)** — common questions answered.

</div>
