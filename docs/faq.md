---
title: FAQ
description: >-
  Frequently asked questions about freshdata — how it differs from pandas
  fillna, sweetviz, and Great Expectations, ML preprocessing, Polars, and more.
keywords: freshdata faq, pandas fillna alternative, automated data cleaning questions, data cleaning vs profiling
---

# Frequently asked questions

## What is freshdata?

`freshdata` is an **automated, explainable data-cleaning library for Python and
pandas**. A single call — `fd.clean(df)` — handles missing values, outliers,
duplicates, dtype repair, and column-name normalization, and logs a rationale for
every decision.

## How is it different from `df.fillna()` / `df.dropna()`?

`fillna`/`dropna` apply one blunt rule to the whole frame and silently corrupt
data — imputing IDs, leaking targets, or deleting meaningful outliers.
`freshdata` infers each column's role and picks the right action per column, then
**explains it**. If a `NaN` survives, the report tells you exactly why.

## How is it different from sweetviz / ydata-profiling?

Profiling tools *describe* your data; they don't change it. `freshdata` both
profiles (`fd.profile`) and **cleans** (`fd.clean`) using the same inference, so
the preview matches the result.

## How is it different from Great Expectations?

Great Expectations *validates* data against expectations but doesn't repair it.
`freshdata` repairs data and (in the enterprise layer) can also gate on a Data
Trust Score. They are complementary.

## Does it work for machine-learning preprocessing?

Yes. `freshdata` produces leakage-aware, typed output ready for scikit-learn,
XGBoost, and other ML libraries. Pass `target_column=` so the label is never
modified, and `id_columns=` so identifiers are never imputed.

## Will it mutate my DataFrame?

No. `fd.clean` returns a new DataFrame and leaves the original untouched, unless
you explicitly pass `preserve_original=False` to reuse memory.

## Does it support Polars?

Yes — install `pip install "freshdata-cleaner[polars]"` and pass a Polars DataFrame to
`fd.clean`; you get a Polars DataFrame back.

## How do I see what it changed?

```python
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())   # human-readable
report.to_frame()         # DataFrame
report.to_dict()          # JSON
```

## How do I preview decisions without cleaning?

Use `fd.profile(df)`, `fd.suggest_plan(df)`, or `fd.compare_plans(df)`.

## Is it production-ready?

`freshdata` is typed (`py.typed`), has 800+ tests with 95%+ coverage, runs on
Python 3.9–3.13, and is validated against 50 real public datasets in CI.

## How do I report a bug or request a feature?

Open an issue on [GitHub](https://github.com/FreshCode-Org/freshdata/issues). For
security issues, see [SECURITY.md](https://github.com/FreshCode-Org/freshdata/blob/main/SECURITY.md).
