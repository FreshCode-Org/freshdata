# FreshData Community Launch Assets (Show HN, Reddit, PyData)

Launch copy and post drafts prepared for major technical developer communities.

---

## 1. Hacker News: "Show HN: FreshData — Automated, explainable data cleaning for pandas and Polars"

**Title**: `Show HN: FreshData – Automated, explainable data cleaning for pandas and Polars`  
**URL**: `https://github.com/FreshCode-Org/freshdata`

**Post Body**:

```text
Hi HN,

We built FreshData (https://github.com/FreshCode-Org/freshdata) because we were tired of writing the same dozens of fragile data-cleaning helper functions every time we ingested a CSV or SQL export into pandas or Polars.

Most existing data quality tools either profile data (telling you what's wrong without fixing it) or validate data against rigid rules (telling you that a test failed, but not repairing it).

FreshData aims to bridge that gap: one call cleans your DataFrame, but crucially, it explains what it changed and why.

A quick example:
```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())
```

What makes FreshData different:
1. Decision engine, not blind fillna: It infers column roles (identifiers, categoricals, numeric, targets). Identifiers (like customer_id) are never imputed, and target labels are locked to prevent ML leakage.
2. Explainability: Every cell modification outputs an Action record with a rationale, a risk level, and a confidence score. If an action is high-risk, it recommends human review rather than guessing silently.
3. Outliers are non-destructive: Outliers are flagged in a boolean column by default rather than silently dropping rows.
4. Native Polars support: Pass a Polars DataFrame and get a Polars DataFrame back with zero pandas boilerplate.

We enforce a 93% test coverage gate in CI and maintain a reproducible benchmark suite across 10k to 25M rows (detailed in benchmarks/README.md).

The PyPI distribution is `freshdata-cleaner` (due to namespace collisions with an older project), but imports as `import freshdata as fd`.

We'd love feedback on our decision engine heuristics and how it behaves on your messiest CSVs!
```

---

## 2. Reddit r/Python & r/dataengineering Launch Post

**Title**: `[P] FreshData – Automated, explainable data cleaning for pandas and Polars (with audit trails)`

**Post Body**:

```text
Hey everyone,

Whenever data pipelines break, it's rarely due to complex ML model errors — it's usually unhandled sentinel strings ("N/A", "missing", "-"), messy padding whitespace, unparsed currency symbols, or accidental duplicate rows.

We created FreshData 2.0 to automate tabular data hygiene safely:
GitHub: https://github.com/FreshCode-Org/freshdata
Docs: https://freshcode-org.github.io/freshdata/

### What it does:
- One call (`fd.clean(df)`) standardizes headers into snake_case, strips whitespace, parses dates/currency, and handles missingness.
- Never mutates primary keys or target labels.
- Dual-engine: Works seamlessly on both pandas and Polars DataFrames (`fd.clean(polars_df)` returns a Polars DataFrame).
- Produces an audit report you can serialize to JSON for compliance and data lineage.
- Includes declarative pipelines: `pipe = fd.pipeline().strip_whitespace().normalize_sentinels().run(df)`.

### How to try it:
```bash
pip install freshdata-cleaner
```

```python
import freshdata as fd
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())
```

We wrote a full cookbook with 7+ recipes for common data engineering workflows:
https://freshcode-org.github.io/freshdata/cookbook/

Feedback, benchmark PRs, and issue reports are very welcome!
```

---

## 3. Launch Checklist & Timing

| Step | Action | Timing |
|:---:|---|---|
| 1 | Ensure `main` branch is clean, CI is green, and docs are built | Day 0 (T-1h) |
| 2 | Post to Hacker News (*Show HN*) | Tuesday 08:00 AM ET / 05:00 AM PT |
| 3 | Post to Reddit (r/Python and r/dataengineering) | Tuesday 09:30 AM ET |
| 4 | Post announcement thread on Twitter/X and LinkedIn | Tuesday 10:00 AM ET |
| 5 | Actively monitor comments on HN and Reddit, responding with code citations | Hours 0 to 12 |
