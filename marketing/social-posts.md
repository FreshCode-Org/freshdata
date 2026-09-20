# Technical Social Media Posts (Twitter/X, LinkedIn, Reddit)

High-signal, developer-focused social copy targeting common tabular data engineering pains.

---

## 1. Problem Post: The Silent Danger of Blanket `fillna()`

**Platform**: Twitter/X & LinkedIn  
**Hook**: Why blindly filling missing values in pandas corrupts your downstream machine learning models.

```text
The most dangerous line of code in data science:

`df.fillna(df.mean(), inplace=True)`

Here is what that single line quietly does to your dataset:
1. Imputes customer_id with a fake float ID, fabricating ghost users and corrupting table joins.
2. Fills missing ground-truth labels in your training target, creating synthetic targets your model can't learn from.
3. Distorts variance on skewed financial columns.

In FreshData, cleaning is role-aware:
- Identifiers are preserved (0% mutation).
- Targets are protected from leakage.
- Missing values are imputed only when statistically safe, with risk & confidence logged.

Try it in one line:
pip install freshdata-cleaner

import freshdata as fd
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())

https://github.com/FreshCode-Org/freshdata
```

---

## 2. Demonstration Post: Before & After in 30 Seconds

**Platform**: Twitter/X & LinkedIn  
**Hook**: What happens when you pass a genuinely messy DataFrame to FreshData?

```text
Raw CSV from production:
- Whitespace in headers: " Customer ID "
- Dirty sentinels: "N/A", "-", "missing"
- Unparsed currency: "$1,200.50"
- Outliers: age = -4

Before:
  Customer ID     Full Name   Age  Annual Spend
0       C-101   Alice Smith    29      $1200.50
1       C-102  Bob Jones       34           N/A
2       C-103  Charlie Brown   -4       missing

After `cleaned, report = fd.clean(df, return_report=True)`:
  customer_id      full_name  age  annual_spend  age_outlier
0       C-101    Alice Smith   29       1200.50        False
1       C-102      Bob Jones   34           NaN        False
2       C-103  Charlie Brown   -4           NaN         True

Plus an audit trail showing why each action occurred and what it refused to touch.

Check out the interactive recipes:
https://github.com/FreshCode-Org/freshdata/tree/main/examples
```

---

## 3. Engineering Post: Why FreshData Refuses to Change Certain Data

**Platform**: LinkedIn & Reddit r/datascience  
**Hook**: The mark of a reliable automated data cleaner isn't what it cleans — it's what it refuses to touch.

```text
When building an automated cleaner for tabular data, the hardest engineering problem isn't imputation or regex parsing.

It's knowing when NOT to touch a cell.

If an auto-cleaner tries to be "helpful" by force-imputing every NaN, it introduces silent data corruption. In FreshData 2.0, we enforce invariant boundaries:

1. Identifier columns (UUIDs, customer keys): Never imputed. Fabricating a primary key is worse than leaving it null.
2. Target labels: Strictly protected against imputation to prevent leakage in ML pipelines.
3. Outliers: Flagged in a boolean `_outlier` column by default rather than silently dropped.
4. Ambiguous sentinels: Normalized to NaN and routed to `report.recommendations` for human review.

Every action outputs an Action dataclass with `risk` (low/medium/high) and `confidence` (0.0 to 1.0).

Read our full trust invariant spec:
https://freshcode-org.github.io/freshdata/trust-claims/
```

---

## 4. Ecosystem Post: Polars Support

**Platform**: Twitter/X & Reddit r/Python  
**Hook**: Automated data cleaning for Polars without pandas boilerplate.

```text
Using Polars for data engineering? 

You can now run automated, explainable data cleaning natively:

import polars as pl
import freshdata as fd

pl_df = pl.read_csv("messy_telemetry.csv")

# Polars in -> Polars out
clean_pl = fd.clean(pl_df)
assert isinstance(clean_pl, pl.DataFrame)

- Strips whitespace
- Standardizes headers to snake_case
- Normalizes "N/A", "null", and "-" sentinels
- Flags outliers in a native boolean column

Install with Polars extra:
pip install "freshdata-cleaner[polars]"

Repo: https://github.com/FreshCode-Org/freshdata
```
