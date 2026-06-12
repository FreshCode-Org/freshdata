# freshdata

**Fast, safe, automatic data cleaning for real-world tabular data.**

[![PyPI Version](https://img.shields.io/pypi/v/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![CI](https://github.com/JohnnyWilson-Portfolio/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyWilson-Portfolio/freshdata/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`freshdata` cleans messy CSV / Excel / SQL-export data in one call — and tells
you exactly what it did and *why*. It is not a `fillna` wrapper: a rule-based
decision engine profiles every column (missing ratio, dtype, skewness,
cardinality, inferred role) and chooses the right action per column, logging a
rationale, a risk level, and a confidence score for each one.

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")

cleaned = fd.clean(df)                             # one line
cleaned, report = fd.clean(df, return_report=True) # ... with a full audit trail
print(report.summary())
```

```text
freshdata clean report
  rows:    525 -> 500 (-25)
  columns: 7 -> 6 (-1)
  missing: 421 -> 0 cell(s)
  memory:  100.8 KB -> 89.2 KB
  time:    0.017s
  engine:  25 duplicate row(s) removed; 20 outlier(s) handled; dropped: mostly_gone; imputed: age, segment
  actions (7):
    - [fix_dtypes] 'mostly_gone': converted to Int64
    - [drop_duplicates] dropped 25 duplicate row(s) (4.8% of rows, keep='first')
    - [missing] 'age': filled 12 missing value(s) with median (39.6846)
    - [missing] 'segment': filled 90 missing value(s) with sentinel "Missing" ('Missing')
    - [missing] 'mostly_gone': dropped column (60.0% missing, high band)
    - [outliers] 'amount': capped 15 outlier(s), 3.0% of values (method=iqr, factor=1.5) to [-13.88, 121.39]
    - [outliers] 'age': capped 5 outlier(s), 1.0% of values (method=iqr, factor=1.5) to [20.72, 59.32]
  review (1):
    ? column 'mostly_gone' was dropped at 60.0% missing; pass preserve_columns=('mostly_gone',) to keep it
```

## Install

```bash
pip install freshdata-cleaner          # pandas + numpy only
pip install "freshdata-cleaner[ml]"    # + scikit-learn (KNN imputation, IsolationForest)
```

Requires Python ≥ 3.9 and pandas ≥ 1.5.

## How cleaning works

**Layer 1 — representation repair** (always on):

| order | step | what it does |
|---|---|---|
| 1 | `column_names` | snake_case names, deduplicate collisions (`"a", "a"` → `"a", "a_2"`) |
| 2 | `strip_whitespace` | trim surrounding whitespace in text cells (internal spacing kept) |
| 3 | `normalize_sentinels` | `"N/A"`, `"null"`, `"-"`, `""`, `"#REF!"`, … → missing |
| 4 | `drop_empty_columns` / `drop_empty_rows` | remove all-missing columns and rows |
| 5 | `fix_dtypes` | text → numeric (`"$1,234.56"` works) / datetime / boolean, validated |
| 6 | `drop_duplicates` | resolve duplicate rows (`duplicate_keep`: first/last/drop/aggregate) |

**Layer 2 — the decision engine** (`strategy="auto"`, the default) infers
each column's role — **id**, **target/label**, **datetime**, **free text**,
**categorical**, **numeric** — and applies explicit threshold rules.

### Missing values

| missing ratio | numeric | categorical | datetime |
|---|---|---|---|
| ≤ 5% (low) | mean if ~normal & no outliers, else median | mode if clear majority, else `"Unknown"` | ffill/bfill if time-ordered |
| ≤ 30% (medium) | KNN if correlated features + scikit-learn, else median | mode if dominant, else `"Missing"` | ffill/bfill if time-ordered |
| ≤ 60% (high) | kept (+ warning) only if preserved or missingness is informative; dropped otherwise | same | same |
| > 60% (extreme) | dropped unless preserved or a label | same | same |

Role gates run first: **targets are never modified**, **IDs are never
imputed**, **free text is never force-filled** — those columns are preserved
with the reason written into the report, so a remaining NaN is never silent.
A `<col>_was_missing` indicator column is added when the missingness itself
correlates with other features (configurable via `missing_indicators`).
On frames under 30 rows the ratios are too noisy: the engine preserves and
recommends manual review instead of guessing.

### Outliers

Detection: IQR fences (default), z-score, `outlier_method="auto"` (z-score
for ~normal columns, IQR for skewed), or `"isolation_forest"` (scikit-learn,
≥ 100 rows, falls back to IQR). The method, threshold, and action are always
logged.

Action (`outlier_action`): **`"cap"`** winsorizes to the fences (default —
keeps rows, tames magnitudes), `"remove"` drops rows, `"flag"` adds a boolean
`<col>_outlier` column, `None` detects and reports only. Outliers in ID and
target columns, `preserve_columns`, and domain-sensitive columns
(fraud/anomaly/risk-like names) are always preserved — there the extremes
usually *are* the signal. Heavy-tailed columns (> 15% outside the fences) are
flagged instead of capped, with a warning.

### Duplicates

Exact duplicates are removed by default (count and percentage reported).
Time-indexed frames never lose rows unless `allow_timeseries_duplicates=True`.
A duplicate ratio above `duplicate_threshold` (10%) raises a data-quality
warning. With `duplicate_subset`, `duplicate_keep="aggregate"` collapses each
group (numeric mean, first non-missing otherwise).

## Tuning the engine

```python
fd.clean(
    df,
    strategy="auto",                 # or "conservative": representation repair only
    missing_threshold_low=0.05,      # band edges for the missing-value rules
    missing_threshold_medium=0.30,
    missing_threshold_high=0.60,
    duplicate_threshold=0.10,        # warn above this duplicate ratio
    outlier_method="iqr",            # "zscore" | "auto" | "isolation_forest"
    outlier_action="cap",            # "remove" | "flag" | None
    target_column="churn",           # never modified
    preserve_columns=("notes",),     # never dropped
    id_columns=("ref",),             # never imputed
    preserve_original=True,          # False allows in-place memory reuse
    verbose=True,                    # one-line summary per clean
    return_report=True,
)
```

Explicit choices always override the engine: `impute="median"` /
`outliers="clip"` force simple uniform handling, and
`strategy="conservative"` restores the old opt-in behavior. Every option
lives on one frozen dataclass — `fd.CleanConfig` — and unknown names fail
immediately with a "did you mean" suggestion:

```python
config = fd.CleanConfig(duplicate_keep="aggregate", duplicate_subset=("order_id",))
fd.clean(df, config=config, outlier_action="flag")   # config + overrides

cleaner = fd.Cleaner(target_column="churn")          # reusable pipeline
for path in paths:
    out = cleaner.clean(pd.read_csv(path))
    log.info(cleaner.report_.summary())
```

## The report

`fd.clean(df, return_report=True)` returns `(cleaned_df, CleanReport)`:

- dataset shape, memory, and missing-cell counts before/after;
- one `Action` per decision — step, column, description, affected count,
  **rationale**, **risk level** (low/medium/high), **confidence score**;
- columns dropped / imputed / preserved, duplicates removed, outliers handled;
- `report.warnings` for risky decisions and `report.recommendations` for
  manual review;
- `report.summary()` (text), `report.to_frame()` (DataFrame),
  `report.to_dict()` (JSON-friendly).

If any NaN survives cleaning, the report says exactly why it was preserved.

## Profiling

`fd.profile(df)` inspects without changing anything — and because it runs the
*same* inference code as `clean`, its suggestions are a faithful preview:

```python
print(fd.profile(df))
```

```text
freshdata profile — 5 rows x 6 columns, 1.5 KB
  missing cells: 6 (20.0%)   duplicate rows: 1
  column        dtype    missing  issues
   First Name   object       20%  20.0% missing; 1 value(s) with surrounding whitespace; …
  AGE           object         -  1 sentinel value(s) meaning missing; would convert to Int64
  Joined Date   object         -  would convert to datetime64[ns]
  Active        object         -  would convert to bool
  Salary($)     object         -  would convert to float64
  empty         object      100%  100.0% missing; constant column
```

## What freshdata will not do

- Touch a target/label column, impute an identifier, or force-fill free text.
- Remove outliers blindly — capping is the default, and fraud/anomaly-style
  columns keep their extremes.
- Guess at fuzzy entity resolution ("Jon" vs "John").
- Parse ambiguous European decimal commas (`"1.234,56"`) — too risky to guess.
- Mutate your DataFrame (unless you pass `preserve_original=False`).

## API

| name | purpose |
|---|---|
| `fd.clean(df, *, return_report=False, config=None, **options)` | clean, optionally returning a `CleanReport` |
| `fd.profile(df, *, config=None, **options)` | read-only inspection with actionable issues |
| `fd.Cleaner(config=None, **options)` | reusable configured pipeline (`.clean()`, `.report_`) |
| `fd.CleanConfig` | frozen dataclass holding every option |
| `fd.CleanReport` / `fd.Action` | audit trail with rationale/risk/confidence |
| `fd.Profile` / `fd.ColumnProfile` | profiling results |

## Development

```bash
git clone https://github.com/JohnnyWilson-Portfolio/freshdata
cd freshdata
pip install -e ".[dev,ml]"
pytest
ruff check src tests
```

Benchmarks live in `benchmarks/bench.py` (`python benchmarks/bench.py`).

## License

MIT — see [LICENSE](LICENSE).
