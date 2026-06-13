# freshdata

**Fast, safe, automatic data cleaning for real-world tabular data.**

[![PyPI Version](https://img.shields.io/pypi/v/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
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
  engine:  25 duplicate row(s) removed; 20 outlier(s) flagged; imputed: age, segment
  actions (7):
    - [fix_dtypes] 'mostly_gone': converted to Int64
    - [drop_duplicates] dropped 25 duplicate row(s) (4.8% of rows, keep='first')
    - [missing] 'age': filled 12 missing value(s) with median (39.6846)
    - [missing] 'segment': filled 90 missing value(s) with sentinel "Missing" ('Missing')
    - [missing] 'mostly_gone': preserved 300 missing value(s)
    - [outliers] 'amount': flagged 15 outlier(s), 3.0% of values (method=iqr, factor=1.5) in new column 'amount_outlier'
    - [outliers] 'age': flagged 5 outlier(s), 1.0% of values (method=iqr, factor=1.5) in new column 'age_outlier'
  review (1):
    ? column 'mostly_gone' preserved at 60.0% missing in balanced mode
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

**Layer 2 — the decision engine** (`strategy="balanced"`, the default) infers
each column's role — **id**, **target/label**, **datetime**, **free text**,
**categorical**, **numeric** — and applies explicit threshold rules.
Use `strategy="aggressive"` for v0.2-style scrubbing (KNN imputation, column
drops, winsorization). `strategy="auto"` is deprecated (alias for
`"aggressive"`).

### Missing values (balanced default)

| missing ratio | numeric | categorical | datetime |
|---|---|---|---|
| ≤ 5% (low) | mean if ~normal & no outliers, else median | mode if clear majority, else `"Unknown"` | ffill/bfill if time-ordered |
| > 5% and ≤ 30% (medium) | median (KNN only in aggressive mode) | mode if dominant, else `"Missing"` | ffill/bfill if time-ordered |
| > 30% (high/extreme) | **preserved** + warning (balanced); dropped in aggressive unless preserved/informative | same | same |

**Aggressive** mode additionally: KNN imputation for correlated numerics,
column drops for high/extreme missingness without informative signal.

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

Action (`outlier_action`): in **balanced** mode the default `"cap"` is
converted to **`"flag"`** (adds a boolean `<col>_outlier` column). Explicit
`"remove"` still drops rows. In **aggressive** mode, `"cap"` winsorizes to
the fences. `None` detects and reports only. Outliers in ID and target
columns, `preserve_columns`, and domain-sensitive columns (AQI, pollutants,
fraud/anomaly/risk-like names) are always preserved — there the extremes
usually *are* the signal. Heavy-tailed columns (> 15% outside the fences) are
flagged instead of capped.

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
    strategy="balanced",             # "aggressive" | "conservative"
    missing_threshold_low=0.05,      # band edges for the missing-value rules
    missing_threshold_medium=0.30,
    missing_threshold_high=0.60,
    duplicate_threshold=0.10,        # warn above this duplicate ratio
    outlier_method="iqr",            # "zscore" | "auto" | "isolation_forest"
    outlier_action="cap",            # balanced converts cap→flag; "remove" | None
    target_column="churn",           # never modified
    preserve_columns=("notes",),     # never dropped
    id_columns=("ref",),             # never imputed
    preserve_original=True,          # False allows in-place memory reuse
    verbose=True,                    # one-line summary per clean
    return_report=True,
)

# Preview engine choices before cleaning
plan = fd.suggest_plan(df)
print(plan.summary())
fd.clean(df, config=plan.config)

# Compare strategies side-by-side
print(fd.compare_plans(df))
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
*same* inference code as `clean`, its suggestions are a faithful preview.
With `include_plan=True`, attach a dry-run cleaning plan:

```python
print(fd.profile(df))
profile = fd.profile(df, include_plan=True)
print(profile.plan.summary())   # primary model per column
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
| `fd.suggest_plan(df, *, config=None, **options)` | dry-run: primary + alternative models per column |
| `fd.compare_clean(df, *, strategies=...)` | side-by-side actual clean outcomes per strategy |
| `fd.compare_plans(df, *, strategies=..., include_metrics=False)` | side-by-side models across strategies |
| `fd.profile(df, *, include_plan=False, config=None, **options)` | read-only inspection with actionable issues |
| `fd.Cleaner(config=None, **options)` | reusable configured pipeline (`.clean()`, `.report_`) |
| `fd.CleanConfig` | frozen dataclass holding every option |
| `fd.CleanPlan` / `fd.ColumnPlan` | engine preview before cleaning |
| `fd.CleanReport` / `fd.Action` | audit trail with rationale/risk/confidence/model_id |
| `fd.Profile` / `fd.ColumnProfile` | profiling results |

## Migrating from 0.2.x

**Breaking:** the default strategy changed from `"auto"` to `"balanced"`.

| If you want… | Do this |
|---|---|
| Same behavior as freshdata 0.2 | `fd.clean(df, strategy="aggressive")` |
| Accuracy-first cleaning (recommended) | `fd.clean(df)` — new default |
| Representation repair only | `fd.clean(df, strategy="conservative")` |

`strategy="auto"` still works but emits a `DeprecationWarning` (alias for
`"aggressive"`). Other notable 0.3 changes:

- High-missing columns are **preserved** in balanced mode (not dropped).
- Outliers are **flagged** by default in balanced mode (not capped).
- KNN imputation runs only in aggressive mode.
- Target heuristics expanded (`aqi`, `*_bucket`, `score`, …).
- `Action.model_id` records which imputation/outlier model was chosen.
- `fd.suggest_plan()` / `fd.compare_plans()` / `fd.compare_clean()` preview and compare engine decisions.

## Validated scenarios

Every fixture in `tests/fixtures/` is run under `conservative`, `balanced`, and
`aggressive` strategies in CI. Use `fd.compare_clean(df)` to reproduce the
quality/efficiency matrix on your own data.

| Fixture | Rows | What it stress-tests |
|---|---|---|
| `aqi_sample` | 500 | Real AQI panel slice — targets, pollutants, outliers |
| `large_panel` | 3,000 | AQI-shaped panel at scale — perf + preserve rules |
| `sales_export` | 200 | CRM export — currency strings, whitespace, dupes |
| `survey_responses` | 150 | High missing categoricals, free-text `notes` |
| `sensor_timeseries` | 120 | Datetime readings, time-ordered fills |
| `fraud_signals` | 180 | Domain-sensitive scores — outliers preserved |
| `tiny_cohort` | 12 | Small frame gate — preserve, don't drop |
| `wide_sparse` | 200×20 | Sparse columns — balanced never drops |
| `duplicate_heavy` | 260 | ~30% duplicate rows — layer-1 dedup |
| `locale_numbers` | 100 | European decimals — must **not** auto-convert |
| `mixed_roles` | 100 | Misnamed target, free text, id-like columns |

### Online datasets (50 curated)

Fifty real public datasets are catalogued in [`tests/fixtures/online/registry.json`](tests/fixtures/online/registry.json).
Pinned URLs and sha256 hashes live in [`manifest.json`](tests/fixtures/online/manifest.json); cached CSV
slices in `tests/fixtures/online/cache/` power CI (no network). Formats include CSV, TSV, JSON, and ZIP.

| Tier | Count | CI scope |
|---|---|---|
| **Tier 1** (anchors) | 10 | Full expectations + golden snapshots + live URL checks |
| **Tier 2** | 40 | Smoke tests (all strategies run, basic invariants) |

Tier 1 anchors: `titanic`, `wine_quality`, `adult_income`, `air_quality_uci`, `iris`,
`loan_approval`, `heart_cleveland`, `bank_marketing`, `mushroom`, `weather_json`.

Domain coverage: UCI classics, GitHub mirrors, environmental panels (OWID), finance/census,
JSON-native (Vega datasets), medical, and high-dimensional numeric sets.

Refresh cached slices:

```bash
python scripts/fetch_online_fixtures.py --discover --update-manifest
python scripts/fetch_online_fixtures.py --refresh --only titanic
python scripts/search_datasets.py --tag missing --domain finance
python scripts/search_datasets.py --format json
```

Debug, explain, and compare:

```bash
python scripts/debug_datasets.py --online --explain titanic
python scripts/debug_datasets.py --infer-roles --online adult_income
python scripts/debug_datasets.py --search missing --online
python benchmarks/bench.py --online-all --compare
python benchmarks/bench.py --online-all --tier 1
```

Reverse-engineering APIs:

```python
import freshdata as fd

# Infer column roles before cleaning
print(fd.infer_roles(df))

# Explain what clean() did and why
explanation = fd.explain_clean(df, strategy="balanced")
print(explanation.summary())
print(explanation.roles)
```

Polars adapter (optional extra):

```bash
pip install freshdata-cleaner[polars]
```

```python
import polars as pl
cleaned = fd.clean(pl_df)  # returns pl.DataFrame when input is Polars
```

Live URL validation (network required, not default CI):

```bash
pytest -m online tests/test_online_datasets.py
pytest -m tier1 tests/test_online_datasets.py
```

### Compare cleaning across strategies

```python
import freshdata as fd

# Actual outcomes: missing after, duration, models used
print(fd.compare_clean(df))

# Planned models + optional actual metrics
print(fd.compare_plans(df, include_metrics=True))
```

### Performance expectations

Typical throughput on a modern laptop (see `tests/fixtures/perf/baselines.json`):

| Dataset size | Balanced | Aggressive |
|---|---|---|
| 500 rows | <0.5s | <1s |
| 3,000 rows | <2.5s | <6s |
| 29k rows (full AQI) | <5s | KNN gated |

Run benchmarks:

```bash
python benchmarks/bench.py --fixtures --compare   # all fixtures, side-by-side
pytest -m large                                   # optional full AQI.csv (set FRESHDATA_AQI_PATH)
```

Performance is achieved via vectorized pandas/NumPy and one-pass engine caching
(correlation matrix, column contexts). A C extension is **not** used — profiling
showed the bottleneck was KNN on large frames (now gated to aggressive mode only).

## Development

```bash
git clone https://github.com/FreshCode-Org/freshdata
cd freshdata
pip install -e ".[dev,ml,polars]"
pytest
ruff check src tests
mypy src/freshdata
```

Update golden report snapshots after intentional engine changes:

```bash
pytest tests/test_golden.py tests/test_online_datasets.py --update-golden
```

Benchmarks: `python benchmarks/bench.py` (synthetic),
`python benchmarks/bench.py --fixtures --compare` (11 local scenario fixtures), or
`python benchmarks/bench.py --online --compare` (6 online cached datasets).

Optional large-file benchmark (29k-row AQI.csv, not committed to repo):

```bash
export FRESHDATA_AQI_PATH=/path/to/AQI.csv
pytest -m large
```

## License

MIT — see [LICENSE](LICENSE).
