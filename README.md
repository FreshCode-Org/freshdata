# freshdata

**Fast, safe, automatic data cleaning for real-world tabular data.**

[![CI](https://github.com/JohnnyWilson-Portfolio/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/JohnnyWilson-Portfolio/freshdata/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/freshdata-cleaner/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

`freshdata` fixes the messy parts of CSV / Excel / SQL-export data — stray
whitespace, `"N/A"` strings, numbers stored as text, duplicate rows — in one
call, and tells you exactly what it did.

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")

cleaned = fd.clean(df)                      # one line
cleaned, report = fd.clean(df, report=True) # ... with a full audit trail
print(report.summary())
```

```text
freshdata clean report
  rows:    5 -> 4 (-1)
  columns: 6 -> 5 (-1)
  memory:  1.5 KB -> 298 B
  time:    0.011s
  actions (12):
    - [column_names] renamed 5 column(s): ' First Name '->'first_name', 'AGE'->'age', …
    - [strip_whitespace] 'first_name': trimmed surrounding whitespace
    - [normalize_sentinels] 'age': replaced sentinel strings ("N/A", "-", "", …) with missing
    - [drop_empty_columns] dropped 1 all-missing column(s): empty
    - [fix_dtypes] 'age': converted to Int64
    - [fix_dtypes] 'joined_date': converted to datetime64[ns]
    - [fix_dtypes] 'active': converted to bool
    - [fix_dtypes] 'salary': converted to float64
    - [drop_duplicates] dropped 1 duplicate row(s)
```

## Install

```bash
pip install freshdata-cleaner
```

Requires Python ≥ 3.9 and pandas ≥ 1.5. No other dependencies.

## Why another cleaning library?

Most auto-cleaners are either trivial wrappers or opaque frameworks that
guess. `freshdata` is built on four rules:

1. **No surprises.** Defaults only repair *representation* — whitespace,
   sentinel strings, wrong dtypes, exact duplicate rows, all-empty
   rows/columns. Anything that changes your data's *statistics* (imputation,
   outlier handling, lossy downcasting) is opt-in.
2. **Everything is reported.** Every transformation is recorded with the
   column name and the number of affected cells. `bool(report)` is `False`
   when nothing changed.
3. **Never mutates your input.** `clean` returns a new frame (built from a
   shallow copy, so unchanged columns cost no extra memory). `profile` is
   read-only.
4. **Fast by construction.** Vectorized pandas operations only — no
   row-wise `apply`. Type inference pre-screens a sample, so hopeless
   conversions are rejected at O(sample), not O(n), and conversions only
   stick when ≥ 95 % of values parse (configurable).

## What `clean` does by default

| order | step | what it does |
|---|---|---|
| 1 | `column_names` | snake_case names, deduplicate collisions (`"a", "a"` → `"a", "a_2"`) |
| 2 | `strip_whitespace` | trim surrounding whitespace in text cells (internal spacing kept) |
| 3 | `normalize_sentinels` | `"N/A"`, `"null"`, `"-"`, `""`, `"#REF!"`, … → missing |
| 4 | `drop_empty_columns` / `drop_empty_rows` | remove all-missing columns and rows |
| 5 | `fix_dtypes` | text → numeric (`"$1,234.56"` works) / datetime / boolean, validated |
| 6 | `drop_duplicates` | drop exact duplicate rows, keep the first |

Conversions are conservative: a column converts only when at least
`numeric_threshold` (default 0.95) of its non-missing values parse, mixed-type
columns never lose their non-string values, and every value coerced to missing
is counted in the report.

## Opt-in steps

```python
fd.clean(
    df,
    impute="auto",              # median for numeric, mode otherwise ("mean"/"median"/"mode")
    outliers="clip",            # or "flag" to add a boolean <col>_outlier column
    outlier_method="iqr",       # or "zscore"; factors default to 1.5 / 3.0
    drop_constant_columns=True, # single-valued columns
    optimize_memory=True,       # downcast numerics, categorize low-cardinality text
    reset_index=True,           # 0..n-1 index instead of original labels
)
```

Every option lives on one frozen dataclass — `fd.CleanConfig` — and unknown
names fail immediately with a "did you mean" suggestion:

```python
config = fd.CleanConfig(drop_duplicates=False, extra_sentinels=("unknown",))
fd.clean(df, config=config, impute="median")   # config + overrides

cleaner = fd.Cleaner(impute="median")          # reusable pipeline
for path in paths:
    out = cleaner.clean(pd.read_csv(path))
    log.info(cleaner.report_.summary())
```

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

`profile.to_frame()` gives the same as a DataFrame; `profile.to_dict()` is
JSON-friendly for logging and data-quality dashboards.

## What freshdata will not do

- Guess at fuzzy entity resolution ("Jon" vs "John").
- Impute, drop outliers, or change distributions unless you ask.
- Parse ambiguous European decimal commas (`"1.234,56"`) — too risky to guess.
- Mutate your DataFrame, ever.

## API

| name | purpose |
|---|---|
| `fd.clean(df, *, report=False, config=None, **options)` | clean, optionally returning a `CleanReport` |
| `fd.profile(df, *, config=None, **options)` | read-only inspection with actionable issues |
| `fd.Cleaner(config=None, **options)` | reusable configured pipeline (`.clean()`, `.report_`) |
| `fd.CleanConfig` | frozen dataclass holding every option |
| `fd.CleanReport` / `fd.Action` | audit trail (`summary()`, `to_dict()`, `to_frame()`) |
| `fd.Profile` / `fd.ColumnProfile` | profiling results |

## Development

```bash
git clone https://github.com/JohnnyWilson-Portfolio/freshdata
cd freshdata
pip install -e ".[dev]"
pytest
ruff check src tests
```

Benchmarks live in `benchmarks/bench.py` (`python benchmarks/bench.py`).

## License

MIT — see [LICENSE](LICENSE).
