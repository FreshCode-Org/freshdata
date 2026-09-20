<div align="center">

<img src="https://raw.githubusercontent.com/FreshCode-Org/freshdata/main/docs/assets/logo.png" alt="freshdata logo" width="220">

# freshdata

**Automated, explainable data cleaning for pandas and Polars — repair messy tabular data safely and see exactly what changed.**

[![PyPI Version](https://img.shields.io/pypi/v/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/FreshCode-Org/freshdata/blob/main/LICENSE)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Docs](https://github.com/FreshCode-Org/freshdata/actions/workflows/docs.yml/badge.svg)](https://freshcode-org.github.io/freshdata/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FreshCode-Org/freshdata/badges/coverage.json)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)

[Documentation](https://freshcode-org.github.io/freshdata/) ·
[Quickstart](https://freshcode-org.github.io/freshdata/quickstart/) ·
[API Reference](https://freshcode-org.github.io/freshdata/api-reference/) ·
[Benchmarks](https://freshcode-org.github.io/freshdata/benchmarks/) ·
[Changelog](https://github.com/FreshCode-Org/freshdata/blob/main/CHANGELOG.md)

</div>

---

### Minimal Visual Proof

```text
Messy DataFrame (pandas or Polars)
              │
              ▼
    fd.clean(df, return_report=True)
              │
      ┌───────┴────────────────┐
      ▼                        ▼
Cleaned DataFrame        Explainable Audit Report
(types repaired,         - Every cell action recorded
 missing handled,        - Rationale, risk level & confidence
 outliers flagged,       - Invariants protected (0% false repairs)
 duplicates reported)    - Review recommendations for humans
```

---

## 10-Second Explanation

Most tabular data tools fall into three camps:

* **Profilers** tell you *what is in your dataset* (summary distributions, missingness percentages).
* **Validators** tell you *whether your data satisfies rules* (schema assertions, pass/fail gates).
* **ML data-quality tools** detect *problematic training examples* (label errors, covariate drift).

**FreshData** safely decides **how to repair messy tabular data and explains what it changed and why**.

Instead of writing dozens of fragile, custom `fillna()`, `astype()`, and regex routines for every pipeline, FreshData inspects each column's distribution and inferred role (identifier, categorical, numeric, date, target label) to apply calibrated repairs. Nothing happens silently: every transformation carries an action log, a risk level, and an engine confidence score.

---

## Installation

```bash
pip install freshdata-cleaner
```

> [!IMPORTANT]
> **Package Identity**:
> * **PyPI distribution name**: [`freshdata-cleaner`](https://pypi.org/project/freshdata-cleaner/)
> * **Python import name**: `import freshdata as fd`
> * **GitHub repository**: [`FreshCode-Org/freshdata`](https://github.com/FreshCode-Org/freshdata)
>
> PyPI requires the distribution name `freshdata-cleaner` to avoid confusion with an older namespace. Always install with `pip install freshdata-cleaner`.

### Optional Extras

Core FreshData requires only Python >= 3.9, **pandas**, and **NumPy**. Additional capabilities are available via extras:

```bash
pip install "freshdata-cleaner[polars,ml]"
```

| Extra | Description |
|---|---|
| `polars` | Native Polars DataFrame support (`polars in -> polars out`) and PyArrow interchange |
| `ml` | Model-based and KNN imputation via `scikit-learn` |
| `duckdb` | Out-of-core execution for datasets exceeding RAM |
| `spark` | Distributed cleaning backend on PySpark clusters |
| `viz` | Interactive HTML report rendering (`itables`, `plotly`) |
| `privacy` | Presidio-based PII detection and format-preserving anonymization |
| `all` | Full feature suite including all engines and format parsers |

---

## Minimal Runnable Example

Clean any messy DataFrame in one line, or request the audit report to inspect decisions:

```python
import freshdata as fd
import pandas as pd

df = pd.read_csv("messy_export.csv")

# 1. Clean the DataFrame
cleaned = fd.clean(df)

# 2. Or clean with an explainable audit report
cleaned, report = fd.clean(df, return_report=True)
print(report.summary())
```

```text
freshdata clean report
  rows:    5,000 -> 4,980 (-20 duplicate rows detected)
  columns: 12 -> 13 (+1 outlier indicator)
  missing: 342 -> 18 cell(s) (preserved in protected/ambiguous fields)
  memory:  480.2 KB -> 412.0 KB
  actions (5):
    - [normalize_sentinels] 'annual_spend': replaced 14 sentinel strings ("N/A", "-") with missing
    - [fix_dtypes] 'annual_spend': converted string to Float64
    - [missing] 'age': imputed 8 missing values using median (skewness < 0.5)
    - [missing] 'account_id': preserved 4 missing values (identifier column)
    - [outliers] 'amount': flagged 6 outliers in new column 'amount_outlier'
  review (1):
    ? review 'account_id' manually: 4 missing identifier cells left untouched
```

Or clean from your terminal without writing Python:

```bash
freshdata clean messy_export.csv -o clean.csv --report audit.json
```

---

## Before / After Example

Here is what happens to a genuinely messy customer record:

### 1. Raw Input Data

| customer_id | full_name | age | signup_date | annual_spend | is_churned |
|---|---|---|---|---|---|
| `"C-101"` | `"  Alice Smith  "` | `29` | `"2023-01-15"` | `"$1,200.50"` | `0` |
| `"C-102"` | `"Bob Jones"` | `34` | `"2023/02/20"` | `"$450.00"` | `1` |
| `"C-103"` | `"Charlie Brown"` | `-4` | `"N/A"` | `"$3,100.00"` | `0` |
| `"C-104"` | `"Diana Prince"` | `42` | `"2023-04-10"` | `"missing"` | `1` |
| `"C-105"` | `"Evan Wright"` | `165` | `"not_a_date"` | `"$890.25"` | `0` |

### 2. Execution Pipeline

```text
RAW DATA
   │
   ▼
DETECTION  ──► 'customer_id' = identifier; 'is_churned' = target label
               'full_name' has padding whitespace
               'signup_date' has sentinel strings & unparseable format
               'annual_spend' has currency strings & sentinel "missing"
               'age' has negative (-4) and impossible (165) outliers
   │
   ▼
DECISION   ──► Protect 'customer_id' and 'is_churned' against mutation
               Trim whitespace on string columns
               Normalize "N/A" and "missing" sentinels to NaN
               Flag age outliers in 'age_outlier' (do not drop rows silently)
               Coerce parseable dates; preserve unparseables for review
   │
   ▼
REPAIR     ──► Execute calibrated non-destructive repairs
   │
   ▼
AUDIT      ──► Generate complete Action log with risk levels and confidence
```

### 3. Cleaned Output DataFrame

```python
cleaned, report = fd.clean(df, target_column="is_churned", return_report=True)
print(cleaned)
```

| customer_id | full_name | age | signup_date | annual_spend | is_churned | age_outlier |
|---|---|---|---|---|---|---|
| `C-101` | `Alice Smith` | `29` | `2023-01-15` | `1200.50` | `0` | `False` |
| `C-102` | `Bob Jones` | `34` | `2023-02-20` | `450.00` | `1` | `False` |
| `C-103` | `Charlie Brown` | `-4` | `NaN` | `3100.00` | `0` | `True` |
| `C-104` | `Diana Prince` | `42` | `2023-04-10` | `NaN` | `1` | `False` |
| `C-105` | `Evan Wright` | `165` | `NaN` | `890.25` | `0` | `True` |

---

## Why FreshData?

| Tool Category | Primary Purpose | Examples | Where FreshData Fits |
|---|---|---|---|
| **Profilers** | Describe distributions and missingness statistics | ydata-profiling, Great Tables | FreshData *uses* profiling internally, but takes action to repair errors rather than just graphing them. |
| **Validators** | Test whether incoming data satisfies rigid expectations | Great Expectations, Pandera | Validators tell you *that* data failed; FreshData safely repairs the data and passes clean frames to validators. |
| **ML Data Quality** | Detect label noise and training data anomalies | Cleanlab, Evidently | ML quality tools diagnose model training risks; FreshData handles structural and representation errors before feature engineering. |
| **FreshData** | **Safely repair messy tabular data and explain what changed** | `freshdata` | **One-call automated cleaning layer with decision rationale, audit trails, and strict safety invariants.** |

---

## Safety & Explainability Proof

Automated cleaning without safety controls corrupts production datasets. FreshData enforces strict safety invariants: **it explicitly refuses to guess when modification would introduce silent errors.**

| Field Type | Sample Input | FreshData Decision | Action Taken | Risk Level | Engine Confidence | Rationale |
|---|---|---|---|:---:|:---:|---|
| **Identifier Column** | `customer_id: [101, NaN, 103]` | **Preserve NaN** | Deliberately left missing | `low` | `1.0` | Imputing primary keys creates counterfeit entity records. |
| **Target Label** | `is_churned: [1, 0, NaN]` | **Preserve & Warn** | Emits review warning | `high` | `1.0` | Imputing target labels causes label leakage in ML models. |
| **Outlier Values** | `age: [25, 30, 165]` | **Flag, do not drop** | Adds `age_outlier = True` | `low` | `0.5` | Silent row drops discard valuable correlated features. |
| **Ambiguous Text** | `code: ["A1", "???", "B2"]` | **Flag sentinel** | Normalizes to `NaN` | `medium` | `0.8` | Leaves cell missing with review recommendation. |
| **Protected Column** | `user_specified` | **Locked** | Zero transformation | `none` | `1.0` | Honors `preserve_columns=["col_name"]` unconditionally. |

---

## Supported Workflows

### 1. In-Memory pandas
```python
import freshdata as fd
cleaned = fd.clean(df)
```

### 2. Native Polars DataFrames
Pass a Polars DataFrame, get a Polars DataFrame back with zero pandas boilerplate:
```python
import polars as pl
import freshdata as fd

pl_df = pl.read_csv("data.csv")
cleaned_pl = fd.clean(pl_df)
assert isinstance(cleaned_pl, pl.DataFrame)
```

### 3. Declarative Pipelines (`fd.pipeline`)
Build serializable, repeatable data hygiene pipelines:
```python
pipe = (
    fd.pipeline()
    .strip_whitespace()
    .normalize_sentinels()
    .fix_dtypes()
    .handle_missing(strategy="balanced")
)
cleaned = pipe.run(df)
```

### 4. Plan-First Review Workflow (`plan` & `apply`)
Suggest actions for human review before touching any data:
```python
plan = fd.plan(df)
print(plan.summary())      # Review proposed repairs
cleaned = fd.apply(plan, df)  # Apply only approved actions
```

### 5. CLI & CI Automation
Clean tabular files in bash scripts and data pipelines:
```bash
freshdata clean input.csv -o output.csv --report audit.json --strict
```

---

## Benchmark Proof

FreshData includes a reproducible, schema-stable benchmark harness measuring **nine standardized metrics** across synthetic enterprise fixtures (CRM, finance, event logs) from 10k to 25M rows.

* **Wall-clock performance**: Polars backend delivers 2–3× throughput vs pandas at 10M rows.
* **Memory footprint**: DuckDB backend consumes 200 MB peak RAM at 1M rows vs 1,046 MB for pandas.
* **Safety verification**: **0% false-repair rate** on protected identifiers and targets across all test fixtures.
* **Reproducibility**: Seed-controlled generation (`generate(n_rows, seed=42)`).

To run the standard benchmark suite locally:

```bash
python benchmarks/bench.py run
python benchmarks/bench.py report
```

See [`benchmarks/README.md`](benchmarks/README.md) and [`docs/benchmarks.md`](https://freshcode-org.github.io/freshdata/benchmarks/) for full methodology and hardware configurations.

---

## Flagship Examples

Runnable examples live in [`examples/`](examples/):

| Script | Purpose | Focus |
|---|---|---|
| [`01_csv_cleaning.py`](examples/01_csv_cleaning.py) | End-to-end messy CSV repair | Delimiters, whitespace, sentinels, JSON audit trail |
| [`02_missing_values.py`](examples/02_missing_values.py) | Role-aware missing data handling | Median vs mode vs preservation for identifiers |
| [`03_duplicate_detection.py`](examples/03_duplicate_detection.py) | Duplicate detection & resolution | Detecting duplicates without silent row drops |
| [`04_outlier_handling.py`](examples/04_outlier_handling.py) | Non-destructive outlier handling | IQR and Z-score outlier flagging vs clipping |
| [`05_ml_preprocessing.py`](examples/05_ml_preprocessing.py) | Machine learning pipeline readiness | Leakage prevention and scikit-learn integration |

Explore integration recipes in [`examples/integrations/`](examples/integrations/):
* [Polars Workflow](examples/integrations/polars_workflow.py)
* [DuckDB Out-of-Core](examples/integrations/duckdb_workflow.py)
* [Scikit-Learn Pipeline](examples/integrations/sklearn_pipeline.py)
* [Airflow Operator](examples/integrations/airflow_task.py)

---

## Documentation

* 📖 **[Documentation Site](https://freshcode-org.github.io/freshdata/)** — Complete tutorials and guides.
* 🚀 **[Quickstart Guide](https://freshcode-org.github.io/freshdata/quickstart/)** — Clean your first dataset in under 2 minutes.
* 🍳 **[FreshData Cookbook](https://freshcode-org.github.io/freshdata/cookbook/)** — Copy-paste recipes for common data engineering problems.
* 🔍 **[API Reference](https://freshcode-org.github.io/freshdata/api-reference/)** — Comprehensive signatures and docstrings.
* ⚖️ **[Honest Limitations](https://freshcode-org.github.io/freshdata/limitations/)** — Where FreshData stops and what it will not do.

---

## Community & Discussions

We welcome questions, ideas, and feedback:

* **[GitHub Discussions](https://github.com/FreshCode-Org/freshdata/discussions)** — Ask for architectural advice, propose integrations, or share benchmarks.
* **[Issue Tracker](https://github.com/FreshCode-Org/freshdata/issues)** — Report reproducible bugs or submit feature proposals.
* **[Contributor Roadmap](https://freshcode-org.github.io/freshdata/community/contributor-roadmap/)** — Browse open opportunities by difficulty level.

---

## Contributing

We love contributions! FreshData enforces a **93% test coverage gate** in CI to guarantee that automated cleaning never introduces silent bugs.

* **First-time contributor?** Start with our step-by-step **[First Contribution Guide](https://freshcode-org.github.io/freshdata/contributing/first-contribution/)**.
* **Browse open issues**: Check [Good First Issues](https://github.com/FreshCode-Org/freshdata/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).
* **Development setup**:
  ```bash
  git clone https://github.com/FreshCode-Org/freshdata.git
  cd freshdata
  python -m venv .venv && source .venv/bin/activate
  pip install -e ".[dev,ml]"
  pytest -m "not online and not large"
  ```
  *(Tip: while iterating on single files, use `pytest tests/test_my_change.py --no-cov`)*

Review our [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before submitting a pull request.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
