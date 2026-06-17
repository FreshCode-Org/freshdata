<div align="center">

# freshdata

### Automated DataFrame cleaning for pandas — explainable, safe, and production-ready.

*One call turns a messy CSV, Excel, or SQL export into analysis- and ML-ready data — and tells you exactly what it changed and **why**.*

[![PyPI Version](https://img.shields.io/pypi/v/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Docs](https://github.com/FreshCode-Org/freshdata/actions/workflows/docs.yml/badge.svg)](https://freshcode-org.github.io/freshdata/)
[![Downloads](https://img.shields.io/pypi/dm/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FreshCode-Org/freshdata/badges/coverage.json)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org/)

[**Documentation**](https://freshcode-org.github.io/freshdata/) ·
[**Quickstart**](https://freshcode-org.github.io/freshdata/quickstart/) ·
[**API Reference**](https://freshcode-org.github.io/freshdata/api-reference/) ·
[**Examples**](examples/) ·
[**Changelog**](CHANGELOG.md)

</div>

![Terminal demo: run `fd.clean(df)` and inspect `report.summary()`](docs/assets/demo/freshdata-terminal-demo.gif)

*Before/after at a glance: the terminal starts with a messy `messy_export.csv`, runs `fd.clean(df)`, and then prints `report.summary()` showing the cleaning actions and before/after shape.*

---

`freshdata` is an **automated data-cleaning library for Python** that does real,
intelligent preprocessing of real-world tabular data. It is **not** a `fillna`
wrapper: a rule-based decision engine profiles every column (missing ratio,
dtype, skewness, cardinality, inferred role) and chooses the right action per
column — then logs a rationale, a risk level, and a confidence score for each
decision so nothing happens silently.

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("export.csv")

cleaned = fd.clean(df)                              # one line
cleaned, report = fd.clean(df, return_report=True)  # ... with a full audit trail
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

## ✨ Key features

- **Automated DataFrame cleaning in one call** — `fd.clean(df)` handles missing
  values, outliers, duplicates, dtype repair, and messy column names.
- **Per-column decision engine** — infers each column's role (id, target,
  datetime, free text, categorical, numeric) and applies explicit, documented
  threshold rules instead of one blunt global strategy.
- **Explainable by design** — every decision carries a rationale, risk level,
  and confidence score. If a `NaN` survives, the report says exactly why.
- **Safe defaults** — never imputes an identifier, never modifies a target/label
  column, never force-fills free text, never removes outliers blindly.
- **AI-ready preprocessing** — produces clean, typed, leakage-aware frames ready
  for scikit-learn, XGBoost, or any ML pipeline.
- **Data profiling** — `fd.profile(df)` gives read-only data-quality insight
  using the same inference code as `clean`, so previews are faithful.
- **pandas-first, Polars-optional** — pandas + NumPy core; pass a Polars frame
  and get a Polars frame back when the optional adapter is installed.
- **Enterprise layer** — opt-in fuzzy clustering, PII masking, semantic
  validation, a 0–100 Data Trust Score, OpenLineage metadata, and a batch CLI.
- **Typed, tested, fast** — fully type-hinted (`py.typed`), 800+ tests, 95%+
  coverage, vectorized pandas/NumPy throughout.

## 🤔 Why FreshData exists

Most data-cleaning code is hand-written, one-off, and silent. People reach for
`df.dropna()` or `df.fillna(0)` and quietly corrupt their analysis — imputing an
ID, leaking a target, or deleting the very outliers that *were* the signal.
General-purpose tools don't fix this:

- **pandas** gives you primitives, not decisions — you still write every rule.
- **profiling tools** (sweetviz, ydata-profiling) *describe* data but don't
  clean it.
- **validation tools** (Great Expectations) *check* data but don't repair it.

`freshdata` fills the gap: an opinionated engine that **makes the right cleaning
decision per column and explains it**, so you get reproducible, auditable,
ML-ready data without writing — or trusting — yet another bespoke script.

## 📦 Installation

```bash
pip install freshdata-cleaner                 # pandas + numpy only
pip install "freshdata-cleaner[ml]"           # + scikit-learn (KNN imputation, IsolationForest)
pip install "freshdata-cleaner[enterprise]"   # + polars, pyarrow, requests, pyyaml (enterprise layer + CLI)
pip install "freshdata-cleaner[all]"          # everything, including cleanlab
```

Requires **Python ≥ 3.9** and **pandas ≥ 1.5**. Verify the install:

```bash
python -c "import freshdata as fd; print(fd.__version__)"
```

## 🚀 Quickstart

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("messy_export.csv")

# Clean with sensible, explainable defaults
cleaned, report = fd.clean(df, return_report=True)

print(report.summary())        # human-readable audit trail
report.to_frame()              # decisions as a DataFrame
report.to_dict()               # JSON-friendly for logging / dashboards
```

Preview the engine's choices *before* touching your data:

```python
print(fd.profile(df))                    # read-only data-quality report
print(fd.suggest_plan(df).summary())     # the exact plan clean() would run
print(fd.compare_plans(df))              # strategies side by side
```

## 🔁 Before vs after

<table>
<tr><th>Before — raw export</th><th>After — <code>fd.clean(df)</code></th></tr>
<tr><td>

| First Name | AGE | Salary($) | empty |
|---|---|---|---|
| ` Ann ` | `34` | `$1,200.50` | |
| `Bob` | `N/A` | `-` | |
| `Bob` | `N/A` | `-` | |
| `Cara` | `41` | `$2,000` | |

*whitespace, `N/A`/`-` sentinels, currency strings, an all-empty column, a duplicate row, text dtypes*

</td><td>

| first_name | age | salary | age_was_missing |
|---|---|---|---|
| Ann | 34 | 1200.50 | False |
| Bob | 38 | _Missing_ | True |
| Cara | 41 | 2000.00 | False |

*snake_case names, real `Int64`/`float64` dtypes, sentinels → missing → imputed, duplicate dropped, empty column removed, a missingness indicator added*

</td></tr>
</table>

Every one of those changes appears in `report.summary()` with a rationale, risk
level, and confidence score — no silent mutations.

## 🧩 Core API

| name | purpose |
|---|---|
| `fd.clean(df, *, return_report=False, config=None, **options)` | clean, optionally returning a `CleanReport` |
| `fd.plan(df, *, mode="suggest", **options)` | build a serializable `RepairPlan` with row, column, and cell patches |
| `fd.repair(df, *, mode="repair_safe", return_plan=False, **options)` | apply safe, reviewed, or aggressive repair modes |
| `fd.profile(df, *, include_plan=False, **options)` | read-only inspection with actionable issues |
| `fd.suggest_plan(df, **options)` | dry-run: primary + alternative models per column |
| `fd.compare_plans(df, *, strategies=...)` | side-by-side models across strategies |
| `fd.compare_clean(df, *, strategies=...)` | side-by-side actual clean outcomes |
| `fd.explain_clean(df, **options)` | what `clean()` did and why, plus inferred roles |
| `fd.Cleaner(config=None, **options)` | reusable configured pipeline (`.clean()`, `.report_`) |
| `fd.CleanConfig` | frozen dataclass holding every option |
| `fd.CleanReport` / `fd.Action` | audit trail with rationale / risk / confidence |
| `fd.RepairPlan` / `fd.RepairPatch` | reversible repair artifact and patch log |

```python
# Tune the engine — explicit choices always override the defaults
cleaned = fd.clean(
    df,
    strategy="balanced",          # "aggressive" | "conservative"
    target_column="churn",        # never modified (no leakage)
    id_columns=("customer_id",),  # never imputed
    preserve_columns=("notes",),  # never dropped
    outlier_method="iqr",         # "zscore" | "auto" | "isolation_forest"
    return_report=True,
)

# Reusable pipeline across many files
cleaner = fd.Cleaner(target_column="churn")
for path in paths:
    out = cleaner.clean(pd.read_csv(path))
    log.info(cleaner.report_.summary())
```

<details>
<summary><b>How the cleaning engine works (two layers)</b></summary>

**Layer 1 — representation repair** (always on):

| order | step | what it does |
|---|---|---|
| 1 | `column_names` | snake_case names, deduplicate collisions (`"a", "a"` → `"a", "a_2"`) |
| 2 | `strip_whitespace` | trim surrounding whitespace in text cells |
| 3 | `normalize_sentinels` | `"N/A"`, `"null"`, `"-"`, `""`, `"#REF!"`, … → missing |
| 4 | `drop_empty_columns` / `drop_empty_rows` | remove all-missing columns and rows |
| 5 | `fix_dtypes` | text → numeric (`"$1,234.56"` works) / datetime / boolean, validated |
| 6 | `drop_duplicates` | resolve duplicate rows (`first`/`last`/`drop`/`aggregate`) |

**Layer 2 — the decision engine** (`strategy="balanced"`, the default) infers
each column's role and applies explicit threshold rules:

| missing ratio | numeric | categorical | datetime |
|---|---|---|---|
| ≤ 5% | mean if ~normal & no outliers, else median | mode if clear majority, else `"Unknown"` | ffill/bfill if time-ordered |
| 5–30% | median (KNN only in aggressive mode) | mode if dominant, else `"Missing"` | ffill/bfill if time-ordered |
| > 30% | **preserved** + warning (balanced) | same | same |

Role gates run first: **targets are never modified**, **IDs are never imputed**,
**free text is never force-filled**. Outliers in ID/target columns,
`preserve_columns`, and domain-sensitive columns (AQI, pollutants, fraud/risk
names) are always preserved — there the extremes usually *are* the signal.

</details>

## ⚡ Performance highlights

Typical throughput on a modern laptop (vectorized pandas/NumPy, one-pass engine
caching — no C extension required):

| Dataset size | Balanced | Aggressive |
|---|---|---|
| 500 rows | < 0.5 s | < 1 s |
| 3,000 rows | < 2.5 s | < 6 s |
| 29k rows (full AQI) | < 5 s | KNN gated |

```bash
python benchmarks/bench.py --fixtures --compare   # all fixtures, side by side
```

## 📊 How FreshData compares

| Capability | **freshdata** | pandas | pyjanitor | Great Expectations | sweetviz | cleanlab |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| One-call automatic cleaning | ✅ | ❌ | ➖ | ❌ | ❌ | ❌ |
| Per-column decisions by inferred role | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Missing-value imputation (smart) | ✅ | ➖ | ➖ | ❌ | ❌ | ❌ |
| Outlier detection & handling | ✅ | ❌ | ❌ | ➖ | ➖ | ✅ |
| Duplicate resolution | ✅ | ➖ | ✅ | ❌ | ❌ | ❌ |
| Dtype / format repair | ✅ | ➖ | ✅ | ❌ | ❌ | ❌ |
| Explainable audit trail | ✅ | ❌ | ❌ | ➖ | ❌ | ➖ |
| Data profiling | ✅ | ➖ | ❌ | ➖ | ✅ | ❌ |
| Data validation / quality gates | ✅¹ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PII masking | ✅¹ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Label-noise (ML) detection | ✅¹ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Polars support | ✅ | ❌ | ❌ | ➖ | ❌ | ❌ |

✅ built-in · ➖ partial / manual · ❌ not a goal · ¹ via the optional enterprise layer

## 🌍 Real-world use cases

- **ML preprocessing** — turn raw CSVs into leakage-aware, typed feature matrices
  before scikit-learn / XGBoost, without imputing IDs or touching the label.
- **Analytics & BI ingestion** — clean CRM, finance, and survey exports
  (currency strings, `N/A` sentinels, duplicate rows) on the way into a warehouse.
- **Data-quality gates in ETL** — run the enterprise CLI in Airflow/Prefect/cron;
  fail the job when the Data Trust Score drops below a threshold.
- **Exploratory data analysis (EDA)** — `fd.profile(df)` surfaces missingness,
  dtype issues, and duplicates before you commit to a modeling approach.
- **Notebook hygiene** — replace ad-hoc `dropna`/`fillna` cells with one
  auditable, reproducible call.

## 🛠️ Example pipeline

```python
import pandas as pd
import freshdata as fd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

raw = pd.read_csv("customers.csv")

# 1. Clean with the target protected from leakage
clean_df, report = fd.clean(raw, target_column="churn", return_report=True)
assert not report.warnings, report.warnings        # gate on data quality

# 2. Split & model on AI-ready data
X = pd.get_dummies(clean_df.drop(columns="churn"))
y = clean_df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

model = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
print("accuracy:", model.score(X_te, y_te))
```

See [`examples/`](examples/) for 8 runnable scripts and [`notebooks/`](notebooks/)
for narrated walkthroughs.

<details>
<summary><b>Enterprise layer — clustering, PII masking, trust scores, lineage, CLI</b></summary>

```python
from freshdata.enterprise import (
    clean_enterprise, EnterpriseConfig, ClusterConfig, MaskingRule, SemanticValidatorConfig,
)

ec = EnterpriseConfig(
    enable_clustering=True,
    clustering=ClusterConfig(columns=("vendor",)),       # merge "Acme Inc" / "ACME  inc"
    masking=(MaskingRule(name="pii", columns=("email",), strategy="hash", salt="…"),),
    semantic=(SemanticValidatorConfig(name="iso", kind="reference",
              columns=("country",), reference=("US", "CA", "GB")),),
    fail_under_trust=80,                                  # quality gate
)
result = clean_enterprise(df, enterprise=ec)             # df may be pandas OR polars
print(result.quality.to_markdown())                      # before/after trust report
result.lineage.emit("lineage.json")                      # OpenLineage RunEvents
assert result.passed_gate
```

Batch CLI (exits non-zero when the trust gate fails):

```bash
freshdata clean in.csv -o out.parquet --mask email:hash --cluster vendor \
    --report quality.json --lineage lineage.json --fail-under-trust 80
freshdata trust in.csv --fail-under 90
freshdata profile in.csv --json
```

</details>

## 📚 Documentation

Full documentation lives at **<https://freshcode-org.github.io/freshdata/>**:

- [Installation](https://freshcode-org.github.io/freshdata/installation/)
- [Quickstart](https://freshcode-org.github.io/freshdata/quickstart/)
- [Cleaning engine](https://freshcode-org.github.io/freshdata/cleaning-engine/)
- [Data profiling](https://freshcode-org.github.io/freshdata/data-profiling/)
- [API reference](https://freshcode-org.github.io/freshdata/api-reference/)
- [Examples](https://freshcode-org.github.io/freshdata/examples/)
- [Benchmarks](https://freshcode-org.github.io/freshdata/benchmarks/)
- [FAQ](https://freshcode-org.github.io/freshdata/faq/)

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). Quick start:

```bash
git clone https://github.com/FreshCode-Org/freshdata
cd freshdata
pip install -e ".[dev,ml,polars]"
pre-commit install
pytest && ruff check src tests && mypy src/freshdata
```

Security issues: see [SECURITY.md](SECURITY.md) for private disclosure.

## 🗺️ Roadmap

- [x] Per-column decision engine with explainable reports (0.3)
- [x] Enterprise layer: clustering, masking, trust score, lineage, CLI (0.4)
- [x] Documentation site + examples + packaging governance (0.5)
- [ ] Pluggable custom cleaning rules / strategy registry
- [ ] Native Polars cleaning engine (beyond the adapter)
- [ ] HTML/interactive profiling report
- [ ] Config-as-YAML for the core cleaner (not just the CLI)
- [x] 1.0 — stable public API

Have an idea? [Open a discussion or issue.](https://github.com/FreshCode-Org/freshdata/issues)

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Maintainer

Built and maintained by **Johnny Wilson Dougherty**
([@JohnnyWilson-Portfolio](https://github.com/JohnnyWilson-Portfolio)).

Contributions by **Kevin Costner**
([@kevincostner17](https://github.com/kevincostner17)).

If `freshdata` saves you time, please ⭐ the
[repository](https://github.com/FreshCode-Org/freshdata) — it genuinely helps
others discover the project.
