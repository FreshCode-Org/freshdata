<div align="center">

<img src="https://raw.githubusercontent.com/FreshCode-Org/freshdata/main/docs/assets/logo.png" alt="freshdata logo" width="220">

# freshdata

**The explainable cleaning layer for pandas — decision-preserving data hygiene.**

One call turns a messy CSV, Excel, or SQL export into analysis- and ML-ready
data, and tells you exactly what it changed and why.

[![PyPI Version](https://img.shields.io/pypi/v/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/FreshCode-Org/freshdata/blob/main/LICENSE)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Docs](https://github.com/FreshCode-Org/freshdata/actions/workflows/docs.yml/badge.svg)](https://freshcode-org.github.io/freshdata/)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FreshCode-Org/freshdata/badges/coverage.json)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Benchmarks](https://img.shields.io/badge/benchmarks-ASV-blue.svg)](https://freshcode-org.github.io/freshdata/benchmarks/)

[Documentation](https://freshcode-org.github.io/freshdata/) ·
[Quickstart](https://freshcode-org.github.io/freshdata/quickstart/) ·
[API Reference](https://freshcode-org.github.io/freshdata/api-reference/) ·
[Changelog](https://github.com/FreshCode-Org/freshdata/blob/main/CHANGELOG.md)

</div>

## Overview

`freshdata` is an automated data-cleaning library for Python. It is not a
`fillna` wrapper: a rule-based decision engine profiles every column — missing
ratio, dtype, skewness, cardinality, inferred role — and chooses the right
action per column. Every decision carries a rationale, a risk level, and a
confidence score, so nothing happens silently and nothing is left unexplained.

It fills the gap between tools that only *describe* data (ydata-profiling) or
only *validate* it (Great Expectations): freshdata makes the cleaning decision
and shows its work, producing reproducible, auditable, ML-ready output.

It's aimed at data scientists, analytics engineers, and ML practitioners who
are tired of hand-rolling the same missing-value/outlier/dtype boilerplate for
every new dataset and want an audit trail they can hand to a reviewer.

## Key features

- **One-call cleaning** — `fd.clean(df)` handles missing values, outliers,
  duplicates, dtype repair, and messy column names.
- **Per-column decision engine** — infers each column's role and applies
  explicit, documented rules instead of one blunt global strategy.
- **Explainable by design** — every action carries a rationale, risk level, and
  confidence score; if a `NaN` survives, the report says why.
- **Safe defaults** — never imputes an identifier, modifies a target column, or
  removes outliers blindly.
- **pandas-first, Polars-optional** — pandas + NumPy core; pass a Polars frame
  and get one back, with optional Polars/DuckDB/Spark execution backends for
  scaling beyond pandas (see the
  [backends guide](https://freshcode-org.github.io/freshdata/backends/) for the
  measured out-of-core boundary per backend and output format).
- **CLI included** — `clean`, `plan`, `apply-plan`, `profile`, `learn`, and
  `trust` subcommands for scripting and CI pipelines without writing Python.
- **Typed, tested, fast** — fully type-hinted (`py.typed`), vectorized, with a
  93% coverage gate enforced in CI.

## Installation

```bash
pip install freshdata-cleaner
```

> The PyPI distribution is `freshdata-cleaner`; the import name is `freshdata`.

Requires Python >= 3.9 and pandas >= 1.5.

Most functionality beyond core cleaning ships as optional extras:

| Extra | Adds |
|---|---|
| `ml` | KNN/model-based imputation |
| `polars` | Polars DataFrame support |
| `duckdb` | Out-of-core execution via DuckDB |
| `spark` | Out-of-core execution via PySpark |
| `viz` | Interactive HTML report rendering |
| `privacy` | PII detection and anonymization |
| `enterprise` | Compliance reporting, orchestration hooks, quality-ops exporters |
| `all` | Everything above |

```bash
pip install "freshdata-cleaner[ml,polars]"
```

See the [installation guide](https://freshcode-org.github.io/freshdata/installation/)
for the full list of extras (domain packs, format parsers, streaming, entity
resolution, and more).

## Quickstart

```python
import pandas as pd
import freshdata as fd

df = pd.read_csv("messy_export.csv")

cleaned = fd.clean(df)                               # one line
cleaned, report = fd.clean(df, return_report=True)   # ... with a full audit trail
print(report.summary())
```

```text
freshdata clean report
  rows:    525 -> 500 (-25)
  columns: 7 -> 6 (-1)
  missing: 421 -> 0 cell(s)
  memory:  100.8 KB -> 89.2 KB
```

The same operation is available from the command line:

```bash
freshdata clean messy_export.csv -o clean.csv --report audit.json
```

## AI Copilot: explainable cleaning for messy real-world data (experimental)

One call analyzes a messy dataset and hands back an explainable cleaning
plan, a PII warning, context-policy violations, and copy-ready freshdata
code — **deterministic, offline, no API key required**.

```python
import pandas as pd
from freshdata.experimental.ai_copilot import analyze_dataset

df = pd.read_csv("examples/data/messy_customers.csv")

report = analyze_dataset(
    df,
    goal="Prepare this customer dataset for analytics and ML",
    privacy="mask_pii_before_reasoning",
    context_policy={
        "email": "must_mask",
        "phone": "must_mask",
        "age": "must_be_between_0_and_120",
        "salary": "must_be_positive",
        "city": "normalize_spelling",
    },
)

print(report.summary)           # ranked problems + trust score + PII warning
print(report.cleaning_plan)     # ordered steps, each with a rationale and the tool to use
print(report.recommended_code)  # runnable freshdata pipeline for THIS dataset
```

```text
FreshData AI Copilot report (experimental)
  engine:  deterministic-local
  goal:    Prepare this customer dataset for analytics and ML
  shape:   50 rows x 11 columns
  trust:   93.9/100 (grade A)
  problems: 14 found — 4 high, 6 medium, 4 low
    - (high) [email] contains EMAIL personal data — mask before sharing
    - (high) [age] 'age' has 8 value(s) outside [0, 120]
    - (high) [salary] 'salary' has 4 value(s) outside [0, inf]
    - (medium) [plan] near-duplicate spellings: 'Gold' ~ 'GOLD', 'gold'; …
    …
```

Privacy-first by design: raw PII never enters the copilot's model context —
sample rows are hashed/scrubbed before inclusion (or omitted entirely with
`privacy="schema_only"`), so nothing an LLM provider would see contains a raw
identifier. The default engine is rule-based and fully local; an optional
`provider` hook exists for plugging in an LLM later and is clearly marked
experimental.

Run the full messy-data-to-audit-ready story (masking → plain-English rules →
cleaning → trust re-scoring) with:

```bash
python examples/freshdata_ai_copilot_demo.py
```

See the [AI Copilot guide](https://freshcode-org.github.io/freshdata/ai-copilot/)
for the report anatomy, the context-policy rule vocabulary, and honest
limitations.

## Usage examples

The [`examples/`](https://github.com/FreshCode-Org/freshdata/tree/main/examples) directory has runnable, self-contained scripts.
A few starting points:

- [`freshdata_ai_copilot_demo.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/freshdata_ai_copilot_demo.py) —
  the flagship story: messy customer data → masked, validated, audit-ready.
- [`01_missing_values.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/01_missing_values.py) — the one-call
  cleaning path and reading the resulting report.
- [`04_profiling.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/04_profiling.py) — profiling a DataFrame
  without modifying it.
- [`05_ml_pipeline.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/05_ml_pipeline.py) — wiring `fd.clean` into a
  scikit-learn pipeline.
- [`07_pandas_integration.py`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/07_pandas_integration.py) — using
  freshdata alongside existing pandas code.

See [`examples/README.md`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/README.md) for the complete, indexed list.

## Project structure

```
freshdata/
├── src/freshdata/          # library source (engine, domains, enterprise, execution backends, CLI)
├── tests/                  # pytest suite
├── examples/               # runnable usage examples
├── docs/                   # mkdocs-material documentation site
├── benchmarks/             # CleanBench accuracy/performance benchmark harness
├── freshdata-benchmarks/   # comparative ASV benchmark suite (run locally)
├── training/               # dev-only training pipeline for the semantic models
└── crates/                 # optional Rust acceleration crate (freshcore)
```

## CLI reference

Installing the package provides a `freshdata` command with several
subcommands:

| Command | Purpose |
|---|---|
| `clean` | Clean a file and optionally write a JSON audit report |
| `plan` / `apply-plan` | Suggest a reviewable repair plan, then apply exactly the approved actions |
| `profile` | Print a read-only profile of a file, or audit/diff/merge `.fdprofile` files |
| `learn` | Learn a reusable cleaning profile from a (messy, clean) file pair |
| `trust` | Print the Data Trust Score of a file |
| `quality-ops` | Export a report to dbt/Great Expectations/exception-table/lineage artifacts |
| `policy compile` | Compile natural-language cleaning rules into a reviewable policy |
| `models status` / `models pull` | Manage optional local semantic models |

Run `freshdata <command> --help` for the full option list, or see the
[quickstart guide](https://freshcode-org.github.io/freshdata/quickstart/) for
CLI usage.

## Development setup

```bash
git clone https://github.com/FreshCode-Org/freshdata.git
cd freshdata
python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev,ml]"

pytest -m "not online and not large"   # fast lane, matches CI
ruff check .                           # lint, matches CI
mypy src/freshdata                     # typecheck
```

`pre-commit` hooks are configured in `.pre-commit-config.yaml`; run
`pre-commit install` after cloning to have them run automatically.

## Contributing

Contributions are welcome. The workflow is the standard GitHub flow: fork,
create a branch, make your change, add or update tests, and open a pull
request. CI runs linting (`ruff`), type checking (`mypy`), and the fast pytest
lane on every PR.

See [CONTRIBUTING.md](https://github.com/FreshCode-Org/freshdata/blob/main/CONTRIBUTING.md) for full details, including how to work
with the online-fixture test registry, and [CODE_OF_CONDUCT.md](https://github.com/FreshCode-Org/freshdata/blob/main/CODE_OF_CONDUCT.md)
for community guidelines.

## Roadmap

`freshdata` is under active development; see [CHANGELOG.md](https://github.com/FreshCode-Org/freshdata/blob/main/CHANGELOG.md) for
what has shipped and the [issue tracker](https://github.com/FreshCode-Org/freshdata/issues)
for what's being discussed.

## License

MIT — see [LICENSE](https://github.com/FreshCode-Org/freshdata/blob/main/LICENSE).
