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

[Documentation](https://freshcode-org.github.io/freshdata/) ·
[Quickstart](https://freshcode-org.github.io/freshdata/quickstart/) ·
[API Reference](https://freshcode-org.github.io/freshdata/api-reference/) ·
[Changelog](https://github.com/FreshCode-Org/freshdata/blob/main/CHANGELOG.md)

</div>

## What is freshdata?

`freshdata` is an automated data-cleaning library for Python. A rule-based
decision engine profiles every column — missing ratio, dtype, skewness,
cardinality, inferred role — and chooses the right action per column. Every
decision carries a rationale, a risk level, and a confidence score, so
nothing happens silently and nothing is left unexplained.

It fills the gap between tools that only *describe* data and tools that only
*validate* it: freshdata makes the cleaning decision and shows its work,
producing reproducible, auditable, ML-ready output with an audit trail you can
hand to a reviewer.

## Key features

- **One-call cleaning** — `fd.clean(df)` handles missing values, outliers,
  duplicate detection (removal is opt-in), dtype repair, and messy column
  names.
- **Per-column decision engine** — infers each column's role and applies
  explicit, documented rules instead of one blunt global strategy.
- **Explainable by design** — every action carries a rationale, risk level, and
  confidence score; if a `NaN` survives, the report says why.
- **Safe defaults** — never imputes an identifier, modifies a target column, or
  removes outliers blindly.
- **pandas-first, scalable when needed** — pandas + NumPy core; pass a Polars
  frame and get one back, with optional Polars/DuckDB/Spark
  [execution backends](https://freshcode-org.github.io/freshdata/backends/)
  for larger-than-memory data.
- **CLI included** — `clean`, `plan`, `apply-plan`, `profile`, `learn`, and
  `trust` subcommands for scripting and CI pipelines without writing Python.
- **Typed and tested** — fully type-hinted (`py.typed`), vectorized, with a
  93% coverage gate enforced in CI.

## Installation

```bash
pip install freshdata-cleaner
```

> The PyPI distribution is `freshdata-cleaner`; the import name is `freshdata`.

Requires Python >= 3.9 and pandas >= 1.5. The core install depends only on
pandas and NumPy; everything else is an optional extra:

```bash
pip install "freshdata-cleaner[ml,polars]"
```

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

See the [quickstart guide](https://freshcode-org.github.io/freshdata/quickstart/)
for strategies, reports, and CLI usage.

## Beyond core cleaning

Optional layers, all off by default and covered in the documentation:

- [Repair plans](https://freshcode-org.github.io/freshdata/repair-plans/) —
  suggest a reviewable plan, then apply exactly the approved actions.
- [Context policies](https://freshcode-org.github.io/freshdata/context-policies/) —
  compile plain-English cleaning rules into an enforceable policy.
- [Streaming](https://freshcode-org.github.io/freshdata/streaming/) —
  micro-batch and time-series-aware cleaning with bounded memory.
- [Privacy](https://freshcode-org.github.io/freshdata/feature-overview/) —
  PII detection, masking, and jurisdiction-aware anonymization policies.
- [Plugins](https://freshcode-org.github.io/freshdata/plugins/) — extend the
  engine with your own experts, validators, and backends.
- [AI Copilot](https://freshcode-org.github.io/freshdata/ai-copilot/)
  *(experimental)* — deterministic, offline dataset analysis that returns an
  explainable cleaning plan and copy-ready freshdata code; no API key required.

## Documentation and examples

- [Documentation site](https://freshcode-org.github.io/freshdata/) — guides,
  [API reference](https://freshcode-org.github.io/freshdata/api-reference/),
  [benchmarks](https://freshcode-org.github.io/freshdata/benchmarks/), and
  [honest limitations](https://freshcode-org.github.io/freshdata/limitations/).
- [`examples/`](https://github.com/FreshCode-Org/freshdata/tree/main/examples) —
  runnable, self-contained scripts, indexed in
  [`examples/README.md`](https://github.com/FreshCode-Org/freshdata/blob/main/examples/README.md).

## Contributing

Contributions are welcome — standard GitHub flow: fork, branch, add tests,
open a pull request. CI runs `ruff`, `mypy`, and the fast pytest lane on every
PR. See [CONTRIBUTING.md](https://github.com/FreshCode-Org/freshdata/blob/main/CONTRIBUTING.md)
for setup and guidelines, and
[CODE_OF_CONDUCT.md](https://github.com/FreshCode-Org/freshdata/blob/main/CODE_OF_CONDUCT.md)
for community standards.

New here? Good places to start:

- [Good first issues](https://github.com/FreshCode-Org/freshdata/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
  and the [contributor roadmap](https://freshcode-org.github.io/freshdata/community/contributor-roadmap/) —
  open work grouped by difficulty.
- [ARCHITECTURE.md](https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md) —
  how the code is laid out.
- [Discussions](https://github.com/FreshCode-Org/freshdata/discussions) — ask
  questions and float ideas before you build.

## License

MIT — see [LICENSE](https://github.com/FreshCode-Org/freshdata/blob/main/LICENSE).
