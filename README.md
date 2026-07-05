<div align="center">

<img src="docs/assets/logo.svg" alt="freshdata logo" width="180">

# freshdata

**The explainable cleaning layer for pandas — decision-preserving data hygiene.**

One call turns a messy CSV, Excel, or SQL export into analysis- and ML-ready
data, and tells you exactly what it changed and why.

[![PyPI Version](https://img.shields.io/pypi/v/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![Python Versions](https://img.shields.io/pypi/pyversions/freshdata-cleaner.svg)](https://pypi.org/project/freshdata-cleaner/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml)
[![Docs](https://github.com/FreshCode-Org/freshdata/actions/workflows/docs.yml/badge.svg)](https://freshcode-org.github.io/freshdata/)

[Documentation](https://freshcode-org.github.io/freshdata/) ·
[Quickstart](https://freshcode-org.github.io/freshdata/quickstart/) ·
[API Reference](https://freshcode-org.github.io/freshdata/api-reference/) ·
[Changelog](CHANGELOG.md)

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

## Installation

```bash
pip install freshdata-cleaner
```

Optional extras add ML imputation, domain packs, the enterprise layer, privacy,
and entity resolution — see the [installation guide](https://freshcode-org.github.io/freshdata/installation/).
Requires Python >= 3.9 and pandas >= 1.5.

## Quick start

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
  and get one back with the optional adapter.
- **Typed, tested, fast** — fully type-hinted (`py.typed`), vectorized, with a
  93% coverage gate enforced in CI.

## Documentation

Full guides, the API reference, the enterprise layer (drift, privacy, entity
resolution), streaming, and the semantic cleaning layer live at
**[freshcode-org.github.io/freshdata](https://freshcode-org.github.io/freshdata/)**.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local
setup, the checks to run before a PR, and the code-quality standards.

## License

Released under the [MIT License](LICENSE).
