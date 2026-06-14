---
title: Contributing
description: >-
  How to contribute to freshdata — set up a development environment, run the test
  suite, linting, and type checks, and submit a pull request.
keywords: contribute freshdata, open source data cleaning, freshdata development setup
---

# Contributing

Contributions are welcome — bug reports, fixes, docs, and features. Please also
read the [Code of Conduct](https://github.com/FreshCode-Org/freshdata/blob/main/CODE_OF_CONDUCT.md).

## Development setup

```bash
git clone https://github.com/FreshCode-Org/freshdata
cd freshdata
pip install -e ".[dev,ml,polars]"
pre-commit install
```

## Run the checks

```bash
pytest                       # full test suite (coverage gate ≥ 93%)
ruff check src tests         # lint
ruff format --check src tests
mypy src/freshdata           # type check
```

All four must pass; CI runs them on Python 3.9–3.13.

## Build the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve     # live preview at http://127.0.0.1:8000
mkdocs build --strict
```

## Updating golden snapshots

After an intentional engine change:

```bash
pytest tests/test_golden.py tests/test_online_datasets.py --update-golden
```

## Pull requests

1. Branch from `main`.
2. Add tests for new behavior and keep coverage ≥ 93%.
3. Ensure `pytest`, `ruff`, and `mypy` pass.
4. Update `CHANGELOG.md` under the `Unreleased` section.
5. Open the PR using the template; describe the change and its rationale.

## Reporting security issues

Do not open public issues for vulnerabilities — see
[SECURITY.md](https://github.com/FreshCode-Org/freshdata/blob/main/SECURITY.md)
for private disclosure.
