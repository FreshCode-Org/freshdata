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
If you are choosing a first task, start with the
[First PR guide](community/first-pr.md), browse the
[Contributor roadmap](community/contributor-roadmap.md) for work grouped by
difficulty, and skim the
[architecture map](https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md)
to find where your change belongs.

## Development setup

```bash
git clone https://github.com/FreshCode-Org/freshdata
cd freshdata
pip install -e ".[dev,ml]"     # matches CI; the dev extra already includes polars
pre-commit install
```

## Run the checks

```bash
pytest -m "not online and not large"   # fast test lane (coverage gate ≥ 93%)
ruff check .                           # lint
mypy src/freshdata                     # type check
```

All three must pass — they are exactly what the required CI lane runs. The
test matrix repeats the fast lane on Python 3.9–3.13; `online`/`large` marked
tests need network access and run in the nightly lane instead.

The 93% coverage gate fires on *every* `pytest` run, so running a single test
file on its own will fail the gate. While iterating, disable it for that run:

```bash
pytest tests/test_foo.py --no-cov       # iterate on one file
```

Then run the full `pytest -m "not online and not large"` before you open the PR.

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
