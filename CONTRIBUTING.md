# Contributing to freshdata

Thanks for your interest! freshdata aims to stay **small and sharp** — a
focused data-cleaning library, not a framework. Contributions that fit that
philosophy are very welcome.

New to the project? The fastest paths in:

- **Pick something to work on** — the [contributor roadmap](docs/community/contributor-roadmap.md)
  groups open work by difficulty, from good first issues to architecture work.
- **Find where your change belongs** — [ARCHITECTURE.md](ARCHITECTURE.md) maps
  the `src/freshdata` layout and the `clean()` flow.
- **Ask a question** — open a thread in
  [Discussions](https://github.com/FreshCode-Org/freshdata/discussions); the
  issue tracker is for reproducible bugs and concrete proposals.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

- **Safety first.** Every statistical change (imputation, outlier handling,
  column drops) must be logged with rationale, risk, and confidence. Use
  ``strategy="conservative"`` when you only want representation repair.
- **Everything is reported.** Any new transformation must record an
  `Action` with an affected-cell count.
- **Vectorized only.** No row-wise `apply` / Python loops over rows in the
  cleaning path.
- **Tested.** New behavior needs tests, including the "does not fire when it
  shouldn't" case.

## Setup

```bash
git clone https://github.com/FreshCode-Org/freshdata
cd freshdata
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
pre-commit install     # optional: run the same hooks CI expects on every commit
```

## Checks to run before a PR

These mirror the required CI lane:

```bash
pytest -m "not online and not large"   # fast test lane (CI-required; includes the 93% coverage gate)
ruff check .                           # lint
mypy src/freshdata                     # types
```

Bare `pytest` additionally collects the `online`/`large` marked tests, which
need network access and local datasets — they run in the nightly CI lane, not
on PRs. If you change user-facing behavior, update the matching `docs/` page
and add a note under `Unreleased` in `CHANGELOG.md`.

The 93% coverage gate fires on every `pytest` run, so a single-file run fails it
on its own. While iterating, add `--no-cov` (e.g. `pytest tests/test_foo.py
--no-cov`), then run the full fast lane above before opening the PR.

First time contributing? See the
[first-PR walkthrough](docs/community/first-pr.md).

## Adding an online dataset fixture

1. Add an entry to [`tests/fixtures/online/registry.json`](tests/fixtures/online/registry.json)
   with `url`, `format` (`csv`/`tsv`/`json`/`jsonl`/`zip`), `domain`, `tags`, `tier`, and
   optional `read_csv` / `read_json` / `zip_member` options.
2. Run `python scripts/fetch_online_fixtures.py --only <id> --discover --update-manifest`
   to download, write `cache/<id>.csv`, and pin `sha256` in the manifest.
3. Add `tests/fixtures/online/<id>.expectations.json`. Tier 1 datasets need stricter rules
   (`columns_never_imputed`, `idempotent`, `max_duration_seconds`).
4. For tier 1 promotion: run `pytest tests/test_online_datasets.py --update-golden`, then
   full `pytest`.

Search/discover helpers:

```bash
python scripts/search_datasets.py --tag missing
python scripts/search_datasets.py --discover --limit 5
```

Live URL checks: `pytest -m online -m tier1 tests/test_online_datasets.py` (network required).

## Tier promotion criteria

Promote a tier-2 dataset to tier 1 when it has: stable URL (6+ months), meaningful
expectations beyond smoke tests, and a balanced golden snapshot that catches regressions.

## Reporting bugs

Please include a minimal DataFrame that reproduces the issue and the output of
`fd.profile(df)` — it usually contains exactly the information needed.
