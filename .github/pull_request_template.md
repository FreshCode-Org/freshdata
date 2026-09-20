## What changed?

<!-- Describe what was modified, added, or removed. -->

## Why?

<!-- Explain the problem this solves, the motivation, or link the related issue (Fixes #...) -->

Fixes #

## How was it tested?

<!-- Detail unit tests, regressions, or benchmark commands executed locally. -->

- [ ] New unit tests added in `tests/`
- [ ] Ran fast CI lane locally: `pytest -m "not online and not large"`
- [ ] Ran linting and type checks: `ruff check .` and `mypy src/freshdata`

## Any performance impact?

<!-- Will this change affect wall-clock runtime, memory allocation, or startup time? If yes, provide timings. -->

- [ ] None / negligible
- [ ] Measured with `benchmarks/bench.py` (details below):

## Any compatibility concerns?

<!-- Does this change public API signatures, default behavior, or supported Python/pandas versions? -->

- [ ] None / fully backward-compatible
- [ ] Deprecation or behavior change documented below:

## Documentation updated?

<!-- If user-facing behavior changed, did you update docs/, examples/, or CHANGELOG.md? -->

- [ ] Documentation updated in `docs/`
- [ ] Examples verified or updated
- [ ] Note added under `[Unreleased]` in `CHANGELOG.md`
