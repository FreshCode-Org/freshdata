# Release & maintenance guide

This document describes how to cut a release of **freshdata**, publish to PyPI,
and maintain the project.

## Versioning

`freshdata` follows [Semantic Versioning](https://semver.org/). The version
lives in two places that must stay in sync:

- `pyproject.toml` → `[project] version`
- `src/freshdata/__init__.py` → `__version__`

Pre-1.0, minor versions (`0.x.0`) may include breaking changes; patch versions
(`0.x.y`) are backward-compatible fixes.

## Release checklist

1. **Green main** — `pytest`, `ruff check .`, `mypy src/freshdata`, and
   `mkdocs build --strict` all pass.
2. **Bump the version** in `pyproject.toml` and `src/freshdata/__init__.py`.
3. **Update `CHANGELOG.md`** — move `Unreleased` notes under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading.
4. **Commit & PR** — merge to `main`.
5. **Build & validate** locally (see below).
6. **Publish to TestPyPI**, smoke-test the install.
7. **Publish to PyPI** (or push the tag and let CI do it).
8. **Tag & GitHub release** — `git tag vX.Y.Z` and create the release with
   notes from the changelog.
9. **Verify** — `pip install freshdata-cleaner` in a clean environment imports
   (`import freshdata as fd`) and reports the new version; docs site updated.

## Build and validate

```bash
python -m pip install --upgrade build twine
rm -rf dist build
python -m build                 # builds sdist + wheel into dist/
twine check dist/*              # validates metadata + long-description rendering
```

## Publish

### Option A — automated (recommended)

Push a version tag; the `Release` workflow builds and publishes via PyPI
**Trusted Publishing** (OIDC, no stored token):

```bash
git tag v0.5.0
git push origin v0.5.0
```

One-time setup: add a trusted publisher at
<https://pypi.org/manage/project/freshdata-cleaner/settings/publishing/>
(workflow `release.yml`, environment `pypi`).

### Option B — manual with `twine`

```bash
# 1. TestPyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ freshdata-cleaner
python -c "import freshdata as fd; print(fd.__version__)"

# 2. Real PyPI
twine upload dist/*
```

Use a [PyPI API token](https://pypi.org/help/#apitoken) (username `__token__`).
Never commit tokens; prefer `~/.pypirc` or the `TWINE_PASSWORD` env var.

## Naming

The PyPI distribution is **`freshdata-cleaner`**; the import name is
**`freshdata`** (`import freshdata as fd`). Keep that split — do not rename the
distribution (the bare name `freshdata` is unavailable on PyPI).

## Maintenance guide

- **Branching** — work on feature branches; PR into `main`. CI must be green.
- **Dependencies** — keep the floor versions in `pyproject.toml` realistic;
  test against the lowest supported (`pandas>=1.5`, Python 3.9).
- **Coverage** — keep the gate at ≥ 93% (`--cov-fail-under`).
- **Docs** — update the relevant `docs/` page with any user-facing change; the
  docs site redeploys automatically on push to `main`.
- **Golden snapshots** — after intentional engine changes, refresh with
  `pytest tests/test_golden.py tests/test_online_datasets.py --update-golden`.
- **Security** — triage reports per `SECURITY.md`; release a patch promptly.
