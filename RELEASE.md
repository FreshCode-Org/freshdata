# Release & maintenance guide

This document describes how to cut a release of **freshdata**, publish to PyPI,
and maintain the project.

## Versioning

`freshdata` follows [Semantic Versioning](https://semver.org/). The version
lives in two places that must stay in sync:

- `pyproject.toml` → `[project] version`
- `src/freshdata/__init__.py` → `__version__`

From 1.0 the public API is stable: bump **MAJOR** (`X.0.0`) for breaking changes,
**MINOR** (`1.x.0`) for backward-compatible features, and **PATCH** (`1.0.y`) for
backward-compatible fixes.

## Release checklist

1. **Green main** — `pytest -m "not online and not large"`, `ruff check .`, `mypy src/freshdata`, and
   `mkdocs build --strict` all pass.
2. **Bump the version** in `pyproject.toml` and `src/freshdata/__init__.py`.
3. **Update `CHANGELOG.md`** — move `Unreleased` notes under a new
   `## [X.Y.Z] - YYYY-MM-DD` heading.
4. **Commit & PR** — merge to `main`.
5. **Build & validate** locally (see below).
6. **Publish to TestPyPI**, smoke-test the install — run the *TestPyPI dry run*
   workflow (see [TestPyPI dry run](#testpypi-dry-run-optional)).
7. **Publish to PyPI** (push the tag and let CI do it through trusted publishing).
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

### Option A — automated (required)

Push a version tag; the `Release` workflow runs a quality gate first
(lint, typecheck, `pytest -m "not online and not large"`, docs strict),
then builds and publishes via PyPI **Trusted Publishing** (OIDC, no stored token):

```bash
git tag vX.Y.Z   # must match the version in pyproject.toml / __version__
git push origin vX.Y.Z
```

One-time setup: add a trusted publisher at
<https://pypi.org/manage/project/freshdata-cleaner/settings/publishing/>
(workflow `release.yml`, environment `pypi`).

### TestPyPI dry run (optional)

Rehearse a release without touching real PyPI and without any stored token:
the manual **TestPyPI dry run** workflow (`.github/workflows/testpypi.yml`)
builds the current commit as a unique development version
(`X.Y.Z.dev<run><attempt>`, because TestPyPI never accepts a re-upload),
publishes it to TestPyPI through Trusted Publishing, then installs it from
TestPyPI in a clean environment, imports it and prints `fd.__version__`.

```bash
gh workflow run testpypi.yml --ref main
gh run watch "$(gh run list --workflow testpypi.yml --limit 1 --json databaseId --jq '.[0].databaseId')"
```

To verify a TestPyPI upload by hand (use the version from the run summary):

```bash
python -m venv /tmp/testpypi-smoke
/tmp/testpypi-smoke/bin/pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ "freshdata-cleaner==X.Y.Z.devN"
/tmp/testpypi-smoke/bin/python -c "import freshdata as fd; print(fd.__version__)"
```

One-time setup (a maintainer, once):

1. Add a pending trusted publisher at
   <https://test.pypi.org/manage/account/publishing/> for project
   `freshdata-cleaner`, owner `FreshCode-Org`, repository `freshdata`,
   workflow `testpypi.yml`, environment `testpypi`.
2. Create a `testpypi` environment under the repository's
   *Settings → Environments* (no secrets needed).

### Option B — manual with `twine` (fallback only)

```bash
# 1. TestPyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ freshdata-cleaner
python -c "import freshdata as fd; print(fd.__version__)"

# 2. Real PyPI
twine upload dist/*
```

Use a [PyPI API token](https://pypi.org/help/#apitoken) (username `__token__`) only
for emergency/manual fallback releases. Never commit tokens; prefer `~/.pypirc`
or the `TWINE_PASSWORD` env var.

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
