# Task 7 Report: CI-Safe Contracts, Large Workflow, Make Targets, and Docs

## Status

Complete. Task 7 integrates the existing Tasks 1–6 performance CLI without
changing production sources, public behavior, dependency metadata, or the protected
`.venv-qa/`. The requested commit subject is
`bench: integrate large performance investigation workflow`; the resulting hash is
reported to the parent after commit creation.

## Scope

- Added two small end-to-end CLI/report behavioral contracts and three repository
  integration contracts in `tests/performance/test_contracts.py`.
- Replaced the broad benchmark-results ignore rule with selective rules that keep
  raw case files/directories ignored and expose only direct
  `*-summary.json`/`*-report.md` performance evidence.
- Added `performance-ci`, `performance-baseline`, `performance-profile`, and
  `performance-report` Make targets using `$(PY)` and the active
  `python -m benchmarks.performance` CLI.
- Added a manual and weekly large workflow for the required 90-case matrix. The
  benchmark, analysis, and rendering steps retain their outcomes, artifact upload
  runs under `always()`, and a final gate restores failure status after evidence is
  uploaded.
- Added the resolved investigation architecture, supported versions/defaults,
  reproduction commands, evidence methodology, and lifecycle documentation.
- Linked the new page from the existing benchmark documentation and MkDocs nav;
  existing CleanBench and strategic-report content remains intact.

## TDD Record

### Baseline before Task 7 edits

Command:

```text
.venv-qa/bin/python -m pytest tests/performance --no-cov -W error -o addopts=''
```

Result: `181 passed in 19.15s`, with no warnings.

### RED

After adding only `tests/performance/test_contracts.py`, command:

```text
.venv-qa/bin/python -m pytest tests/performance/test_contracts.py -q --no-cov -W error -o addopts=''
```

Result: `2 passed, 3 failed in 10.67s`.

The two end-to-end runtime contracts passed against the completed Tasks 1–6
tooling. The intended integration failures were:

1. `test_performance_artifact_ignore_contract` — the repository still contained
   only the broad `benchmarks/results/` rule.
2. `test_make_targets_use_the_performance_cli` — `performance-ci` and the other
   Task 7 targets did not exist.
3. `test_large_workflow_is_scheduled_manual_and_preserves_failures` —
   `.github/workflows/performance-large.yml` did not exist.

### GREEN

After the minimal ignore, Make, and workflow implementation, the focused command
reported `5 passed in 9.59s`. After correcting the Make profile flags to match the
active CLI, the focused command again reported `5 passed in 10.36s`.

The brief's sample Make recipe used singular `--width`, `--config`, and
`--report-mode` flags. The existing Task 5/6 CLI actually exposes `--widths`,
`--configs`, and `--report-modes`; Task 7 uses those active interfaces and tests
their exact spellings.

## Workflow and Make Checks

- `make -n performance-ci performance-baseline performance-profile performance-report PY=.venv-qa/bin/python`
  exited 0 and printed all four expected active-CLI commands.
- PyYAML loaded `.github/workflows/performance-large.yml`; the parsed job timeout
  is 180 minutes.
- The workflow has only `workflow_dispatch` and one weekly `schedule`; it has no
  `pull_request` trigger.
- The workflow installs the existing `.[dev,bench,ml]` extras and passes the exact
  requested rows, widths, configs, report modes, warm-up, and repetition counts.
- `git check-ignore` confirmed raw direct and nested case JSON is ignored while
  `baseline-summary.json` and `baseline-report.md` are not ignored.

## Final Verification

Fresh verification immediately before report/commit:

```text
.venv-qa/bin/python -m pytest tests/performance --no-cov -W error -o addopts=''
186 passed in 26.41s

.venv-qa/bin/ruff check benchmarks/performance tests/performance
All checks passed!

.venv-qa/bin/ruff format --check benchmarks/performance tests/performance
22 files already formatted

/tmp/freshdata-task7-docs/bin/mkdocs build --strict
exit 0; documentation built in 2.53 seconds

git diff --check
exit 0
```

The strict docs build used a temporary `/tmp` environment containing the project's
already-declared docs extras because `.venv-qa/` intentionally does not contain
MkDocs and had to remain untouched. The build emitted no broken-link warning and
the new page is present in nav. It retained two informational notices for the
pre-existing internal plan/spec pages excluded from public nav, plus Material's
upstream MkDocs 2.0 announcement banner.

The Make target itself was also exercised:

```text
make performance-ci PY=.venv-qa/bin/python
186 passed
```

## Self-Review

- `git diff 8092337 -- src/freshdata pyproject.toml` is empty: no production source
  or runtime dependency changed.
- The workflow does not suppress benchmark failures: it permits evidence-producing
  follow-up steps, uploads the result directory even after failure, then exits 1 if
  benchmark, analysis, render, or upload did not succeed.
- New documentation makes no measured performance claim. The only numeric
  thresholds and compatibility ranges come from the approved design/project
  metadata.
- `mkdocs.yml` is included only to expose the new public documentation page.
- The untracked `.venv-qa/` was neither staged, edited, nor removed.

## Concerns

- The large 90-case workflow is intentionally not run locally as part of Task 7;
  it is the scheduled/manual evidence job and has the required 180-minute job cap.
- Material for MkDocs prints an upstream announcement banner even though strict
  build verification succeeds. This is external tool output, not a project warning
  or a documentation defect.
