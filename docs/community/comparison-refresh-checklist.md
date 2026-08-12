---
title: Comparison refresh checklist
description: >-
  A repeatable, evidence-first checklist for keeping FreshData comparisons with
  pandas, PyJanitor, Great Expectations, ydata-profiling, cleanlab, and
  OpenRefine current.
keywords: freshdata comparison, contributor checklist, pandas, pyjanitor, great expectations, ydata profiling, cleanlab, openrefine
---

# Comparison refresh checklist

Use this checklist when a compared project publishes a significant release,
when FreshData changes a capability named in the comparison, or before a
FreshData release. The canonical detailed tables live on the
[comparison page](../comparison.md); do not copy them into the README.

The goal is not to make FreshData win more rows. It is to keep every verdict
current, reproducible, and useful to someone choosing a tool.

## 1. Record the review scope

- [ ] Record the review date and the FreshData commit under review in the pull
  request description.
- [ ] Record the latest stable version checked for every relevant project. Use
  a released version, not an unreleased documentation preview.
- [ ] Read the release notes since the version used by the previous review.
- [ ] List the rows affected by upstream or FreshData changes. Do not rewrite
  unrelated rows just to make the table sound more uniform.

Check these primary sources first:

| Project | Capability source | Change source | Surfaces to recheck |
|---|---|---|---|
| pandas | [User guide](https://pandas.pydata.org/docs/user_guide/) | [Release notes](https://pandas.pydata.org/docs/whatsnew/index.html) | missing data, duplicates, scaling, and individual DataFrame operations |
| PyJanitor | [Functions API](https://pyjanitor-devs.github.io/pyjanitor/api/functions/) | [GitHub releases](https://github.com/pyjanitor-devs/pyjanitor/releases) | convenience verbs, method chaining, domain modules, and pandas-only assumptions |
| Great Expectations | [GX Core documentation](https://docs.greatexpectations.io/docs/core/introduction/) | [GitHub releases](https://github.com/great-expectations/great_expectations/releases) | expectations, checkpoints, Data Docs, execution backends, and repair boundaries |
| ydata-profiling | [Documentation](https://docs.profiling.ydata.ai/latest/) | [GitHub releases](https://github.com/ydataai/ydata-profiling/releases) | profiling, visual reports, dataset comparison, integrations, and whether data is mutated |
| cleanlab | [Stable documentation](https://docs.cleanlab.ai/stable/) | [GitHub releases](https://github.com/cleanlab/cleanlab/releases) | label issues, Datalab issue types, model-output requirements, and repair workflows |
| OpenRefine | [User manual](https://openrefine.org/docs) | [GitHub releases](https://github.com/OpenRefine/OpenRefine/releases) | interactive curation, reconciliation, operation-history replay, clustering, and scale |

Official documentation, API references, repositories, and release notes are
primary sources for upstream capabilities. Search results, vendor blog posts,
and third-party comparisons can help locate a claim, but must not be its only
support.

## 2. Audit every affected row

For each row that is added or reviewed:

- [ ] Name one narrow criterion. Split rows that combine independently
  testable claims, such as runtime and peak memory.
- [ ] Reproduce the current upstream behavior when the verdict depends on
  runtime behavior rather than a documented API boundary.
- [ ] Give the row an explicit **win**, **loss**, **tie**, or **unproven**
  verdict using the rubric below.
- [ ] Put a source or a short, checkable rationale in the row's evidence cell.
  A bare assertion such as "tool X cannot do this" is not sufficient.
- [ ] Link an upstream claim to the project's official documentation, API, or
  release notes. Prefer a versioned page when behavior differs by version.
- [ ] Link a FreshData capability claim to a specific test, implementation,
  benchmark artifact, or documentation contract in this repository.
- [ ] For performance or quality claims, record versions, inputs, commands,
  environment, repeated measurements, and the committed raw result. Do not
  infer a benchmark result from documentation.
- [ ] Compare like with like: the same workload, output contract, materialized
  or lazy mode, and optional dependencies. State unavoidable differences.
- [ ] Check that every repository-relative path and external link still works.
- [ ] Remove or downgrade a verdict when its evidence is stale or cannot be
  reproduced. **Unproven** is an acceptable result.

Use this verdict rubric consistently:

| Verdict | Required support |
|---|---|
| **win** | FreshData satisfies the criterion and the named alternative does not, supported by primary sources or a reproducible head-to-head result. |
| **loss** | The alternative satisfies the criterion better or FreshData lacks the capability; state the limitation plainly. |
| **tie** | Both satisfy the criterion at the stated scope. Do not use a tie to hide meaningful trade-offs. |
| **unproven** | Evidence is absent, stale, incomparable, or inconclusive. Do not turn expectations into a verdict. |

## 3. Keep evidence reproducible

- [ ] Put reusable competitor adapters in `benchmarks/baselines/`, not only in
  a pull-request comment or an uncommitted notebook.
- [ ] Commit evidence to the tracked destination for its harness:
  `benchmarks/cleanbench/results/` for CleanBench, the relevant accepted
  `baseline.json` for Gauntlet or TruthBench, or compact
  `benchmarks/results/performance/*-summary.json` plus `*-report.md` for the
  performance harness. Generic `benchmarks/results/<run>/`, Gauntlet run
  output, TruthBench `latest.*`, and raw performance cases are intentionally
  ignored; keep them as CI artifacts and link the run. Do not force-add an
  ignored raw result as permanent evidence.
- [ ] Use committed fixtures or document how generated input is seeded.
- [ ] Confirm that the comparison text matches the stored results; do not copy
  rounded numbers from an older run.
- [ ] Note optional extras, paid services, remote APIs, GPUs, or model outputs
  required by either tool.
- [ ] Keep cleaning, profiling, validation, and label-quality detection as
  separate categories. Similar names do not imply equivalent behavior.

## 4. Protect the README and final wording

- [ ] Keep the detailed matrix in `docs/comparison.md` as the single source of
  truth.
- [ ] If the README mentions competitors, use a short neutral summary and link
  to the comparison page. Do not duplicate the detailed tables there.
- [ ] Re-read the introduction, verdict, and "cannot claim" section after
  changing rows so they do not overstate the updated evidence.
- [ ] Use the projects' current names and capitalization: pandas, PyJanitor,
  Great Expectations (GX), ydata-profiling, cleanlab, and OpenRefine.
- [ ] Run `mkdocs build --strict` and `ruff check .` before opening the pull
  request.

In the pull request, summarize which upstream versions were checked, which
rows changed, what evidence was added or retired, and any claims deliberately
left **unproven**.
