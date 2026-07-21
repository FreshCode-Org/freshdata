---
title: Contributor roadmap
description: >-
  A map of freshdata contribution opportunities grouped by difficulty — from
  good first issues and docs tasks to bug fixes, features, and architecture work.
keywords: freshdata contributor roadmap, good first issue, open source data cleaning tasks
---

# Contributor roadmap

This page groups the kinds of work freshdata needs by difficulty, so you can
find something that matches the time and depth you want to invest. Every tier
links to *live* issues where they exist — always check the
[issue tracker](https://github.com/FreshCode-Org/freshdata/issues) for the
current state, since issues get claimed and closed.

New here? Read the [first-PR guide](first-pr.md) first, then come back to pick a
tier. For the code layout behind these tasks, see
[ARCHITECTURE.md](https://github.com/FreshCode-Org/freshdata/blob/main/ARCHITECTURE.md).

!!! tip "Claim before you build"
    Comment on an issue to claim it before opening a PR, so two people don't do
    the same work. No issue for what you want to do? Open one (or start a
    [Discussion](https://github.com/FreshCode-Org/freshdata/discussions)) so a
    maintainer can confirm the direction before you invest time.

## 🟢 Good first issues — small, self-contained, reviewable without touching the engine

Best for your first PR. These change docs, examples, or an isolated helper, so
review does not depend on understanding the whole cleaning pipeline.

- **Recipes and examples** — short, runnable scripts showing freshdata next to
  another tool. Open examples:
  [pyjanitor interop (#9)](https://github.com/FreshCode-Org/freshdata/issues/9),
  [Great Expectations recipe (#8)](https://github.com/FreshCode-Org/freshdata/issues/8),
  [ydata-profiling comparison (#7)](https://github.com/FreshCode-Org/freshdata/issues/7).
  Files live in [`examples/`](https://github.com/FreshCode-Org/freshdata/tree/main/examples).
- **Small isolated cleanups** —
  [de-duplicate `_has_outliers` (#33)](https://github.com/FreshCode-Org/freshdata/issues/33)
  and
  [fix boolean coercion in `_coerce_series` (#31)](https://github.com/FreshCode-Org/freshdata/issues/31)
  are both labeled `good first issue` and name the exact function and file.

**Skills:** Python, pandas basics. **You'll touch:** `examples/`, `docs/`, or one
named module + its test.

## 📖 Documentation tasks

Docs changes are the fastest way to make a real, mergeable contribution.

- Fix anything unclear or out of date in the
  [docs site](https://freshcode-org.github.io/freshdata/) — pages live in
  [`docs/`](https://github.com/FreshCode-Org/freshdata/tree/main/docs).
- [Comparison-table refresh checklist (#2)](https://github.com/FreshCode-Org/freshdata/issues/2).
- Add missing docstrings, or clarify a confusing feature guide.
- Spotted a docs bug? File it with the **Documentation** issue template.

**Skills:** clear writing, Markdown, `mkdocs serve` to preview. **You'll touch:**
`docs/`, module docstrings.

## 🧪 Testing tasks

freshdata enforces a 93% coverage gate, so tests are always welcome — especially
"does not fire when it shouldn't" cases and fixtures.

- Add [benchmark preservation / trust-monotonicity / export tests (#82)](https://github.com/FreshCode-Org/freshdata/issues/82).
- Add synthetic fixture generators:
  [CRM & finance (#77)](https://github.com/FreshCode-Org/freshdata/issues/77),
  [event-log CDC & wide-schema (#78)](https://github.com/FreshCode-Org/freshdata/issues/78),
  [provenance & gold-label repair fixtures (#79)](https://github.com/FreshCode-Org/freshdata/issues/79).
- Add an [online dataset fixture](https://github.com/FreshCode-Org/freshdata/blob/main/CONTRIBUTING.md#adding-an-online-dataset-fixture)
  to widen real-world coverage.

**Skills:** pytest, pandas, an eye for edge cases. **You'll touch:** `tests/`,
`tests/fixtures/`.

## 🐛 Bug fixes

Reproducible bugs with a named symptom — a good step up once you've landed a
first PR.

- [Polars streaming disabled by default dedupe (#53)](https://github.com/FreshCode-Org/freshdata/issues/53)
  and
  [DuckDB `fetchdf()` defeats out-of-core execution (#52)](https://github.com/FreshCode-Org/freshdata/issues/52)
  — both labeled `bug` + `benchmarks`, in `execution/` / `streaming/`.
- [`explain_clean` computes `build_contexts` twice (#32)](https://github.com/FreshCode-Org/freshdata/issues/32).

Reproduce first, add a failing test, then fix. See the "Prove-it" habit in the
[first-PR guide](first-pr.md).

**Skills:** pandas debugging, the affected subpackage. **You'll touch:** the named
module + a regression test.

## 🚀 Feature development

Mid-sized, maintainer-guided additions. Confirm the API in the issue before
building.

- [First-class JSON export for `CleanReport` (#21)](https://github.com/FreshCode-Org/freshdata/issues/21)
  (`help wanted`, mid level).
- [`fd.clean_csv()` one-line CSV cleanup (#19)](https://github.com/FreshCode-Org/freshdata/issues/19)
  (`help wanted`, junior level).
- A **new domain validator pack** — the most self-contained feature path. Follow
  [CONTRIBUTING_DOMAINS.md](https://github.com/FreshCode-Org/freshdata/blob/main/CONTRIBUTING_DOMAINS.md).

**Skills:** the relevant subpackage, API design sense. **You'll touch:** `api.py`,
a subpackage, docs, and tests.

## 🏗️ Advanced architecture work

Deep changes to the engine, execution backends, or CI. Discuss the design first.

- CI / release automation:
  [TestPyPI smoke-install (#6)](https://github.com/FreshCode-Org/freshdata/issues/6),
  [public benchmark refresh workflow (#4)](https://github.com/FreshCode-Org/freshdata/issues/4),
  [notebook smoke-test workflow (#3)](https://github.com/FreshCode-Org/freshdata/issues/3).
- [Benchmark harness CLI with standardized metrics (#80)](https://github.com/FreshCode-Org/freshdata/issues/80).
- Out-of-core execution correctness across Polars / DuckDB / Spark backends
  (`execution/`), where the two open backend bugs above also live.

**Skills:** the engine or execution internals, CI, performance. **You'll touch:**
`engine/`, `execution/`, `.github/workflows/`.

## Where to ask

Not sure which tier fits, or want a maintainer to confirm an approach before you
start? Open a thread in
[Discussions](https://github.com/FreshCode-Org/freshdata/discussions) — that is
the right place for "how should I…?" and "is this worth doing?" questions.
