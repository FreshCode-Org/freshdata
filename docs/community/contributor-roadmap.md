---
title: Contributor roadmap
description: >-
  A map of real, verified FreshData contribution opportunities grouped by difficulty —
  from 20-minute good first issues to advanced architecture projects.
keywords: freshdata contributor roadmap, good first issue, open source data cleaning tasks, contribute to python
---

# Contributor roadmap

This roadmap organizes active contribution opportunities by difficulty and time commitment, so you can find a task that fits your interests and schedule.

Every task listed below is a genuine, verified need in the project.

New here? Start with the [First Contribution Guide](../contributing/first-contribution.md) first!

!!! tip "Claim before you build"
    Comment on the issue to claim it before opening a PR so multiple contributors don't duplicate work. If you have an idea for a task not listed here, open a thread in [GitHub Discussions](https://github.com/FreshCode-Org/freshdata/discussions) under **Ideas** to align with maintainers.

---

## 🟢 Level 1: Quick Wins (20–30 minutes)

Ideal for your first PR. These tasks improve documentation, expand test fixtures, or add recipe examples without altering core decision algorithms.

1. **Recipe: DuckDB Parquet Export Recipe**
   * **Task**: Add a runnable example in `examples/integrations/` showing how to clean a partitioned Parquet dataset and write the clean output back to DuckDB.
   * **Labels**: `good first issue`, `examples`
   * **Files**: `examples/integrations/`

2. **Fixtures: Financial Ledger Sentinel Anomaly**
   * **Task**: Add a synthetic financial ledger fixture in `tests/fixtures/` with localized accounting conventions (e.g. `(1,250.00)` accounting negatives and `EUR` prefixes).
   * **Labels**: `good first issue`, `fixtures`
   * **Files**: `tests/fixtures/`, `benchmarks/fixtures/`

3. **Docs: Clarify Fallback Matrix Error Codes**
   * **Task**: Expand `docs/fallback-matrix.md` with explicit Python error code examples for `FallbackError` when `fallback_policy="error"` is triggered.
   * **Labels**: `good first issue`, `documentation`
   * **Files**: `docs/fallback-matrix.md`

4. **Testing: Missing Sentinel Edge Cases**
   * **Task**: Add unit tests in `tests/test_strings.py` covering unicode whitespace characters (e.g. non-breaking space `\u00A0`, zero-width space `\u200B`) during sentinel normalization.
   * **Labels**: `good first issue`, `testing`
   * **Files**: `tests/test_strings.py`

---

## 🟡 Level 2: Component Improvements (1–2 hours)

Ideal for contributors comfortable with Python, pandas, and regex who want to implement new features or validators.

5. **Field Validation: International Postal Code Validators**
   * **Task**: Implement context-aware regex patterns in `src/freshdata/fieldcheck.py` for UK (`SW1A 1AA`), Canadian (`K1A 0B1`), and German (`10115`) postal codes with unit tests.
   * **Labels**: `help wanted`, `validation`
   * **Files**: `src/freshdata/fieldcheck.py`, `tests/test_fieldcheck.py`

6. **Exporter: HTML Stakeholder Summary Card**
   * **Task**: Build an exporter plugin in `src/freshdata/plugins/` that exports `CleanReport` into a standalone, styled HTML card suitable for embedding in internal dashboards.
   * **Labels**: `help wanted`, `plugins`, `viz`
   * **Files**: `src/freshdata/render/`, `examples/plugins/`

7. **CLI: Add `--dry-run` Flag to FreshData CLI**
   * **Task**: Add a `--dry-run` option to `freshdata clean` CLI that executes `fd.plan()` and prints proposed transformations to stdout without writing the output file.
   * **Labels**: `help wanted`, `cli`
   * **Files**: `src/freshdata/enterprise/cli.py`, `tests/test_enterprise_cli.py`

8. **Polars: Native Windowed Deduplication Support**
   * **Task**: Enhance the Polars execution adapter to preserve partition order during duplicate resolution without round-tripping to pandas.
   * **Labels**: `help wanted`, `polars`, `performance`
   * **Files**: `src/freshdata/adapters/polars.py`, `tests/test_polars_adapter.py`

---

## 🔴 Level 3: Deep Architecture & Integrations (1+ days)

Substantial features for experienced engineers interested in out-of-core execution, domain packs, or ecosystem integrations.

9. **Domain Pack: Telecommunications (CDR / Network Logs)**
   * **Task**: Author a new telecom domain validator pack following `CONTRIBUTING_DOMAINS.md` to validate IMSI, IMEI, Call Detail Records (CDRs), and signal latency telemetry.
   * **Labels**: `help wanted`, `domains`
   * **Files**: `src/freshdata/domains/`, `docs/`

10. **Integration: Prefect 3.0 Task & Flow Decorator**
    * **Task**: Implement `freshdata.integrations.prefect` exposing a `@task` decorator that cleans task input/output frames and logs trust score telemetry to Prefect Cloud.
    * **Labels**: `help wanted`, `integrations`
    * **Files**: `src/freshdata/integrations/`, `tests/test_integrations/`

11. **Performance: SIMD-Accelerated Sentinel Matching**
    * **Task**: Profile and optimize string sentinel scanning in `src/freshdata/steps/sentinels.py` on 10M+ row frames using vectorized NumPy boolean operations.
    * **Labels**: `help wanted`, `performance`, `benchmarks`
    * **Files**: `src/freshdata/steps/`, `benchmarks/performance/`

---

## Contributor Recognition

Every contributor who lands a merged pull request is recognized:
* **Release Notes**: Listed under "Contributors" in the matching GitHub Release and `CHANGELOG.md`.
* **Social Spotlight**: Highlighted in our bi-weekly Friday community update on LinkedIn and Twitter/X.
* **All-Contributors**: Credited in the repository README.
