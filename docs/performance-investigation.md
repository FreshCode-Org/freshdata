---
title: Performance investigation
description: >-
  Reproducible FreshData performance measurement, profiling, analysis, and
  evidence-reporting methodology.
keywords: freshdata performance, scalability, profiling, benchmark methodology
---

# Performance investigation

## 1. Executive summary

Phase 1 baseline capture is complete; no production optimization has been
implemented. All **132/132 FreshData timing cases completed** with no FreshData
timing failure, timeout, or OOM. Coverage is 10k, 100k, 500k, and 1M rows;
narrow, medium, and wide mixed frames; five approved configurations
(`default`, `conservative`, `representation_off`, `statistical_off`, and
`explicit`); report on/off; plus 12 family cases at 100k/medium across six
dataset families and both report modes.

The largest measured baseline runtime is 1M/wide/default/report=false at an
exact median of `112.11013312500017` seconds. The highest measured peak-RSS
delta is 1M/wide/explicit/report=false at `2,028,158,976` bytes. Six profiles
completed; the MissForest profile is missing/incomplete after an interrupted
approximately two-hour CPU-bound run. The 48/48 **named pandas component
operations** produced 41 completed results and seven failed
`numeric_median_fill` operations because fractional medians could not fill
nullable `Int64` columns; there were no component timeouts or OOMs. These
components are not full-pipeline FreshData equivalents.

The authoritative evidence is
[`benchmarks/results/performance/baseline-summary.json`](https://github.com/FreshCode-Org/freshdata/blob/main/benchmarks/results/performance/baseline-summary.json),
with every generated row and profiler summary in the
[`baseline-report.md`](https://github.com/FreshCode-Org/freshdata/blob/main/benchmarks/results/performance/baseline-report.md)
view. Production `src/freshdata` is byte-for-byte unchanged relative to
`6f6c2fe` according to `git diff`.

### Architecture and execution flow

```text
fd.clean
  -> validate and normalize API inputs
  -> select pandas or native path and initialize CleanReport
  -> representation repair (columns, strings, sentinels, rows/columns,
       dtypes, constants, duplicates, optional semantic repair)
  -> build column contexts and shared engine artifacts
  -> automatic and explicit missing-value/outlier decisions
  -> optional memory optimization and protected-column checks
  -> finalize report and wrap/convert the result
```

The pandas path in `src/freshdata/cleaner.py` remains the behavioral reference;
context and shared artifacts live in `src/freshdata/engine/context.py` and
`src/freshdata/engine/cache.py`, and representation operations live under
`src/freshdata/steps/`. Native paths implement documented subsets and fall back
to pandas as required. `return_report=False` still constructs the embedded
report and populates `Cleaner.report_`; future changes must preserve that
contract.

### Supported environment and provenance

The supported Python 3.9–3.13, pandas `>=1.5,<3`, and NumPy `>=1.21` ranges are
unchanged. Measurements were captured on one macOS 15.5 arm64 host with 8
logical/8 physical CPUs and 17,179,869,184 bytes RAM, using Python 3.12.13,
pandas 2.3.3, NumPy 2.4.6, and FreshData 1.1.1. Evidence records tooling commit
`27ee7b653a1318ac66692df2ba9c7813a674e8aa` and `git_dirty=true` because
`.venv-qa/` was untracked; the production-source identity gate against
`6f6c2fe` remained clean.

## 2. Reproduction commands

Install the development, benchmark, and optional ML dependencies first:

```bash
pip install -e ".[dev,bench,ml]"
```

The timing evidence was captured with these selectors (500k and 1M widths were
run as separate commands with the shown width substituted):

```bash
.venv-qa/bin/python -m benchmarks.performance run --rows 10000 --widths narrow,medium,wide --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 900 --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance run --rows 100000 --widths narrow,medium,wide --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 1800 --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance run --rows 100000 --widths medium --dataset-types numeric,categorical,string,nullable,datetime,high_cardinality --configs default --report-modes false,true --warmups 1 --repetitions 5 --timeout 1800 --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance run --rows 500000 --widths narrow --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 3600 --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance run --rows 1000000 --widths narrow --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 7200 --output benchmarks/results/performance/baseline
```

Repeat the last two commands with `--widths medium` and `--widths wide`. The six
completed profiles used the following selectors; the same form with
`--rows 10000 --widths medium --configs missforest --report-modes true` was the
interrupted seventh command.

```bash
.venv-qa/bin/python -m benchmarks.performance profile --rows 10000 --widths wide --configs default --report-modes true --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs default --report-modes true --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance profile --rows 500000 --widths medium --configs default --report-modes false --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance profile --rows 1000000 --widths narrow --configs default --report-modes true --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs aggressive --report-modes true --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs semantic --report-modes true --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance profile --rows 10000 --widths medium --configs missforest --report-modes true --output benchmarks/results/performance/baseline
```

The last command is the incomplete MissForest attempt described in section 4;
the other six commands produced completed profile artifacts.

```bash
.venv-qa/bin/python -m benchmarks.performance baseline --rows 10000,100000,500000,1000000 --widths narrow,medium,wide --dataset-types mixed --baselines shallow_copy,numeric_median_fill,duplicates,null_counts --warmups 1 --repetitions 5 --timeout 3600 --output benchmarks/results/performance/baseline
.venv-qa/bin/python -m benchmarks.performance analyze --input benchmarks/results/performance/baseline --output benchmarks/results/performance/baseline-summary.json
.venv-qa/bin/python -m benchmarks.performance render --input benchmarks/results/performance/baseline-summary.json --output benchmarks/results/performance/baseline-report.md
```

### Measurement methodology and evidence lifecycle

Reportable cases use one warm-up and five measured repetitions. Timing and
memory are measured separately in isolated workers. JSON records the exact case,
samples, median, variability, throughput, peak RSS, Python allocation peak,
fingerprints, environment, status, and errors. The JSON summary is authoritative;
generated Markdown is its view. Function and allocation rankings are capped at
the top 100 and are therefore not exhaustive.

The analyzer reconciles four same-ID plain/profile companion pairs by retaining
the plain timing record and attaching its profile. The regenerated summary has
134 unique FreshData cases: 132 timing cases and two profile-only cases. This
prevents separately measured profile timings from becoming duplicate benchmark
rows while preserving all 186 raw reproduction commands.

A later optimization qualifies only if the relevant median runtime or peak
memory improves by at least 10%, exceeds twice the larger observed coefficient
of variation, preserves behavioral equivalence, and causes no meaningful primary
workload regression. Failures and incomplete evidence remain visible rather than
being discarded.

## 3. Baseline benchmark table

This compact mixed-frame slice shows default scaling and the largest configuration
contrast. Medians and CV coefficients are rounded to six decimal places from the
authoritative JSON; peak-RSS byte counts are exact. See the
[full generated report](https://github.com/FreshCode-Org/freshdata/blob/main/benchmarks/results/performance/baseline-report.md)
for every generated row: all 134 reconciled FreshData cases (including all 132
timing cases), 41 completed named component rows, and seven failed components.

| Rows | Width | Config | Report | Median (s) | CV | Peak RSS (bytes) |
| ---: | :--- | :--- | :---: | ---: | ---: | ---: |
| 10,000 | medium | default | false | 0.428086 | 0.004192 | 5,996,544 |
| 100,000 | medium | default | false | 2.766123 | 0.005914 | 32,522,240 |
| 500,000 | medium | default | false | 20.539099 | 0.022993 | 59,195,392 |
| 1,000,000 | narrow | default | false | 8.902830 | 0.094170 | 157,351,936 |
| 1,000,000 | medium | default | false | 27.221341 | 0.011226 | 212,025,344 |
| 1,000,000 | wide | default | false | 112.110133 | 0.013212 | 1,400,422,400 |
| 1,000,000 | wide | default | true | 110.060040 | 0.001362 | 1,016,152,064 |
| 1,000,000 | wide | conservative | false | 67.080104 | 0.017137 | 2,014,887,936 |
| 1,000,000 | wide | representation_off | false | 6.675709 | 0.006014 | 3,293,184 |
| 1,000,000 | wide | statistical_off | false | 65.251788 | 0.012630 | 1,159,512,064 |
| 1,000,000 | wide | explicit | false | 65.768450 | 0.007518 | 2,028,158,976 |

The exact largest median is `112.11013312500017` seconds, not the rounded table
display. Configuration contrasts describe different behavior and are not
optimization before/after comparisons.

## 4. Profiling findings with functions, files, and lines

Six profile artifacts completed. Representative cumulative hot paths are:

| Profile | Profiler total (s) | Exact FreshData evidence |
| :--- | ---: | :--- |
| 10k/wide/default/report=true | 18.445463412 | `src/freshdata/steps/dtypes.py:332 fix_dtypes`: 1 call, 11.888526209 s cumulative; `:297 suggest_conversion`: 42 calls, 11.883506043 s |
| 100k/medium/default/report=true | 16.006843291 | `src/freshdata/steps/dtypes.py:332 fix_dtypes`: 1 call, 4.828396791 s cumulative |
| 500k/medium/default/report=false | 49.319412332 | `src/freshdata/steps/duplicates.py:83 drop_duplicate_rows`: 1 call, 18.035868375 s cumulative; `:70 _filter_rows`: 0.196458333 s self |
| 1M/narrow/default/report=true | 31.422901707 | `src/freshdata/steps/duplicates.py:83 drop_duplicate_rows`: 1 call, 17.996924333 s cumulative; `:70 _filter_rows`: 0.26744025 s self |
| 100k/medium/aggressive/report=true | 15.655626045 | `src/freshdata/steps/dtypes.py:332 fix_dtypes`: 1 call, 4.873163041 s cumulative |
| 100k/medium/semantic/report=true | 22.015065380 | `src/freshdata/semantic/apply.py:192 run_semantic`: 1 call, 6.300034834 s cumulative; `src/freshdata/semantic/experts.py:146 is_plain_number`: 0.718468278 s self |

Engine-context evidence in the 10k/wide/default profile includes
`src/freshdata/engine/cache.py:21 build_engine_cache` at 1.376872959 seconds
cumulative, `src/freshdata/engine/context.py:282 build_contexts` at 1.308761792
seconds, `src/freshdata/engine/context.py:231 build_context` at 1.308280540
seconds over 128 calls, and `src/freshdata/engine/context.py:79 _mode_ratio` at
0.483927833 seconds. Representative FreshData allocation entries include
`src/freshdata/semantic/experts.py:565` (952 bytes/17 allocations),
`src/freshdata/engine/model_select.py:154` (544/8),
`src/freshdata/steps/outliers.py:119` (516/8), and
`src/freshdata/engine/context.py:104` (480/7). These line rankings are top-100
snapshots, not exhaustive allocation totals or peak-RSS measurements.

The missing 10k/medium/MissForest command ran CPU-bound for approximately two
hours, was last observed at 100% CPU and roughly 650 MiB RSS, and was interrupted
under the quick-finish directive before producing JSON. No MissForest function,
allocation, stage, operation-count, or semantic conclusion is inferred.

## 5. Confirmed root causes

No cause is confirmed by profiling alone. The only `candidate` decision is the
case-specific semantic `optional_ml_overhead` result for 100k/medium/semantic/
report=true: stage fraction `0.13104697297973694`
(`13.104697297973694%`), one observed call, and
`src/freshdata/semantic/apply.py:192 run_semantic` at `6.300034834000001`
seconds cumulative.

Here, `candidate` means an optimization-plan candidate, not a confirmed
production root cause. It still requires a targeted acceptance benchmark and an
output/report equivalence gate before any causal or improvement claim.

A separate evidence-tooling cause was confirmed and fixed before this report:
the analyzer previously emitted four plain/profile companion pairs twice because
it had no reconciliation step. The regenerated compact evidence now contains one
canonical row per case. This was a benchmark-reporting defect, not a FreshData
runtime cause or production optimization.

## 6. Rejected hypotheses

Every other decision is `rejected` or `insufficient_evidence` across the six
completed profiles:

| Hypothesis | Result across six profiles |
| :--- | :--- |
| `copy_pressure` | rejected: 6 |
| `dtype_conversion_pressure` | rejected: 6 |
| `repeated_uniqueness_scans` | rejected: 6 |
| `report_finalization_overhead` | rejected: 6 |
| `backend_conversion_overhead` | insufficient evidence: 6 |
| `unnecessary_correlation` | insufficient evidence: 6 |
| `repeated_null_scans` | rejected: 5; insufficient evidence: 1 |
| `optional_ml_overhead` | candidate: 1; insufficient evidence: 5 |

Correlation remains insufficient: `dataframe.corr=1` was observed in each
completed profile, but the required exact correlation function did not appear in
the capped function evidence. Operation counts alone do not establish a hot path.

## 7. Files and functions changed

not yet applicable: no production optimization has been implemented

The Phase 1 work changes this evidence document only; production
`src/freshdata` remains unchanged relative to `6f6c2fe`.

## 8. Explanation of every optimization

not yet applicable: no production optimization has been implemented

Optimization selection and rationale are pending later-phase work, not unknown
Phase 1 evidence.

## 9. Behavioral compatibility analysis

not yet applicable: no production optimization has been implemented

Behavioral comparison is pending a concrete change. Existing output and report
fingerprints establish the future equivalence gate but do not constitute a
before/after result.

## 10. Tests added or changed

No production optimization was implemented. Task 9.2 added the focused
development-only repetition-discovery fixture in
`tests/performance/test_semantic_probe.py`; it validates within-build reuse
signals without changing FreshData behavior.

## 11. Before-and-after benchmark table

not yet applicable: no production optimization has been implemented

### Task 9.2 semantic repetition discovery

The development-only semantic repetition probe was run against the
representative `DatasetSpec(rows=100000, width="medium", seed=42,
dataset_type="mixed")` frame with `CleanConfig(semantic_mode="assist",
verbose=False)`. The probe commit was `306a01df6a6c23329895f89083caab67c92707b4`.
At measurement start, the worktree was dirty only because of the untracked
`.venv-qa/` directory and the focused test edit (`M
tests/performance/test_semantic_probe.py`, `?? .venv-qa/`); no production
source was modified. The host was macOS 15.5 arm64 (Darwin 24.5.0), Python
3.12.13, pandas 2.3.3, NumPy 2.4.6, and pytest 9.1.1.

The focused fixture verification was:

```text
.venv-qa/bin/python -m pytest tests/performance/test_semantic_probe.py::test_probe_reports_reuse_for_repeated_context_values -q --no-cov
.[100%]
```

The exact command in the brief (`print(result.to_json())`) cannot run because
`SemanticProbeBuild` intentionally has no `to_json()` API. It failed with
`AttributeError: 'SemanticProbeBuild' object has no attribute 'to_json'`. The
following serialization-only equivalent was run without changing the probe or
production API and produced valid JSON:

```bash
PYTHONPATH=src .venv-qa/bin/python -c 'import json; from benchmarks.performance.datasets import DatasetSpec, make_mixed_frame; from benchmarks.performance.semantic_probe import probe_context_build; from freshdata.config import CleanConfig; frame = make_mixed_frame(DatasetSpec(rows=100000, width="medium", seed=42, dataset_type="mixed")); _, result = probe_context_build(frame, CleanConfig(semantic_mode="assist", verbose=False)); print(json.dumps({"total_calls": result.total_calls, "total_eligible_calls": result.total_eligible_calls, "total_bypassed_calls": result.total_bypassed_calls, "total_theoretical_hits": result.total_theoretical_hits, "by_operation": {name: {"total_calls": item.total_calls, "eligible_calls": item.eligible_calls, "bypassed_calls": item.bypassed_calls, "unique_keys": item.unique_keys, "theoretical_hits": item.theoretical_hits, "hit_rate": item.hit_rate} for name, item in result.by_operation.items()}}, sort_keys=True))'
```

The successful single-build JSON reported `639307` total calls,
`497842` eligible calls, `141465` bypassed calls, and `52725` theoretical
within-build hits:

| Operation | Total calls | Eligible | Bypassed | Unique keys | Theoretical hits | Hit rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `is_plain_number` | 192,376 | 50,911 | 141,465 | 50,902 | 9 | 0.000177 |
| `parse_number_words` | 50,911 | 50,911 | 0 | 50,902 | 9 | 0.000177 |
| `parse_boolean` | 192,376 | 192,376 | 0 | 139,705 | 52,671 | 0.273792 |
| `parse_currency` | 50,911 | 50,911 | 0 | 50,902 | 9 | 0.000177 |
| `parse_unit` | 50,911 | 50,911 | 0 | 50,902 | 9 | 0.000177 |
| `email_value` | 50,911 | 50,911 | 0 | 50,902 | 9 | 0.000177 |
| `looks_like_date_value` | 50,911 | 50,911 | 0 | 50,902 | 9 | 0.000177 |

Only calls within this one build count. The repeated `parse_boolean` values are
real eligible hits, but they are not by themselves evidence of a worthwhile
optimization. A temporary timing wrapper around the same single build measured
1.575182417 seconds total and only 0.394354 seconds across all seven helper
operations. Per-operation helper time/call multiplied by each operation's
within-build hit count estimates 0.016639853 seconds of removable repeat work
(1.06% of context-build time, and far below the 10% end-to-end threshold).

Decision: **`rejected_no_material_within_build_reuse`**. The measured repeat
work does not make a 10% end-to-end improvement plausible, so Task 3 is
skipped. No production optimization was implemented; `src/freshdata` remains
unchanged. Warm-up, measurement, memory, profiling, and cross-build
repetitions were not counted toward this decision.

The table in section 3 is baseline-only. A later phase must rerun the same cases
before presenting an optimization comparison.

### Phase 2 acceptance outcome (Task 9.5)

The Task 2 decision is **`rejected_no_material_within_build_reuse`**. The
discovery evidence above estimates only `0.016639853` seconds of removable
repeat helper work, or `1.06%` of context-build time, which cannot plausibly
meet the required 10% end-to-end improvement threshold. Consequently, the
conditional Task 3 production optimization was not implemented.

Per the brief's rejected branch, the serial before/after benchmark commands
and the confirming profile command were intentionally skipped; the Task 2
discovery evidence is retained as the authoritative Phase 2 measurement. There
is therefore no before/after timing, RSS, allocation-peak, output-fingerprint,
report-fingerprint, result-type, or profile delta to claim. The outcome is an
evidence-backed rejection, not a projected performance result. `src/freshdata`
remains byte-for-byte unchanged relative to `6f6c2fe`.

The final acceptance environment was macOS 15.5 arm64, Python 3.12.13, pandas
2.3.3, and NumPy 2.4.6. At verification the only worktree dirt was the
pre-existing untracked `.venv-qa/` directory; no raw benchmark output or
`.superpowers/sdd` report is part of the production change.

## 12. Peak-memory comparison

No optimization before/after memory comparison exists. This baseline-only 1M/
wide mixed-frame contrast reports exact peak-RSS deltas from separate workers:

| Config | Report=false (bytes) | Report=true (bytes) |
| :--- | ---: | ---: |
| default | 1,400,422,400 | 1,016,152,064 |
| conservative | 2,014,887,936 | 1,479,458,816 |
| representation_off | 3,293,184 | 245,760 |
| statistical_off | 1,159,512,064 | 1,248,542,720 |
| explicit | **2,028,158,976** | 1,702,543,360 |

Peak RSS is a process delta, not total resident size. The very low
`representation_off` deltas and report-mode reversals require cautious
interpretation.

For named component-operation context only, the largest component peak RSS was
1M/wide `numeric_median_fill` at 493,223,936 bytes (470.375 MiB). It is not a
full-pipeline equivalent and is not an optimization comparison.

## 13. Remaining bottlenecks

- Wide/default runtime scales to the largest measured median, while wide
  statistics-enabled configurations also carry the largest RSS deltas.
- Dtype repair and duplicate handling appear as high cumulative paths in
  representative profiles, but their defined hypothesis decisions do not confirm
  production root causes.
- The 1M/wide `duplicates` component was the slowest completed named operation
  at a 13.8993894-second median; this supports investigation of duplicate handling
  but is not a full-pipeline FreshData measurement.
- Semantic assist has the sole case-specific optimization-plan candidate.
- Engine-context and correlation work is observed, but the evidence threshold for
  a correlation cause is not met.
- MissForest remains unprofiled, so its internal bottleneck is unknown rather
  than merely pending classification.

## 14. Backend and out-of-core recommendations

not yet applicable: no production optimization has been implemented

Backend and out-of-core recommendations are pending evidence-driven Phase 2
design; Phase 1 does not justify one.

## 15. Documentation corrections

not yet applicable: no production optimization has been implemented

This page now records the completed baseline instead of describing it as in
progress, but no optimization-specific user-documentation correction is claimed.

## 16. Risks and trade-offs

- Report=true is not consistently slower: examples include 1M/wide/default
  (`110.0600397090002` versus `112.11013312500017` seconds) and several other
  report-mode inversions. These pairs must not be read as isolated report-cost
  estimates.
- `representation_off` has unusually low RSS deltas, including 245,760 bytes for
  1M/wide/report=true; this may reflect the selected path or RSS-delta accounting,
  not a stable total-memory footprint.
- Eleven completed component measurements have CV above 20%; the highest is
  82.11% for 100k/narrow `shallow_copy`. Sub-millisecond components are especially
  noisy.
- The seven failed named components are 10k/all widths, 100k/medium and wide, and
  500k/medium and wide `numeric_median_fill` cases. Fractional medians (`5082.5`,
  `4989.5`, or `4995.5`) are invalid for nullable `Int64`; failed cases have no
  timing or memory metric.
- The absent MissForest JSON, top-100 profiler caps, one-host environment,
  unavailable durable shell timing/exit details for completed profile commands,
  and dirty provenance flag limit generalization.

## 17. Exact verification commands and results

```bash
.venv-qa/bin/python -m pytest tests/test_semantic_cleaning.py tests/test_semantic_backends.py tests/test_execution/test_native_semantic.py tests/learning/test_replay.py tests/performance -q --no-cov
.venv-qa/bin/ruff check src tests benchmarks/performance
.venv-qa/bin/mypy src/freshdata
.venv-qa/bin/mypy benchmarks/performance/analysis.py
.venv-qa/bin/mkdocs build --strict
git diff --check
.venv-qa/bin/python -m pytest
```

Final Task 9.5 gate results:

| Command | Exit | Result |
| :--- | ---: | :--- |
| Semantic/native/replay/performance tests | 0 | Completed at `[100%]`; no failures |
| Ruff (`src tests benchmarks/performance`) | 0 | `All checks passed!` |
| mypy `src/freshdata` | 0 | No issues in 187 source files |
| mypy `benchmarks/performance/analysis.py` | 0 | No issues in 1 source file |
| Strict MkDocs build | 0 | Documentation built successfully; only existing upstream warning and un-navigated-page INFO |
| `git diff --check` | 0 | No output |
| Full pytest suite | 0 | 3,238 passed, 7 skipped, 18 warnings in 223.55s; no failures |

The focused and full suites were run serially. The documentation build's
Material-for-MkDocs upstream notice and four existing `docs/superpowers/`
navigation INFO messages are informational and do not fail strict mode.

```bash
.venv-qa/bin/python -m pytest tests/performance -q --no-cov
.venv-qa/bin/mkdocs build --strict
git diff --check
git diff 6f6c2fe -- src/freshdata
```

Results from the earlier documentation verification run:

| Command | Exit | Result |
| :--- | ---: | :--- |
| Performance tests | 0 | All tests completed through `[100%]`; no failures |
| Strict MkDocs build | 0 | Documentation built successfully |
| Diff whitespace check | 0 | No output |
| Production-source identity diff | 0 | No output; `src/freshdata` remains byte-for-byte unchanged from `6f6c2fe` |

The documentation build also printed Material for MkDocs' upstream MkDocs 2.0
notice and INFO that two `docs/superpowers/` pages are not in `nav`; neither was
a strict-build error.

### Optimization-specific acceptance verification

Rejected by the Task 2 gate; no production optimization or before/after
equivalence comparison exists. The required final verification suite passed
serially (see the command table above), and the production-source identity
check remains clean.
