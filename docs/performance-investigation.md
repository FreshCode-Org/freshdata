---
title: Performance investigation
description: >-
  Reproducible FreshData performance measurement, profiling, analysis, and
  evidence-reporting methodology.
keywords: freshdata performance, scalability, profiling, benchmark methodology
---

# Performance investigation

## Executive summary

Baseline measurement is in progress. No slowdown or improvement is claimed yet.
This page defines the architecture, commands, and evidence rules used to produce
later findings without treating an unmeasured hypothesis as a result.

## Architecture and execution flow

The public cleaning path is:

```text
fd.clean
  -> normalize API options and validate contracts/context/memory/profile inputs
  -> select the pandas or native execution path
  -> initialize CleanReport and preserve-original handling
  -> representation repair
       column names
       strings and sentinels
       empty rows and columns
       dtype repair
       optional constant-column removal
       duplicate handling
       optional semantic repair
  -> decision engine
       build per-column contexts
       build reusable engine statistics/correlations
       automatic missing-value decisions
       automatic outlier decisions
  -> explicit imputation and outlier overrides
  -> optional memory optimization
  -> protected-column verification and index handling
  -> finalize report
  -> wrap or convert the result
```

The pandas implementation in `src/freshdata/cleaner.py` is the behavioral
reference. Context profiling lives in `src/freshdata/engine/context.py`, shared
engine artifacts live in `src/freshdata/engine/cache.py`, and representation
operations live in `src/freshdata/steps/`. Native backends reproduce a documented
subset and fall back to pandas when required. Every path must preserve the
`CleanReport` contract.

`return_report=False` does not remove report construction: the pandas result
remains a `CleanResult` with an embedded report, and `Cleaner.report_` is still
populated. Any future optimization must preserve those behaviors.

The development-only harness in `benchmarks/performance/` generates deterministic
mixed-schema frames, runs each benchmark case in an isolated subprocess, validates
each result against the versioned JSON contract, analyzes measurements, and renders
Markdown from that authoritative JSON. It does not add production runtime work.

## Supported environment and defaults

The investigation preserves the supported Python 3.9–3.13 matrix, pandas
`>=1.5,<3`, and NumPy `>=1.21`. The default cleaning strategy remains `balanced`,
including the existing representation-repair defaults. Profiling uses the standard
library and dependencies already declared by the development and benchmark extras.

## Reproduction commands

Install the development, benchmark, and optional ML dependencies before running
the full matrix:

```bash
pip install -e ".[dev,bench,ml]"
```

The Make targets use the active `PY` interpreter; override it when needed, for
example with `make performance-ci PY=.venv/bin/python`.

```bash
make performance-ci
make performance-baseline
make performance-profile
make performance-report
```

`performance-ci` runs the small deterministic contract suite.
`performance-baseline` writes raw case JSON under
`benchmarks/results/performance/baseline/`. `performance-profile` records one
100,000-row medium-width default case with report behavior enabled.
`performance-report` analyzes the raw directory into
`baseline-summary.json` and renders `baseline-report.md` from that summary.

## Measurement methodology

Reportable cases use one warm-up followed by five measured repetitions. Each case
records its exact command and environment, dataset seed and shape, configuration,
individual timing samples and summary statistics, throughput, peak RSS, Python
allocation peak, input size, status, and any failure, timeout, or memory-exhaustion
details. Timing and memory measurements run separately so memory sampling does not
silently redefine the timing result.

Equivalent pandas baselines are limited to named component operations that perform
the same selected transformation. They are not described as equivalent to
FreshData's complete decision and audit pipeline.

An optimization is a verified win only if the relevant median runtime or peak-memory
measurement improves by at least 10%, the improvement exceeds twice the observed
run-to-run variability, and no other primary workload has a meaningful regression.
CI contracts assert stable behavior and structure rather than fragile wall-clock
thresholds; large runtime and memory measurements run manually and weekly.

## Evidence lifecycle

Authoritative, schema-validated JSON is the source for the generated baseline,
profile, root-cause, rejected-hypothesis, before/after, memory, backend,
documentation, risk, and verification sections as their investigation phases
complete. Generated Markdown is a view of that JSON. Raw case files remain local or
in workflow artifacts; only compact `*-summary.json` and `*-report.md` evidence is
eligible to be committed.

Failures, timeouts, memory exhaustion, poor results, and unresolved limitations
remain visible in the evidence. A result is not published without its environment
and reproduction command.
