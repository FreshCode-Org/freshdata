# FreshData Performance and Scalability Investigation Design

**Status:** Approved in conversation on 2026-07-11

**Baseline:** Git commit `6f6c2fe` on branch
`fix/public-exports-and-inf-outliers-jwd`

## Objective

Measure, explain, and reduce FreshData's verified runtime and memory bottlenecks
without changing its public API, cleaning decisions, safety guarantees, audit
trail, context policies, validation behavior, or supported backends. The work
must independently reproduce or disprove each performance hypothesis, retain
only evidence-backed optimizations, and publish the commands and results needed
to reproduce every claim.

An optimization is a verified win only when it improves the relevant median
runtime or peak-memory measurement by at least 10%, the improvement is greater
than twice the observed run-to-run variability, and it introduces no meaningful
regression in another primary workload.

## Compatibility Constraints

- Python support remains `>=3.9`, including the project's tested Python
  3.9-3.13 matrix.
- pandas support remains `>=1.5,<3`; NumPy remains `>=1.21`.
- The default cleaning strategy remains `balanced`, with the current
  representation-repair defaults unchanged.
- Public function and class signatures, configuration fields, return types,
  warnings, exceptions, index semantics, and audit/report contracts remain
  compatible.
- No unrelated runtime dependency is introduced. Profiling additions use the
  standard library and already-declared benchmark/development dependencies.

## Current Architecture and Execution Flow

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
reference. `src/freshdata/engine/context.py` profiles roles and statistics,
`src/freshdata/engine/cache.py` shares artifacts between automatic missing and
outlier processing, and `src/freshdata/steps/` implements representation-level
operations. Native engines in `src/freshdata/execution/backends/` reproduce a
documented subset and fall back to pandas when required. Every path must retain
the existing `CleanReport` contract.

`return_report=False` does not mean that report construction can be removed:
the pandas result remains a `CleanResult` with an embedded report and
`Cleaner.report_` is always populated. Optimization may defer or avoid only
internal work proven unnecessary while preserving those behaviors exactly.

## Chosen Approach

Use an evidence-gated sequence:

1. Establish an immutable baseline and a reproducible benchmark/profiling
   harness.
2. Profile the complete pandas pipeline and separate core cleaning, reporting,
   optional ML/semantic work, conversion, and backend costs.
3. Implement one measured optimization at a time behind equivalence tests.
4. Re-run the affected benchmark slice after each change and discard changes
   that do not exceed the noise threshold.
5. Expand CI-safe regression checks and scheduled/manual large-data coverage.
6. Evaluate native and out-of-core options from measurements rather than begin
   with an architectural rewrite.
7. Correct documentation and assemble the final evidence package.

This approach was selected over a broad refactor, which would obscure causal
attribution, and a backend-first rewrite, which would add compatibility risk
before the reference pandas bottlenecks are known.

## Workstream Boundaries

### 1. Baseline and Profiling Infrastructure

Create a focused `benchmarks/performance/` package with independent modules for
dataset generation, run configuration, subprocess measurement, profiling,
result validation, comparison, and Markdown rendering. Keep it separate from
production imports. Reuse existing benchmark utilities only where their result
schema and measurement semantics match this investigation.

The authoritative output is versioned JSON. Generated Markdown is a view of
that JSON. Each result records:

- Git commit and dirty-state indicator.
- FreshData, Python, pandas, NumPy, and optional-backend versions.
- Operating system, CPU, logical/physical core count, and available RAM.
- Dataset seed, shape, width profile, and column-family counts.
- Full cleaning configuration, backend, output format, and report flag.
- Warm-up count, measured repetitions, and exact command.
- Per-run wall time, median, minimum, maximum, standard deviation, coefficient
  of variation, and rows per second.
- Peak RSS increase, Python allocation peak, input bytes, and input-to-peak
  ratio.
- Equivalent pandas baseline time, FreshData slowdown ratio, and before/after
  FreshData improvement ratio where the comparison is semantically valid.
- Completion, skip, failure, timeout, or memory-exhaustion status.

The command-line interface accepts configurable row count, column count or
width profile, dataset type, cleaning configuration, repetition count, backend,
report generation, output format, seed, timeout, and JSON/Markdown destination.

### 2. Pandas Pipeline Optimization

Treat the following as hypotheses until profiling confirms them:

- Repeated null, non-null, uniqueness, mode, dtype, role, and shape scans can
  be reused through the engine cache.
- The full numeric correlation matrix is built when no consumer can use it.
- Missing and outlier dispatch rescans columns despite exact context data.
- String, sentinel, dtype, or numeric operations perform avoidable copies or
  conversions.
- Report finalization repeats full-frame work that can be safely shared.
- A precomputed execution plan can reduce dispatch overhead for repeated
  compatible transformations.

Only exact reuse is allowed. If a cached value could be stale after an earlier
stage changes a column, the current computation remains authoritative.
Correlation work may be skipped only when it is provably unused, including
cases such as balanced mode, explicit imputation, frames above the existing
10,000-row KNN limit, or the absence of an eligible medium-missing numeric
target. When correlations are needed, a targeted calculation is acceptable
only if it produces pandas-equivalent values.

Batching is permitted only for compatible columns and must retain column order,
dtypes, action ordering, counts, rationales, risks, confidence values, warnings,
and exceptions. Copy reductions must preserve `preserve_original`, pandas
copy-on-write compatibility, protected-column snapshots, and documented input
mutation behavior.

### 3. Correctness and Regression Coverage

Add differential tests that compare the optimized implementation with baseline
behavior for values, shape, column order, index values/names, exact dtypes,
input mutation, actions and action order, counts, rationale, risk, confidence,
warnings, recommendations, fallback events, and serialized reports.

Coverage includes:

- Normal mixed frames, empty frames, and single-row frames.
- All-null and constant columns.
- Boolean and nullable-boolean columns.
- Nullable integer columns and ordinary integer columns.
- Float32, float64, finite values, NaN, and positive/negative infinity.
- Categorical columns, including missing categories.
- Datetime and timezone-aware columns.
- Mixed object payloads and unhashable values.
- Identifiers, targets, context-protected columns, and PII-sensitive fields.
- Duplicate rows, supported duplicate labels, and duplicate-resolution modes.
- Standard indexes, named indexes, DatetimeIndex, and MultiIndex where
  supported.
- Narrow, medium, wide, and large frames.
- Report-enabled and report-disabled calls.
- Conservative, balanced, aggressive, explicit imputation/outlier, semantic,
  context-policy, memory/profile replay, and native-fallback paths.
- Existing exception types, validation order, and warning text/categories.

CI performance tests assert stable structural properties rather than tiny wall
time differences. Examples include proving that correlations are not computed
when unused, exact cached counts are reused, and disabled features do not invoke
their stages. Runtime and peak-memory thresholds belong in scheduled or manual
performance jobs.

### 4. Backend and Out-of-Core Evaluation

Measure pandas, Polars, DuckDB, Spark, and FreshCore where their optional
dependencies and runtime requirements are available. For every relevant step,
record whether execution is native, materialized, converted, or delegated to
pandas, including conversion time, peak memory, fallback events, and audit
differences.

Produce evidence-based recommendations for:

- A native Polars lazy execution engine.
- DuckDB larger-than-memory processing.
- Chunked pandas execution.
- Rust/FreshCore acceleration of verified hot loops.

Do not implement a large backend rewrite unless optimized pandas measurements
still miss the scalability targets and the proposed native path can preserve
the public behavior and audit contract. Otherwise provide an implementation
proposal with boundaries, milestones, compatibility requirements, and
benchmark targets.

### 5. Documentation and Verification

Publish `docs/performance-investigation.md` as the evidence-backed report and
update `README.md`, benchmark documentation, backend documentation,
limitations, and production-readiness claims where measurements contradict or
qualify phrases such as "fast", "vectorized", "scalable", or "memory
efficient".

The documentation states measured performance ranges, known scalability and
correlation-cost limits, memory considerations, report/non-report behavior,
recommended large-data configurations, and the status of optional fast or
backend-native paths.

No benchmark result may be presented without its environment and command. Poor
results and unresolved limitations remain visible.

## Benchmark Matrix

The deterministic mixed-schema generator covers numeric, categorical, string,
nullable, datetime, timezone-aware, identifier, target, duplicate, missing,
outlier, and high-cardinality data.

| Dimension | Values |
|---|---|
| Rows | 10,000; 100,000; 500,000; 1,000,000 |
| Width | narrow: 8; medium: 32; wide: 128 columns |
| Cleaning | default balanced; conservative; representation features disabled; statistical features disabled; explicit imputation/outliers; optional ML/semantic |
| Report flag | `return_report=True`; `return_report=False` |
| Backend | pandas; installed Polars, DuckDB, Spark, FreshCore paths |
| Output | pandas and supported eager/lazy native output formats |
| Repetitions | one warm-up, then five measured runs |

Equivalent pandas baselines are used only for comparable component workloads;
they must implement the same selected transformation and may not be described
as equivalent to FreshData's full decision and audit pipeline.

## Profiling Design

Profiling is development-only and adds no default production overhead:

- `cProfile` attributes function-level runtime.
- `tracemalloc` snapshots identify allocation-heavy files and lines.
- Stage timers isolate context construction, engine-cache construction,
  correlation, missing processing, outlier processing, role inference, dtype
  repair, duplicate handling, audit generation, final report generation,
  optional ML/semantic work, backend conversion, and materialization.
- Controlled instrumentation counts observed calls to `DataFrame.copy`,
  `Series.copy`, and selected conversion methods.
- Scan counters observe null counts, uniqueness calculations, correlations,
  dtype conversions, and role inference.

Copy-call counts are explicitly labelled observations and are not treated as a
complete measure of physical pandas buffer copies. Peak RSS and allocation
measurements run in isolated subprocesses so previous runs and allocator residue
cannot contaminate results.

## Error Handling

Benchmark failures are data, not successes. The harness writes a failure record
with the command, environment, exception type/message, and partial measurements.
Unavailable optional backends are marked skipped with the exact missing
requirement. Timeouts and memory exhaustion remain visible.

Production behavior is stricter: optimizations preserve existing exception
types, messages where asserted, validation order, warnings, and fallback
events. Cached or optimized logic falls back to the current computation when
the required invariant cannot be proven. A change that cannot demonstrate
equivalence is rejected or documented as a future proposal.

## Change Gate

Every production optimization follows this sequence:

```text
write a failing equivalence or structural-performance test
  -> verify the expected failure
  -> implement the smallest internal change
  -> pass focused correctness tests
  -> compare against baseline behavior
  -> run the relevant benchmark slice
  -> retain only if the gain exceeds the noise threshold
```

A local gain that causes a regression elsewhere must be narrowed behind a
proven eligibility predicate or removed.

## CI and Scheduled Workflows

Pull-request CI remains bounded and deterministic. It validates the benchmark
schema, generator determinism, report rendering, structural performance
properties, and representative small/medium equivalence cases.

A scheduled and manually dispatchable workflow runs the 100,000 through
1,000,000-row profiles, publishes JSON/Markdown artifacts, and includes enough
metadata to compare runs only on compatible environments. It supplements rather
than replaces the existing CleanBench and performance-regression workflows.

## Required Deliverables

`docs/performance-investigation.md` contains these explicit sections:

1. Executive summary.
2. Reproduction commands.
3. Baseline benchmark table.
4. Profiling findings with functions, files, and lines.
5. Confirmed root causes.
6. Rejected hypotheses.
7. Files and functions changed.
8. Explanation of every optimization.
9. Behavioral compatibility analysis.
10. Tests added or changed.
11. Before-and-after benchmark table.
12. Peak-memory comparison.
13. Remaining bottlenecks.
14. Backend and out-of-core recommendations.
15. Documentation corrections.
16. Risks and trade-offs.
17. Exact verification commands and results.

Each important change separately states its problem, evidence, implementation,
correctness protection, measured performance impact, and remaining risk.

## Final Verification

Run and record all repository-supported gates:

- Complete pytest suite, including online/large tests where their fixtures are
  available.
- Ruff linting and formatting checks.
- mypy type checking.
- Repository-supported security checks.
- Strict MkDocs build.
- Source and wheel build plus `twine check`.
- Existing benchmark and CleanBench suites.
- New complete benchmark matrix and profiler.
- Available Polars, DuckDB, Spark, and FreshCore checks.

The final completion audit maps every objective requirement to authoritative
evidence. A missing optional service, dependency, fixture, or runtime is
reported explicitly and prevents an unsupported claim; it is never silently
omitted.

## Non-Goals

- Changing public signatures or defaults.
- Weakening validation, auditability, safety, context policy, PII, identifier,
  or target protections.
- Hiding report work by removing the embedded report contract.
- Adding unrelated runtime dependencies.
- Replacing pandas with a native backend without measured justification.
- Publishing marketing claims that are not supported by recorded results.
