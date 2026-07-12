# Task 6 Report: Comparable Pandas Baselines, Analysis, and Markdown Rendering

## Status

Implemented Task 6 against base `7bcf97541a9b7f186e7cc68d11b00e1d11c8ce7d`.

The implementation adds only named pandas component baselines. It does not claim a
full pandas equivalent of FreshData's balanced decision/audit pipeline. A comparison
is produced only when a FreshData case explicitly declares
`options.comparable_baseline` with the exact component baseline name.

## RED

- `tests/performance/test_analysis_render.py` initially failed collection with
  `ModuleNotFoundError: No module named 'benchmarks.performance.analysis'`.
- The strict result-model tests initially failed because `BenchmarkResult.completed`
  rejected `baseline_name` and ordinary payloads omitted the required nullable field.
- Boundary coverage exposed floating-point behavior at an exact 10% change; the test
  expected `improved` while the initial implementation returned `noise`.
- Exact-evidence coverage demonstrated that partial names such as
  `correlation_label` were incorrectly accepted as correlation evidence.
- Semantic-comparison coverage demonstrated that explicitly matched component cases
  initially produced no comparison.

## GREEN

- Added required nullable `baseline_name` to the strict result model/schema; ordinary
  FreshData results serialize it as `null`, component results serialize the exact
  baseline key, and round trips preserve it.
- Added only the corrected component operations: `shallow_copy`,
  `numeric_median_fill`, `duplicates`, and `null_counts`.
- Added an isolated baseline worker path using one warm-up/five repetitions by default
  and a separate PeakRSS/tracemalloc measurement. The
  `pandas-component-baseline` sentinel is handled only by this path and is never sent
  to `fd.clean`.
- Added `classify_change` with the required absolute 10% threshold and twice the
  larger observed CV threshold.
- Added all eight hypothesis classifications. Candidate status requires non-zero
  relevant observed calls, exact function/path/line evidence, and either a 10% stage
  fraction or a 10% traced-peak allocation fraction.
- Added deterministic analysis ordering and Markdown rendering with exact commands,
  metric ranges/CV/throughput, memory ratios, scoped comparisons, stage percentages,
  exact function locations, candidate/rejected/evidence-limited hypotheses, durable
  failures, and limitations. Noise is labeled as noise and never rendered as an
  improvement.
- Added `baseline`, `analyze`, and `render` CLI commands.

## Verification

- `.venv-qa/bin/python -m pytest tests/performance --no-cov -W error -o addopts=''`
  - `144 passed in 15.97s`
- `.venv-qa/bin/ruff check benchmarks/performance tests/performance`
  - `All checks passed!`
- `.venv-qa/bin/ruff format --check benchmarks/performance tests/performance`
  - `21 files already formatted`
- `git diff --check`
  - clean

## Broader-suite observation

A diagnostic full-repository run with warnings promoted to errors reached one
unrelated existing failure in `tests/test_cleanbench_public_release.py`: the legacy
CleanBench pandas baseline emits a pandas datetime-inference `UserWarning`, which its
harness converts to `status="skipped"` under `-W error`. The Task 6 performance suite
is warning-free, and no production or legacy CleanBench code was changed.

## Commit

- `7ef5e79` — `bench: analyze and render scalability evidence`

## Concerns

- Component comparisons are intentionally opt-in and semantically scoped. Existing
  full FreshData cases do not declare component equivalence, so their comparison list
  remains empty rather than presenting a misleading balanced-pipeline claim.
- `.venv-qa/` remains untracked and untouched.

---

## Review Fix Wave

### RED

The focused review suite produced 25 expected failures before implementation:

- `load_results` accepted `NaN`, `Infinity`, and `-Infinity` until downstream
  validation, while the render CLI accepted all three outright.
- Hypothesis evidence accepted approved function names from `.bak` files and
  unrelated functions from approved files because matching used OR/substring logic.
- Profile hypotheses collided under an incomplete label when cases differed only by
  dataset type or seed.
- Component backend payloads accepted null, unknown, and forbidden `balanced`
  baseline names; non-component backends accepted non-null baseline names.
- `BenchmarkResult` could construct an inconsistent component-baseline identity.
- Profile headings omitted case identity and function rendering depended on payload
  order.

### GREEN

- Both JSON input paths now pass `parse_constant` callbacks to `json.loads` and reject
  all three non-standard constants before accepting the payload.
- Hypotheses are keyed by `BenchmarkCase.case_id`; each record retains a complete
  human label with case ID, rows, width, dataset type, config/options, report flag,
  backend/output, seed, warm-ups, and repetitions.
- The strict result schema conditionally requires one of exactly
  `shallow_copy`, `numeric_median_fill`, `duplicates`, or `null_counts` for the
  `pandas-component-baseline` backend and requires null for every other backend.
  `BenchmarkResult` enforces the same invariant at construction time.
- Every hypothesis uses explicit approved path-suffix/exact-function relationships.
  Matching requires both conditions, so `.bak` paths and unrelated context functions
  cannot qualify. Allocation significance is considered only alongside exact
  function evidence for the same hypothesis family.
- Profile headings render the complete case identity. Top functions are sorted by a
  canonical metric/location key before the top ten are selected, making equivalent
  payloads order-independent.

### Review Verification

- `.venv-qa/bin/python -m pytest tests/performance --no-cov -W error -o addopts=''`
  - `172 passed in 23.12s`
- `.venv-qa/bin/ruff check benchmarks/performance tests/performance`
  - `All checks passed!`
- `.venv-qa/bin/ruff format --check benchmarks/performance tests/performance`
  - `21 files already formatted`
- Python 3.9 grammar check using `ast.parse(..., feature_version=(3, 9))`
  - `Python 3.9 grammar: 21 files parsed`

### Review Commit

- `917bc29` — `bench: harden performance evidence identity`

---

## Final Review Fix Wave

### RED

A focused regression run produced eight expected failures before implementation:

- allocation bytes from `context.py:44` incorrectly combined with exact
  `build_context` function evidence from `context.py:231`, creating a candidate;
- `classify_change` accepted direct `NaN`, positive infinity, and negative infinity;
- the render CLI accepted JSON `1e999`, which Python parses as positive infinity;
- `render_report` accepted nested direct `NaN`, positive infinity, and negative
  infinity values.

The matching `load_results` overflow case and direct `analyze_results` cases were
also added; their existing strict result validation already rejected the values.

### GREEN

- Added recursive finite-float validation and invoked it after JSON parsing, during
  strict result validation, and at the public analysis, hypothesis, comparison, and
  render entry points. Boolean handling remains delegated to the existing schema and
  model semantics.
- Kept strict JSON serialization with `allow_nan=False` on generated JSON output.
- Allocation significance now counts only records whose normalized file and exact
  line equal a location from already matched exact function evidence. Same-file bytes
  from another line no longer satisfy the 10% threshold.
- Added positive and negative allocation-location coverage and JSON-overflow/direct
  programmatic non-finite coverage for load, analysis, render CLI, and renderer paths.

### Final Review Verification

- `.venv-qa/bin/python -m pytest tests/performance/test_analysis_render.py tests/performance/test_models_schema.py tests/performance/test_instrumentation.py -q --no-cov -W error -o addopts=''`
  - `148 passed in 6.54s`
- `.venv-qa/bin/python -m pytest tests/performance --no-cov -W error -o addopts=''`
  - `181 passed in 16.52s`
- `.venv-qa/bin/ruff check benchmarks/performance tests/performance`
  - `All checks passed!`
- `.venv-qa/bin/ruff format --check benchmarks/performance tests/performance`
  - `21 files already formatted`
- Python 3.9 grammar check using `ast.parse(..., feature_version=(3, 9))`
  - `Python 3.9 grammar: 21 files parsed`
