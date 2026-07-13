# FreshData Semantic Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Either ship a verified request-local semantic-context reuse optimization, or publish a measured rejection without expanding scope.

**Architecture:** A benchmark-only probe measures only the seven approved context operations and resets for every context build. If within-build repetition is material, build_semantic_context owns one private memo shared only with its _build_info calls. Separately, analyzer field validators remove the six known mypy errors without changing the JSON contract.

**Tech Stack:** Python 3.9+, pandas 1.5–<3, pytest, unittest.mock, cProfile/tracemalloc harness, Ruff, mypy, MkDocs.

## Global Constraints

- Preserve Python >=3.9, pandas >=1.5,<3, NumPy >=1.21, public APIs/configuration, and add no dependency.
- Preserve dataframe values/order/index/dtypes, input mutation, warnings/errors, report/audit serialization and ordering, policies/protected columns, semantic modes, memory/profile replay, and native/fallback behavior.
- Scope is only case 9ea9cb03dfc114c5 and its semantic-context candidate. Do not optimize MissForest, correlation, backend conversion, copy, dtype, reporting, uniqueness, or null scanning.
- The operation universe is exactly is_plain_number, parse_number_words, post-str(v) parse_boolean, parse_currency, parse_unit, email-shape matching, and looks_like_date_value.
- Cache exact built-in str/int/float/bool only for is_plain_number and exact built-in str only for every other operation. Bypass every other value before hashing/equality.
- Cache lifetime is one build_semantic_context call. No global/lru cache, public switch, cross-request state, dependency, or speculative size cap.
- Cache only successful immutable results; never cache/suppress/reorder exceptions.
- Accept only a >=10% median primary-case win exceeding 2x the larger CV, exact output/report fingerprints, no material memory regression, no meaningful default/aggressive control regression, and a confirming profile.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| benchmarks/performance/semantic_probe.py | Development-only single-build instrumentation; never imported by production FreshData. |
| tests/performance/test_semantic_probe.py | Probe reset, keying, conversion, and bypass tests. |
| src/freshdata/semantic/context.py | Conditional private one-build memo and unchanged context scoring. |
| tests/test_semantic_cleaning.py | Memo lifetime, key separation, bypass, exception, and public behavior tests. |
| benchmarks/performance/analysis.py | Runtime-checked typed analyzer helpers. |
| tests/performance/test_analysis_render.py | Analyzer malformed-record regressions. |
| docs/performance-investigation.md | Actual Phase 2 evidence and accepted/rejected result. |

### Task 1: Add the development-only semantic repetition probe

**Files:**

- Create: benchmarks/performance/semantic_probe.py
- Create: tests/performance/test_semantic_probe.py

**Interfaces:**

- Consumes: freshdata.semantic.context.build_semantic_context, the seven context operation references, pandas.DataFrame, CleanConfig.
- Produces: SemanticProbeBuild containing per-operation total_calls, eligible_calls, bypassed_calls, unique_keys, theoretical_hits, hit_rate, and in-memory eligible value sequence; probe_context_build(df, config, *, stats=None).
- Does not modify production code, the benchmark schema, or the CLI.

- [ ] **Step 1: Write failing probe tests**

~~~python
def test_probe_counts_only_within_one_context_build() -> None:
    frame = pd.DataFrame({
        "left": ["yes", "no", "2024-01-01"],
        "right": ["yes", "no", "2024-01-01"],
    })
    config = CleanConfig(semantic_mode="assist", verbose=False)

    _context, first = probe_context_build(frame, config)
    _context, second = probe_context_build(frame, config)

    assert first.by_operation["parse_boolean"].theoretical_hits >= 1
    assert second.total_theoretical_hits == first.total_theoretical_hits


def test_probe_uses_exact_types_and_bypasses_unsafe_values() -> None:
    frame = pd.DataFrame({
        "ints": pd.Series([1], dtype=object),
        "bools": pd.Series([True], dtype=object),
        "text": pd.Series(["1"], dtype=object),
    })
    _context, result = probe_context_build(
        frame, CleanConfig(semantic_mode="assist", verbose=False)
    )

    numeric = result.by_operation["is_plain_number"]
    assert {type(value) for value in numeric.eligible_values} == {int, bool, str}
    assert numeric.unique_keys == 3
~~~

Also add direct-helper tests proving a list is bypassed without hashing and that parse_boolean records the post-str(v) string.

- [ ] **Step 2: Verify RED**

Run: .venv-qa/bin/python -m pytest tests/performance/test_semantic_probe.py -q --no-cov

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the minimal probe**

~~~python
@dataclass(frozen=True)
class OperationProbe:
    total_calls: int
    eligible_calls: int
    bypassed_calls: int
    unique_keys: int
    theoretical_hits: int
    eligible_values: tuple[object, ...]

    @property
    def hit_rate(self) -> float:
        return self.theoretical_hits / self.eligible_calls if self.eligible_calls else 0.0


def _eligible(operation: str, value: object) -> bool:
    return type(value) in _ALLOWED_TYPES[operation]


def probe_context_build(
    df: pd.DataFrame,
    config: CleanConfig,
    *,
    stats: dict[object, tuple[int, int, int | None]] | None = None,
) -> tuple[SemanticContext, SemanticProbeBuild]:
    with _patched_context_operations() as probe:
        context = semantic_context.build_semantic_context(df, config, stats=stats)
    return context, probe.finish_build()
~~~

Patch the six imported callable references in freshdata.semantic.context plus an _EMAIL_VALUE proxy whose match method records the already-stripped value then delegates. Eligible keys are (operation, type(value), value). A new probe is created for every function call, all raw keys remain in memory only, and bypassed values are never hashed.

- [ ] **Step 4: Verify GREEN**

Run: .venv-qa/bin/python -m pytest tests/performance/test_semantic_probe.py -q --no-cov && .venv-qa/bin/ruff check benchmarks/performance/semantic_probe.py tests/performance/test_semantic_probe.py

Expected: PASS.

- [ ] **Step 5: Commit**

~~~bash
git add benchmarks/performance/semantic_probe.py tests/performance/test_semantic_probe.py
git commit -m "test: add semantic context repetition probe"
~~~

### Task 2: Record the hard discovery decision

**Files:**

- Modify: docs/performance-investigation.md
- Create locally only: .superpowers/sdd/task-9-semantic-discovery.md

**Interfaces:**

- Consumes: Task 1 and make_mixed_frame(DatasetSpec(rows=100000, width="medium", seed=42, dataset_type="mixed")) with CleanConfig(semantic_mode="assist", verbose=False).
- Produces: a committed entry with the command, environment, per-operation metrics, and accepted_for_implementation or rejected_no_material_within_build_reuse.
- Decision boundary: only hits within one build count; warm-up, measurement, memory, profiling, and all cross-build repetitions never count.

- [ ] **Step 1: Add a focused discovery fixture test**

~~~python
def test_probe_reports_reuse_for_repeated_context_values() -> None:
    frame = pd.DataFrame({
        "one": ["yes", "no", "2024-01-01"],
        "two": ["yes", "no", "2024-01-01"],
    })
    _context, result = probe_context_build(
        frame, CleanConfig(semantic_mode="assist", verbose=False)
    )
    assert result.total_theoretical_hits > 0
    assert result.by_operation["looks_like_date_value"].theoretical_hits > 0
~~~

- [ ] **Step 2: Run the test and single-build primary measurement**

Run: .venv-qa/bin/python -m pytest tests/performance/test_semantic_probe.py::test_probe_reports_reuse_for_repeated_context_values -q --no-cov

Then, serially:

~~~bash
PYTHONPATH=src .venv-qa/bin/python -c 'from benchmarks.performance.datasets import DatasetSpec, make_mixed_frame; from benchmarks.performance.semantic_probe import probe_context_build; from freshdata.config import CleanConfig; frame = make_mixed_frame(DatasetSpec(rows=100000, width="medium", seed=42, dataset_type="mixed")); _, result = probe_context_build(frame, CleanConfig(semantic_mode="assist", verbose=False)); print(result.to_json())'
~~~

Expected: valid JSON for one context build.

- [ ] **Step 3: Make the explicit branch decision**

Advance only if repeated eligible calls occur in measured hot operations and their timed repeat work makes a 10% end-to-end win plausible. Otherwise skip Task 3, record rejection, do not change src/freshdata/semantic/context.py, and continue to Tasks 4 and 5.

- [ ] **Step 4: Document and commit actual evidence**

Add the exact command, commit/dirty state, each per-operation metric, decision, and reason to docs/performance-investigation.md. A rejected result must say no production optimization was implemented; an accepted result must say it only passed discovery.

Run: .venv-qa/bin/mkdocs build --strict && git diff --check

~~~bash
git add docs/performance-investigation.md
git commit -m "docs: record semantic discovery decision"
~~~

### Task 3: Conditionally add request-local reuse

**Files:**

- Modify: src/freshdata/semantic/context.py:43-253
- Modify: tests/test_semantic_cleaning.py:349-550

**Interfaces:**

- Consumes: Task 2's explicitly accepted operation subset.
- Produces: private _SemanticContextMemo.call(operation, value, fn), constructed once in build_semantic_context and passed to _build_info.
- Guarantees: operation/type/value key separation; pre-hash bypass; successful immutable results only; no cross-build state or cached exceptions.

- [ ] **Step 1: Write failing cache and equivalence tests**

~~~python
def test_context_memo_reuses_only_within_one_memo() -> None:
    memo = semantic_context._SemanticContextMemo()
    calls = 0

    def observed(value: object) -> bool:
        nonlocal calls
        calls += 1
        return semantic_context.is_plain_number(value)

    assert memo.call("is_plain_number", "12", observed) is True
    assert memo.call("is_plain_number", "12", observed) is True
    assert calls == 1
    assert semantic_context._SemanticContextMemo().call(
        "is_plain_number", "12", observed
    ) is True
    assert calls == 2


def test_context_memo_distinguishes_bool_int_and_str() -> None:
    memo = semantic_context._SemanticContextMemo()
    calls: list[object] = []

    def observed(value: object) -> bool:
        calls.append(value)
        return semantic_context.is_plain_number(value)

    assert memo.call("is_plain_number", 1, observed) is True
    assert memo.call("is_plain_number", True, observed) is False
    assert memo.call("is_plain_number", "1", observed) is True
    assert calls == [1, True, "1"]
~~~

Add failing tests for list bypass with two underlying calls, two identical raised
exceptions, and a repeated semantic fixture for the exact operation selected by
Task 2. That fixture must prove the selected call site invokes its underlying
operation once per repeated key within one build and again in a second build,
then assert the exact cleaned dataframe/report action fingerprint. Parameterize
public cleaning assertions over off, assist, review, and auto and retain
protected-column expectations.

- [ ] **Step 2: Verify RED**

Run: .venv-qa/bin/python -m pytest tests/test_semantic_cleaning.py -q --no-cov -k "context_memo or repeated_values or protected or semantic"

Expected: FAIL because _SemanticContextMemo does not exist.

- [ ] **Step 3: Implement the minimal private memo**

~~~python
_MISSING = object()

class _SemanticContextMemo:
    def __init__(self) -> None:
        self._values: dict[tuple[str, type[object], object], object] = {}

    def call(self, operation: str, value: object, fn: Callable[[Any], T]) -> T:
        if type(value) not in _ALLOWED_TYPES[operation]:
            return fn(value)
        key = (operation, type(value), value)
        cached = self._values.get(key, _MISSING)
        if cached is not _MISSING:
            return cast(T, cached)
        result = fn(value)
        self._values[key] = result
        return result
~~~

Introduce _email_shape(value: str) returning exactly bool(_EMAIL_VALUE.match(value.strip())). Preserve post-str(v) Boolean conversion by memo.call("parse_boolean", str(v), parse_boolean). Instantiate the memo only inside build_semantic_context, pass it to _build_info, and enable only Task 2's proven operation subset.

- [ ] **Step 4: Verify behavior**

Run: .venv-qa/bin/python -m pytest tests/test_semantic_cleaning.py tests/test_semantic_backends.py tests/test_execution/test_native_semantic.py tests/learning/test_replay.py -q --no-cov

Expected: PASS. Any changed report ordering, exception, policy result, or native parity is a stop-and-revert condition.

- [ ] **Step 5: Commit**

~~~bash
git add src/freshdata/semantic/context.py tests/test_semantic_cleaning.py
git commit -m "perf: reuse semantic context predicate results"
~~~

### Task 4: Fix the six inherited analyzer mypy errors

**Files:**

- Modify: benchmarks/performance/analysis.py:12-252
- Modify: tests/performance/test_analysis_render.py

**Interfaces:**

- Consumes: schema-validated dict[str, object] profiles.
- Produces: HypothesisRule and checked numeric/record helpers while retaining output ordering, thresholds, and JSON semantics.

- [ ] **Step 1: Write failing malformed-profile tests**

~~~python
def test_hypothesis_classifier_rejects_non_integer_profile_line() -> None:
    profile = _empty_profile()
    profile["functions"] = [{
        "file": "src/freshdata/engine/context.py",
        "line": "bad",
        "function": "build_context",
        "self_seconds": 0.0,
        "cumulative_seconds": 0.0,
        "calls": 1,
    }]
    with pytest.raises(TypeError, match="profile function line must be an integer"):
        classify_hypotheses(profile)
~~~

Add matching assertions for non-finite stage values and non-integer operation/call counts. Existing valid-profile classifier tests remain unchanged.

- [ ] **Step 2: Verify RED**

Run: .venv-qa/bin/python -m pytest tests/performance/test_analysis_render.py -q --no-cov

Expected: FAIL because malformed fields currently fall through to int/float conversion.

- [ ] **Step 3: Add checked types and helpers**

~~~python
class HypothesisRule(TypedDict):
    stages: tuple[str, ...]
    operations: tuple[str, ...]
    evidence: tuple[tuple[str, tuple[str, ...]], ...]


def _integer_field(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _finite_number_field(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"{label} must be a finite number")
    return float(value)
~~~

Annotate _HYPOTHESES as dict[str, HypothesisRule]. Check and narrow function/allocation arrays before using them. Use the helpers for lines, stage values, operations, calls, and allocation bytes; do not use a cast to evade validation.

- [ ] **Step 4: Verify GREEN**

Run: .venv-qa/bin/python -m pytest tests/performance/test_analysis_render.py -q --no-cov && .venv-qa/bin/mypy benchmarks/performance/analysis.py && .venv-qa/bin/ruff check benchmarks/performance/analysis.py tests/performance/test_analysis_render.py

Expected: PASS and mypy reports Success: no issues found.

- [ ] **Step 5: Commit**

~~~bash
git add benchmarks/performance/analysis.py tests/performance/test_analysis_render.py
git commit -m "fix: validate performance analysis record fields"
~~~

### Task 5: Run acceptance, publish the Phase 2 outcome, and verify completion

**Files:**

- Modify: docs/performance-investigation.md
- Modify only if regenerated: benchmarks/results/performance/baseline-summary.json
- Modify only if regenerated: benchmarks/results/performance/baseline-report.md

**Interfaces:**

- Consumes: Task 2 decision, Task 3 if accepted, primary/control benchmark JSON, and a confirming profile.
- Produces: evidence-backed accepted or rejected Phase 2 result, never an unsupported performance claim.

- [ ] **Step 1: Capture clean serial benchmark evidence**

Before Task 3, run:

~~~bash
.venv-qa/bin/python -m benchmarks.performance run --rows 100000 --widths medium --configs semantic --report-modes true --backends pandas --output-formats pandas --seed 42 --warmups 1 --repetitions 5 --output /private/tmp/freshdata-semantic-before
~~~

After Task 3, run serially:

~~~bash
.venv-qa/bin/python -m benchmarks.performance run --rows 100000 --widths medium --configs semantic,default,aggressive --report-modes true --backends pandas --output-formats pandas --seed 42 --warmups 1 --repetitions 5 --output /private/tmp/freshdata-semantic-after
~~~

If Task 2 rejects, skip before/after runs and retain the discovery evidence.

- [ ] **Step 2: Apply the acceptance gate**

For an accepted implementation, require equal output_fingerprint, report_fingerprint, and result_type; classify timing with classify_change(before_median, after_median, before_cv, after_cv); inspect raw samples, RSS, Python allocation peak, and both controls. Revert Task 3 if any required gate fails.

- [ ] **Step 3: Capture a confirming profile**

~~~bash
.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs semantic --report-modes true --backends pandas --output-formats pandas --seed 42 --warmups 1 --repetitions 5 --output /private/tmp/freshdata-semantic-profile-after
~~~

Confirm targeted context-operation work decreases without merely shifting to another stage.

- [ ] **Step 4: Publish actual, not projected, evidence**

Update docs/performance-investigation.md with command, environment, dirty state, metrics, fingerprints, profile delta, and accepted/rejected conclusion. If rejected, retain the no-production-optimization statement and explain why. Regenerate compact result/report artifacts only when their schema stays valid and no duplicate case rows are introduced.

- [ ] **Step 5: Final verification and commit**

Run serially:

~~~bash
.venv-qa/bin/python -m pytest tests/test_semantic_cleaning.py tests/test_semantic_backends.py tests/test_execution/test_native_semantic.py tests/learning/test_replay.py tests/performance -q --no-cov
.venv-qa/bin/ruff check src tests benchmarks/performance
.venv-qa/bin/mypy src/freshdata
.venv-qa/bin/mypy benchmarks/performance/analysis.py
.venv-qa/bin/mkdocs build --strict
git diff --check
.venv-qa/bin/python -m pytest
~~~

Commit intended documentation/compact artifacts only:

~~~bash
git add docs/performance-investigation.md benchmarks/results/performance/baseline-summary.json benchmarks/results/performance/baseline-report.md
git commit -m "docs: publish semantic optimization outcome"
~~~

Omit unchanged paths. Never commit raw /private/tmp results, .venv-qa, or .superpowers/sdd reports.

## Review Gates

After each task, create an exact base..HEAD review package, use a fresh read-only reviewer, fix every Critical or Important finding, then re-review the correction range. The final review checks the accepted/rejected evidence chain, production diff, static checks, links, and intended worktree.

## Plan Self-Review

- Tasks 1–2 implement the mandatory per-build discovery gate.
- Task 3 is the sole conditional production optimization.
- Task 4 resolves exactly the six inherited analysis.py mypy errors, not unrelated benchmark typing debt.
- Task 5 enforces equivalence, timing, memory, controls, profile, documentation, and complete verification.
- Every code task includes paths, interfaces, RED, GREEN, commands, and a commit.
