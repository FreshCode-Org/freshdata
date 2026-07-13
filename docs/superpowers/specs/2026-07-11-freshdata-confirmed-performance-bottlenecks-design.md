# FreshData Evidence-Gated Semantic Optimization Design

**Status:** Approved in conversation on 2026-07-11
**Scope:** Phase 2 design only; no production change is authorized by this document alone

This design follows the
[approved performance investigation design](./2026-07-11-freshdata-performance-scalability-design.md)
and the completed [performance investigation](../../performance-investigation.md).
It addresses the investigation's sole optimization candidate without treating
that candidate as a confirmed root cause or promising a performance gain.

## Decision

Measure exact repeated semantic predicate/parser arguments first. Only if that
measurement demonstrates a material cacheable opportunity may implementation
add exact, request-local result reuse inside semantic-context construction. The
change is accepted only if it preserves the complete behavioral contract and
passes the end-to-end performance and memory gates below. Otherwise it is
rejected or reverted; the scope does not expand to another optimization.

## Measured Evidence

The only Phase 2 candidate is baseline case `9ea9cb03dfc114c5`:

- 100,000 rows, medium width, mixed data, pandas input/output;
- `semantic_mode="assist"`, `verbose=False`, `return_report=True`, seed 42;
- one warm-up and five measured repetitions;
- profile total `22.015065380` seconds;
- semantic-stage fraction `0.13104697297973694`
  (`13.104697297973694%`);
- `run_semantic`: one call, `6.300034834000001` seconds cumulative;
- `build_semantic_context`: one call, approximately `6.184` seconds cumulative;
- `_build_info`: 32 calls, approximately `5.200` seconds cumulative;
- `_share`: 192 calls, approximately `4.232` seconds cumulative.

The capped self-time samples include `is_plain_number` at approximately
`0.718468278` seconds over 192,602 calls, `looks_like_date_value` at
approximately `0.387268678` seconds over 50,956 calls, and the boolean
predicate lambda in `semantic/context.py` at approximately `0.327692425`
seconds over 192,602 calls.

### Evidence limits

These measurements identify a candidate and hot functions, not a cause. The
profiler lists are capped and non-exhaustive. `_build_info` already applies
`pd.unique` within every bounded column sample, so reuse across columns or
context builds cannot be assumed. Exact repetition must be measured.

Every other investigated hypothesis was rejected or had insufficient evidence.
The required MissForest profile was interrupted after approximately two
CPU-bound hours and produced no JSON, so no MissForest conclusion is available.

## Compatibility Contract

The optimized and current paths must be indistinguishable across:

- public signatures, configuration fields, return types, and deterministic output;
- values, shape, row/column order, index values, names, and types;
- exact dtypes and input-mutation behavior;
- warning and exception type, message, and ordering;
- actions and action ordering, counts, rationales, risks, confidence,
  recommendations, and serialized reports;
- protected identifier, target, preserve, and PII behavior and privacy policy;
- semantic modes `off`, `assist`, `review`, and `auto`;
- memory/profile replay and native/fallback behavior.

Sampling, `pd.unique`, thresholds, classification, proposal resolution,
replacement order, and audit/report construction remain unchanged.

## Current Data Flow

```text
run_semantic
  -> build_semantic_context
       -> build_contexts
       -> for each column, _build_info
            -> bounded non-null sample
            -> pd.unique within that sample
            -> semantic predicates/parsers
            -> SemanticColumnInfo
       -> SemanticContext
  -> gather proposals
  -> calibrate proposals and decide
  -> build replacement map
  -> record every decision
  -> apply replacements
```

The only proposed boundary is the single `build_semantic_context` invocation.
The cache must not reach proposal gathering, replacement resolution, or report
construction.

## Proposed Design

### 1. RED discovery gate

Add development-only instrumentation around the named pure predicate/parser
calls made by `_build_info`. On the representative workload, record per
operation:

- total calls and the exact argument sequence;
- cache-eligible and bypassed calls;
- unique keys, repeated keys, theoretical hits, and hit rate;
- repeated-evaluation time sufficient to judge whether a 10% end-to-end win is
  feasible.

Reset the discovery counters and simulated cache at every
`build_semantic_context` entry. Compute unique keys and theoretical hits only
within that one build, then emit and aggregate the completed per-build metrics.
Aggregation may sum per-build counts but must not deduplicate across builds or
carry simulated entries forward. Repetition between warm-up, measured, memory,
or profile runs—or between any other context builds—never counts as a cacheable
hit.

The prospective key is `(predicate identity, exact input type, exact input
value)`. Measurement must use the same eligibility and key rules as the proposed
implementation. Call count alone is not sufficient: repeated eligible work must
be material enough to make the final acceptance threshold plausible. Record the
predeclared calculation and result. If it does not establish that opportunity,
stop with no production change.

After discovery passes, add a failing focused test proving that the current path
re-evaluates a repeated eligible key within one context build. The test must also
capture the uncached dataframe/context and complete report/audit fingerprint for
later differential comparison.

Instrumentation belongs in focused tests or benchmark tooling and adds no
default production overhead.

#### Candidate universe and exact input eligibility

Discovery instruments exactly these seven operations and input types:

- `is_plain_number`: exact built-in `str`, `int`, `float`, or `bool`;
- `parse_number_words`: exact built-in `str`;
- `parse_boolean`: the actual post-`str(v)` argument, exact built-in `str`;
- `parse_currency`: exact built-in `str`;
- `parse_unit`: exact built-in `str`;
- the email-shape predicate `bool(_EMAIL_VALUE.match(v.strip()))`: exact built-in
  `str`;
- `looks_like_date_value`: exact built-in `str`.

All non-allowlisted exact types bypass before any hashing or equality operation.
Subclasses, NumPy scalars, and user-defined values are never cached. Discovery
measures all seven operations; production may enable only the subset that
discovery proves material and that the implementation plan names explicitly. No
other operation is eligible.

### 2. Minimal GREEN change

Create one private cache when `build_semantic_context` starts and share it only
with that invocation's `_build_info` calls. Destroy it when the build returns or
raises. Do not add a public parameter or configuration switch. Do not add a size
cap: the one-build lifetime and the memory acceptance gate are the controlling
bounds, and the evidence does not justify another policy.

Evaluate only explicitly named, pure deterministic operations through the
request-local helper. Use the underlying parser/predicate identity, not an
ephemeral lambda identity. For transformations such as `parse_boolean(str(v))`,
perform the existing `str(v)` conversion in its existing order, then key the
parser by the exact string it receives.

Include both `type(value)` and `value` in every key so `1`, `True`, and
numeric-equivalent values of different Python types cannot collide. Apply the
exact operation/type allowlist above before hashing or equality; every other
value bypasses directly.

Call the original operation on a miss. Store a result only after successful
return and only when its result contract is immutable. The current candidate
results are `None`, booleans, integers, floats, and the immutable `(float, str)`
unit tuple. They remain internal to context scoring, so object identity is not
exposed. Mutable results are never cached.

Do not cache exceptions. Do not catch, translate, reorder, or suppress a parser
exception. A bypass calls the original operation at the same logical point. A
miss that raises propagates the same exception, and a second context build starts
with an empty cache.

### 3. Refactor and verification gate

Keep cache policy, lookup, and result eligibility in one small private unit;
keep `_build_info` responsible for unchanged sampling and scoring. Refactoring
must not generalize the cache beyond semantic-context construction.

Run differential tests against the captured uncached reference and the full
compatibility matrix. Re-run the representative benchmark and profiler only
after correctness passes.

## Test Design

The later implementation plan must use strict TDD:

1. RED: prove measured repetition, then fail on avoidable underlying calls while
   capturing the complete reference behavior.
2. GREEN: add only the request-local exact-result cache needed to pass.
3. REFACTOR: isolate the private helper without changing behavior or scope.
4. VERIFY: run differential, regression, performance, memory, and profile gates.

Focused cache tests cover a hit within one build, no reuse across builds,
predicate/type/value key separation, safe bypass, immutable result reuse, and
unchanged exception propagation. Broader differential coverage includes:

- normal and repeated inputs; mixed Python types and unhashable values;
- nullable and object dtypes and supported duplicate labels;
- protected columns, semantic hints, policies, and all semantic modes;
- report enabled/disabled, empty, single-row, and all-null inputs;
- native/pandas parity and native fallback;
- memory and profile replay.

Differential fingerprints compare the complete dataframe surface—values, shape,
ordering, index, and exact dtypes—and the complete report/audit surface,
including ordered actions, counts, rationales, risks, confidence,
recommendations, metadata, warnings, and serialization.

## Performance Acceptance Gate

Measure before and after in the same recorded environment using the primary
case: 100k rows, medium width, semantic assist, `verbose=False`, report enabled,
pandas input/output, seed 42, one warm-up, and five measured repetitions.

Acceptance requires all of the following:

- at least 10% median wall-time improvement;
- improvement greater than twice the larger before/after coefficient of
  variation, using the repository's comparison semantics;
- exact dataframe and report/audit fingerprint equality on every measured run;
- no material peak-RSS or Python-allocation-peak regression;
- no meaningful regression in the 100k/medium/report=true default and aggressive
  controls under the repository's precise comparison semantics;
- a new semantic profile showing time removed from the targeted functions rather
  than shifted to another stage.

The evidence record includes environment, commit, dirty state, exact commands,
individual repetitions, medians, coefficients of variation, peak RSS, Python
allocation peak, fingerprints, profile deltas, and accepted/rejected outcome.

## Failure, Rollout, and Rejection

There is no feature flag or staged public API. This is an internal optimization
that may land only after every compatibility and acceptance gate passes.

Reject or revert the production change if repetition is not material, any
behavior differs, the runtime threshold is missed, memory or controls regress,
or profiling shows displaced rather than removed work. Preserve the negative
measurement as evidence and do not broaden the implementation to recover a win.

## Rejected Alternatives

- Module, global, persistent `functools.lru_cache`, or cross-request caching:
  rejected for state leakage, unbounded retention, and semantic risk.
- Vectorizing or rewriting parsers/classifiers: rejected as excessive behavioral
  risk before exact reuse is measured.
- MissForest optimization: rejected because no completed profile exists.
- Backend rewrites and correlation, copy, dtype, reporting, uniqueness, or
  null-scan work: outside the sole measured Phase 2 candidate.

## Non-Goals

This phase does not change semantic decisions, sampling, configuration, public
interfaces, backend architecture, dependencies, reports, or documentation
claims. It does not create a general memoization framework or optimize any
unmeasured subsystem.
