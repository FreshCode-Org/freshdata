# FreshData TruthBench Design

**Date:** 2026-07-14  
**Target:** `origin/main` at `a1da862`  
**Implementation branch:** `feature/truthbench-jwd`  
**Status:** Approved for implementation planning

## Purpose

FreshData TruthBench is a deterministic semantic red-team and regression system.
It continuously searches for cases where a public FreshData data-quality surface
makes an unsafe, incorrect, inconsistent, or unexplained decision, then converts
each reproduced failure into a minimized case and a permanent regression test.

TruthBench is test infrastructure, not a cleaning engine. No LLM, teacher model,
provider hook, or generative runtime may participate in FreshData's default
cleaning path or in a TruthBench release run. An LLM may only help humans author
adversarial seed cases, criticize results, and analyze reproduced failures outside
the executable benchmark.

## Repository Context

TruthBench is additive to the existing quality layers:

- CleanBench measures frame-level repair fidelity, calibration, preservation,
  profile replay, privacy, and performance.
- The Validation Gauntlet measures labelled dispositions across five fixtures,
  but uses partial cell labels and a deliberately loose value comparator.
- The enterprise fixture benchmark measures repair, preservation, trust
  monotonicity, reporting completeness, and scale.
- Golden, real-world, online, domain, integration, backend, Copilot, privacy, and
  generated-code pytest suites protect individual public contracts.
- The performance investigation provides subprocess-isolated scalability
  evidence and is not a semantic correctness oracle.

TruthBench does not replace, rename, weaken, or rebaseline these systems. It adds
the strict cross-surface cell-level contract they do not currently provide.

The isolated baseline run on the target commit produced 3,692 passes, 11 skips,
12 deselections, and 93.66% coverage. Two existing throughput assertions failed
during the seven-minute full run:
`balanced-abalone` missed its allowed floor by approximately 0.004%, and
`aggressive-gapminder` was slower than its allowed floor. Both exact tests passed
immediately when rerun with coverage enabled. These are recorded as pre-existing,
non-reproducible environment timing flakes. TruthBench must not relax, skip, or
change either expectation.

## Chosen Approach

TruthBench will live in a dedicated `benchmarks/truthbench/` package. It may reuse
stable public APIs and small general-purpose helpers from existing benchmarks,
but its oracle, exact comparator, result schema, gates, surface adapters, failure
reducer, and eight-domain corpus are independent.

This is preferred over extending the Validation Gauntlet because the Gauntlet's
loose canonical equality and partial labels are part of its historical contract.
It is preferred over a CleanBench T6 track because CleanBench is organized around
aggregate frame metrics rather than one record per cell and surface.

## Scope

### In scope

TruthBench will:

1. Generate deterministic gold-labelled fixtures for finance, healthcare,
   retail, CRM, logistics, government, education, and insurance.
2. Assign exactly one disposition to every test cell:
   `preserve`, `repair`, `flag`, or `review`.
3. Exercise every public FreshData surface that makes, applies, transports,
   renders, serializes, or gates a data-quality decision.
4. Normalize heterogeneous surface outputs into a common decision record.
5. Compare actual decision, expected decision, output value, confidence,
   rationale, audit completeness, trust-score change, backend consistency, and
   determinism.
6. Detect raw PII leakage into reports, logs, model context, generated code,
   rendered output, CLI output, and committed benchmark artifacts.
7. Minimize every reproduced failure to a stable, privacy-safe case.
8. Enforce absolute PR and release gates independent of baseline-relative trends.
9. Commit versioned machine-readable results and a human-readable report.

### Out of scope

TruthBench will not:

- add an LLM or network call to `fd.clean`, semantic routing, domain repair,
  privacy processing, trust scoring, or CI;
- automatically approve or apply an ambiguous repair;
- silently accept documented backend divergence;
- use timing, timestamps, process memory, random salts, or generated identifiers
  as decision-determinism inputs;
- replace performance benchmarks with correctness measurements;
- treat an unavailable required backend as a skip;
- overwrite benchmark gold labels to make an implementation pass.

## Disposition Contract

Each physical test cell has a stable identifier:
`<fixture-version>:<domain>:<row-id>:<column>`.

The four dispositions mean:

| Disposition | Required default behavior |
|---|---|
| `preserve` | The value is valid, possibly unusual. Mutating its value or semantic meaning is corruption. An error-severity false alarm is also a failure. |
| `repair` | A safe deterministic repair has one gold output. A mutating surface must produce that exact value and dtype; a read-only surface must identify the problem without inventing another value. |
| `flag` | The value must be surfaced with evidence but remain unchanged by default. |
| `review` | The value is ambiguous or policy-conflicted and must enter a human-review, quarantine, reject, or explicit suggestion path without an automatic guess. |

All pristine background cells default to `preserve`. Adversarial injections replace
that label explicitly. Fixture construction fails if any cell has no label, more
than one label, a missing stable ID, or an invalid expected output.

Row-level expectations such as exact duplicate removal and schema-level
expectations such as added, missing, renamed, or type-drifted columns are stored as
separate case records. They never substitute for cell labels.

## Gold Fixture Design

Each domain fixture contains a pristine frame, an adversarial frame, a complete
cell-label matrix, explicit schemas, context policies, protected columns, PII
canaries, and row/schema expectations. Generation is pure Python with fixed seeds,
fixed reference dates, fixed timezone, fixed locale assumptions, and stable row
identifiers.

The corpus includes:

- valid extremes, rare categories, leading-zero identifiers, Unicode names,
  uncommon but valid codes, zero-value transactions, and empty free text;
- ambiguous decimal separators, numeric dates, percentages, abbreviated units,
  policy aliases, category variants, and case-sensitive identifiers;
- mixed kilograms/pounds, Celsius/Fahrenheit, percentages/fractions, local time
  zones, currencies, currency symbols, and Indian/international number grouping;
- English and non-English text, mixed scripts, emoji, combining characters,
  mojibake, zero-width characters, HTML entities, and alternate encodings;
- ISO, US, European, textual, partial, timezone-aware, and impossible dates;
- hidden email, phone, SSN/national-ID, card, address, medical, and FERPA-style
  identifiers in free text, numeric columns, category values, and late rows;
- added, removed, reordered, renamed, duplicated, and type-drifted columns;
- contradictory protection/repair, currency/unit, range, locale, and mapping
  policies;
- semantic traps where `apple`, `Apple`, and `AAPL` have different correct
  meanings based on column, schema, and domain.

Domain emphasis:

- **Finance:** instruments, company names, tickers, prices, currencies, ledger
  signs, percentages, settlement dates, and transaction memos.
- **Healthcare:** patient identifiers, clinical codes, lab values and units,
  dates of birth, vital signs, free-text notes, and protected health information.
- **Retail:** SKUs, quantities, prices, promotions, returns, product names,
  currencies, reviews, and zero-price edge cases.
- **CRM:** customer IDs, Unicode names, email, phone, country codes, lifecycle
  states, signup dates, and free-text contact notes.
- **Logistics:** shipment IDs, UN/LOCODE-like locations, weights, dimensions,
  temperatures, time zones, tracking states, delivery windows, and addresses.
- **Government:** case IDs, agency and district codes, postal identifiers,
  multilingual values, fiscal amounts, public dates, and restricted identifiers.
- **Education:** student IDs, grades, assessment scales, school years, programs,
  enrollment dates, guardian details, and FERPA-protected notes.
- **Insurance:** claim and policy IDs, coverage codes, premiums, reserves,
  incident dates, status transitions, medical/loss descriptions, and claimant PII.

Fixture files may contain synthetic PII canaries because the detector needs raw
inputs. No result, report, failure artifact, or committed baseline may contain
those literals.

## Public Surface Inventory

TruthBench maintains a versioned manifest of decision-bearing and decision-sink
surfaces. A contract test compares the manifest with the public exports and known
CLI commands so a new surface cannot be added without an explicit coverage
classification.

The release profile covers:

- `fd.clean`, `Cleaner`, `clean_csv`, `CleanResult`, and `CleanReport`;
- fluent pipelines, plan/suggest/apply workflows, and decision hashes;
- field validation and remediation policy;
- validation suites, enterprise contracts, and context policies;
- bundled domain validators and domain repair integration;
- deterministic semantic default, review, and auto behavior;
- safe text cleaning and encoding linting;
- PII detection, anonymization, privacy policies, and privacy reports;
- trust scoring, quality reports, and trust gates;
- streaming with fixed batch partitions;
- pandas, Polars, and DuckDB required backend paths;
- AI Copilot with `provider=None` only;
- generated Copilot code parsing, compilation, and controlled execution;
- report exporters, JSON, Markdown, HTML/Peel rendering, CLI stdout/stderr, and
  persisted artifacts as privacy and audit sinks.

Explicit low-level utilities that perform a caller-requested transform but make no
semantic decision are classified as `explicit-transform`. They receive input/output
and safety contract tests, but they do not pretend to infer a disposition.

## Surface Adapter Contract

Every adapter declares:

- stable name and version;
- whether it mutates, validates, suggests, serializes, or renders;
- supported fixture features and required dependencies;
- the surface-specific mapping from the global gold disposition to the expected
  surface decision;
- output extraction, decision extraction, and audit extraction rules;
- deterministic fields and explicitly excluded telemetry fields;
- whether exact backend parity is required.

For example, a mutating cleaner is expected to repair `repair`, preserve
`preserve`, and leave `flag`/`review` unchanged while surfacing them. A read-only
validator is expected to preserve values and emit `flag` or `review` evidence for
the corresponding defects. A PII detector is expected to flag labelled PII and
remain silent on non-PII preservation traps.

Adapters must not catch broad exceptions. An unexpected exception is a benchmark
failure with its type, safe message, fixture, surface, and reproduction ID.

## Normalized Decision Record

TruthBench emits one record for every `(surface, backend, repeat, cell)` tuple:

- record schema version;
- run, fixture, case, and cell IDs;
- domain, row ID, and column;
- expected and actual disposition;
- input type and privacy-safe digest;
- expected and actual output type;
- output value for non-sensitive cells or a redacted value plus digest for
  sensitive cells;
- confidence, risk, status, model/rule ID, and rationale;
- evidence kinds without raw sensitive samples;
- mutation, detection, quarantine, and human-review booleans;
- audit-required and audit-complete booleans plus matching audit IDs;
- trust before, trust after, and delta at the applicable frame/surface level;
- requested backend, actual backend, fallback events, and backend differences;
- normalized decision hash and repeat-consistency status.

Exact output equality includes dtype and semantic representation. The comparator
does not equate `"402.10"` with `402.1`, strip whitespace before preservation
checks, or treat case-folded identifiers as identical. Explicit fixture metadata
may authorize a representation equivalence only when that equivalence is itself
the gold repair.

## Backend Consistency

The release environment must install and successfully execute pandas, Polars, and
DuckDB. Missing required dependencies, silent materialization, unexpected fallback,
or a mismatched requested/actual backend fails the run.

Parity uses the common deterministic native subset with
`fallback_policy="error"`. It compares normalized decisions, exact values where
the documented contract promises equality, row identity/order, action status,
confidence, rationale class, and audit coverage. A documented dtype or aggregation
difference is only accepted when represented by an explicit, tested equivalence
rule and disclosed in the report.

Spark and FreshCore require JVM/native build infrastructure. They belong to the
extended scheduled profile, where absence is a profile failure rather than a skip.
Their adapter contracts and gate-tampering tests remain part of normal pytest.

## Determinism

Each deterministic surface runs at least twice in a fresh logical context.
TruthBench compares normalized data, decisions, confidences, rationales, findings,
audit records, ordering, decision hashes, and trust scores.

Wall-clock duration, peak memory, generated timestamps, run IDs, lineage IDs, and
documented cryptographic randomness are excluded from the decision hash. Privacy
operations that intentionally use random salts must still make the same masking
decision, redact the same spans, disclose the randomness, and produce no raw PII.
Parity tests use an explicit fixed secret so output equality remains testable.

The default cleaning path, default semantic backend, validators, domain decisions,
trust scores, and generated code must be fully deterministic.

## Privacy Boundary

The raw-PII gate scans every sink that could escape the input data boundary:

- serialized reports and findings;
- action metadata, coerced-cell recovery records, examples, and audit logs;
- Copilot model context, narrative inputs, recommended code, and rendered report;
- plan JSON, validation JSON, domain logs, privacy reports, and quality reports;
- CLI stdout/stderr, Markdown, HTML, JSON, exception tables, and committed results;
- minimized reproductions and failure catalogues.

The scanner checks exact canaries plus case, whitespace, punctuation, encoding,
Unicode-normalization, and digit-only variants. Sensitive values are represented by
`[REDACTED]` and a one-way run-scoped digest. Explicit in-memory APIs whose purpose
is to return the user's own data remain usable, but their default serialization and
rendering paths must not leak raw PII. Any explicit `include_pii=True`-style escape
hatch is outside the release profile and receives a separate opt-in warning test.

## Runner Flow

For each profile, seed, fixture, and surface, the runner:

1. validates the complete oracle and fixture hash;
2. snapshots the input and protected columns;
3. computes pristine and adversarial trust scores;
4. invokes the surface through its public API;
5. extracts output, decisions, audit evidence, and backend disclosures;
6. verifies input immutability and protected-column byte identity;
7. emits normalized cell records and schema/row case records;
8. reruns deterministic surfaces and compares normalized hashes;
9. compares required backends;
10. scans every sink for PII canaries;
11. evaluates absolute gates;
12. minimizes every failure and writes privacy-safe reproduction artifacts;
13. writes versioned JSON and Markdown results atomically.

Infrastructure errors fail closed. A partial run cannot report passed gates.

## Mandatory Release Gates

The release command fails if any of these conditions is true:

1. **Valid-value corruption:** any `preserve` cell changes value, dtype, row
   identity, or semantic representation, or receives an error-severity false alarm.
2. **Protected-column modification:** any protected cell or protected schema
   property differs, even if another oracle labelled the value repairable.
3. **Raw PII leakage:** any canary or normalized variant appears in a scanned sink.
4. **Backend inconsistency:** any required backend produces a divergent decision,
   output, ordering, confidence class, rationale class, or audit outcome without an
   explicit permitted equivalence.
5. **Default non-determinism:** repeated default decisions or normalized outputs
   differ.
6. **Broken generated code:** generated code fails AST parsing, compilation, or
   controlled execution against its fixture, performs a network call, or exposes a
   PII canary.
7. **Unexplained high confidence:** a decision at confidence `>= 0.90` lacks a
   non-empty substantive rationale, rule/model provenance, and matching audit
   evidence. Boilerplate or whitespace-only text does not count.
8. **Trust inversion:** an adversarial/corrupted frame scores higher than its
   pristine source, or a destructive degenerate output is scored as trustworthy.

Additional correctness gates require complete cell labels, zero unexpected
exceptions, zero unresolved required backends, 100% audit coverage for mutations,
100% routing of `review` cases, exact repair outputs, zero mutations of `flag`
cases, and internally consistent result counts.

Absolute gates cannot be waived by a historical baseline. Baselines may detect
additional regressions but cannot make a mandatory failure pass.

## Failure Reproduction and Minimization

Every failure receives a stable ID derived from fixture version, surface, backend,
cell/case ID, gate, and normalized evidence. The reducer repeatedly tests the same
public surface while attempting, in order:

1. removal of unrelated fixtures and policies;
2. removal of unrelated columns;
3. removal of unrelated rows while retaining required role/context evidence;
4. removal of unrelated labelled mutations;
5. simplification of values, schemas, and policy sentences;
6. reduction to one backend and one repeat when those dimensions are not causal.

The reducer stops at a local one-minimal case or a configured evaluation budget.
It never changes the expected disposition or treats disappearance of the target
cell as success. The minimized artifact includes a sanitized frame, schema, policy,
config, exact command, expected/actual record, likely component, and result hash.

For every validated implementation failure, the development workflow is:

1. reproduce through TruthBench;
2. minimize;
3. identify the source-to-decision root cause;
4. add a focused failing pytest regression;
5. implement one root-cause fix;
6. run the focused test, affected suite, TruthBench, and full required tests;
7. update committed TruthBench results and failure catalogue;
8. record the fix, regression test, and remaining limitation.

Expected outputs are changed only when the written product contract changes and a
human explicitly approves that contract change. They are never changed to conceal
a failure.

## Result Artifacts

Committed artifacts live under `benchmarks/truthbench/results/`:

- `latest.json`: versioned complete machine-readable result;
- `latest.md`: concise metrics, gate status, failures, fixes, and limitations;
- `baseline.json`: fixture/result hashes and regression metrics, not gate waivers;
- `failures/<failure-id>.json`: minimized privacy-safe reproductions for unresolved
  or newly fixed failures;
- `README.md`: schema, reproduction, comparison, and update policy.

The JSON records repository commit, FreshData version, Python and dependency
versions, operating system, profile, seeds, fixture hashes, surface versions,
required backends, gate configuration, command, and normalized decision hashes.
Timestamps are metadata and never part of reproducibility hashes.

Result verification rejects unknown schema versions, missing fixtures, incomplete
runs, mismatched fixture hashes, inconsistent aggregates, leaked PII, absent gate
results, and claims not backed by cell records.

## CI and Release Integration

Normal pytest receives fast contract tests for models, fixture completeness,
adapters, comparators, gates, minimization, serialization, privacy scanning,
generated code, and every fixed implementation failure.

PR CI runs the deterministic release profile:

```bash
python -m benchmarks.truthbench run \
  --profile release \
  --backends pandas,polars,duckdb \
  --require-backends \
  --repeats 2 \
  --check
```

The production release workflow runs the same command against the exact resolved
release commit before building distributions. A failure prevents publication.

A scheduled extended profile adds Spark, FreshCore, more seeds, more generated
variants, alternative batch partitions, and larger fixtures. It has no
`continue-on-error`, no silent optional-dependency skip, and publishes its full
result artifacts.

The Makefile exposes `truthbench`, `truthbench-release`, and
`truthbench-extended` targets. Documentation gives exact local reproduction
commands and distinguishes semantic correctness from performance evidence.

## Testing Strategy

TruthBench tests are layered:

- model invariants and JSON schema round trips;
- complete cell-label and stable-ID validation;
- deterministic fixture and adversarial generator tests;
- exact comparator tests for values, dtypes, Unicode, missing values, dates, and
  protected bytes;
- adapter contract tests using small synthetic surface outputs;
- public-surface inventory completeness tests;
- gate-tampering tests proving each mandatory gate fails independently;
- PII scanner mutation tests for every normalized canary variant;
- backend parity and fallback-honesty tests;
- repeated-run determinism tests;
- generated-code parse, compile, controlled-execution, network-denial, and privacy
  tests;
- reducer tests proving the target failure remains after minimization;
- end-to-end smoke and full release-profile tests;
- one focused permanent regression for every reproduced FreshData defect.

Tests must not use broad exception catches, unconditional skips, weakened existing
assertions, or changed gold outputs to make a run green. Optional infrastructure is
selected by profile; within a selected profile it is required.

## Initial Audit Hypotheses

Repository inspection identified high-priority hypotheses that TruthBench must
attempt to reproduce before they are called defects:

- domain repair may occur after the core protected-column verification;
- post-domain report totals may describe the pre-domain frame;
- the 1,000-entry `coerced_cells` cap may also cap quarantine protection and allow
  later coerced cells to be imputed;
- semantic review mode may auto-apply high-confidence proposals from non-default
  backends;
- Validation Gauntlet semantic actions may not map to cells because semantic
  actions aggregate distinct values rather than carrying a row;
- Validation Gauntlet domain findings may be counted without earning cell-level
  detection evidence;
- column-level audit attribution may overstate cell-level completeness;
- serialized raw recovery values, text-cleaning originals, validation originals,
  plan examples, or privacy groups may cross the intended audit privacy boundary;
- generated timestamps and random masking salts may be confused with decision
  non-determinism unless normalization is explicit;
- plan signatures that sample only early rows may miss tail-only drift.

Each hypothesis is subject to the reproduce/minimize/root-cause workflow. No source
change is justified solely by inspection.

## Acceptance Criteria

TruthBench is complete when:

- all eight domain fixtures exist and every cell has one valid disposition;
- the public-surface manifest has no unclassified decision-bearing surface;
- every required comparison dimension appears in each applicable decision record;
- pandas, Polars, and DuckDB complete the release profile with no inconsistent
  decisions or undisclosed fallback;
- all eight mandatory gates have independent negative tests and pass the real run;
- all discovered implementation failures have minimized reproductions, root-cause
  fixes, focused pytest regressions, and updated benchmark results;
- generated code compiles and executes in the controlled offline harness;
- committed artifacts contain no planted PII;
- default behavior is deterministic under normalized repeat comparison;
- trust never increases after labelled corruption;
- the existing test suite, Validation Gauntlet, CleanBench, and benchmark tests are
  not weakened;
- PR CI and the production release workflow execute TruthBench as a required gate;
- final documentation lists implemented files, failures, fixes, commands, results,
  limitations, and evidence for every release gate.

