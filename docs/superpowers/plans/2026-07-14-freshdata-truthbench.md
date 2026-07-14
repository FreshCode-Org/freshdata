# FreshData TruthBench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, privacy-safe semantic red-team and regression system that gives every test cell a gold disposition, exercises every public FreshData decision surface, minimizes failures, and blocks releases on the eight approved safety gates.

**Architecture:** Add an independent `benchmarks.truthbench` package with immutable oracle models, eight deterministic domain fixtures, exact typed comparison, public-surface adapters, normalized records, privacy scanning, absolute gates, repeat/backend comparison, generated-code isolation, deterministic minimization, and atomic result reporting. Keep all model-assisted activity outside FreshData's runtime; release runs call only deterministic public APIs and Copilot with `provider=None`. Source fixes are allowed only after a TruthBench reproduction and a focused failing pytest regression establish the defect.

**Tech Stack:** Python 3.9+, pandas, Polars, DuckDB, FreshData public APIs, dataclasses/enums, hashlib/HMAC, `jsonschema`, `ast`, `subprocess`, pytest, GitHub Actions.

## Global Constraints

- Work only in `/Users/wilson/freshdata-worktrees/truthbench-jwd` on `feature/truthbench-jwd`; do not modify `/Users/wilson/freshdata`.
- Keep GPT/model/provider calls out of `fd.clean`, semantic routing, domain repair, privacy processing, trust scoring, CI, and TruthBench release execution.
- Do not weaken, skip, rebaseline, or rewrite existing tests or gold outputs to obtain a pass.
- Retain the approved baseline note: the two one-off `tests/test_benchmark.py` throughput failures are pre-existing environment timing noise; both exact reruns passed. Do not change their thresholds.
- A required backend that is unavailable or falls back is a failure, not a skip.
- Never write raw fixture PII into result JSON, Markdown, logs, exceptions, minimized artifacts, or committed baselines.
- Use one focused commit per task or validated source defect. Run the named focused test before every commit.
- For every implementation defect: reproduce, minimize, explain root cause, add a permanent regression, fix the root cause, rerun the affected suite, and update TruthBench artifacts.

## Planned File Structure

```text
benchmarks/truthbench/
  __init__.py                 # supported public harness API
  __main__.py                 # module entry point
  cli.py                      # release/extended/check command line
  models.py                   # immutable oracle/result/gate models
  exact.py                    # typed exact values and equality
  schema.py                   # JSON schema and aggregate validation
  privacy.py                  # canary variants, redaction, sink scanner
  inventory.py                # classified public-surface manifest
  fixtures/
    __init__.py               # registry and build_fixture
    base.py                   # fixture builder and completeness validation
    finance.py
    healthcare.py
    retail.py
    crm.py
    logistics.py
    government.py
    education.py
    insurance.py
  surfaces/
    __init__.py               # adapter registry
    base.py                   # adapter protocol and observation envelope
    cleaning.py               # clean/Cleaner/CSV/pipeline/plan/streaming
    validation.py             # field/suite/context/domain/contracts/text
    privacy.py                # PII/anonymization/privacy policy
    reporting.py              # reports/findings/export/render/CLI sinks
    backends.py               # pandas/Polars/DuckDB parity execution
    copilot.py                # provider=None and generated-code harness
  normalize.py                # observations -> per-cell decision records
  gates.py                    # absolute release gates
  determinism.py              # stable decision hashes and repeat comparison
  generated_code.py           # offline AST/compile/subprocess execution
  minimize.py                 # deterministic one-minimal failure reducer
  runner.py                   # end-to-end orchestration
  report.py                   # atomic JSON/Markdown/baseline output
  results/
    README.md
    baseline.json
    latest.json
    latest.md
    failures/.gitkeep
tests/truthbench/
  conftest.py
  test_models_exact.py
  test_fixtures.py
  test_privacy.py
  test_inventory.py
  test_surface_adapters.py
  test_normalize.py
  test_gates.py
  test_backends_determinism.py
  test_generated_code.py
  test_minimize.py
  test_runner_cli.py
tests/regressions/
  test_truthbench_domain_guard.py
  test_truthbench_domain_report.py
  test_truthbench_quarantine.py
  test_truthbench_semantic_review.py
  test_truthbench_findings_audit.py
docs/truthbench.md
.github/workflows/truthbench-extended.yml
```

Existing files changed only where the failing tests justify it: `src/freshdata/api.py`, `src/freshdata/report.py`, `src/freshdata/steps/dtypes.py`, `src/freshdata/engine/missing.py`, `src/freshdata/semantic/policy.py`, `src/freshdata/findings.py`, `Makefile`, `.github/workflows/ci.yml`, and `.github/workflows/release.yml`.

---

### Task 1: Establish the immutable oracle and exact typed comparator

**Files:**
- Create: `benchmarks/truthbench/__init__.py`
- Create: `benchmarks/truthbench/models.py`
- Create: `benchmarks/truthbench/exact.py`
- Create: `tests/truthbench/test_models_exact.py`

**Interfaces:** Consumes Python/pandas scalar values and fixture metadata. Produces frozen `GoldCell`, `CaseExpectation`, `DecisionRecord`, `GateResult`, and `RunResult` objects plus canonical JSON-safe typed values. No FreshData runtime dependency.

- [ ] Write failing tests covering the four-only disposition enum, stable cell IDs, sensitive-value redaction, JSON round trips, and exact distinctions among `"402.10"`, `402.1`, `np.float64(402.1)`, `None`, `pd.NA`, `NaN`, NFC/NFD Unicode, timezone-aware timestamps, leading-zero IDs, and categorical/string dtypes.

```python
def test_exact_values_do_not_use_gauntlet_canonicalization():
    assert not exact_equal("402.10", 402.1)
    assert not exact_equal(" AAPL", "AAPL")
    assert not exact_equal("AAPL", "aapl")
    assert exact_equal(pd.NA, pd.NA)

def test_sensitive_record_never_serializes_raw_value():
    cell = GoldCell.create("v1", "crm", "r7", "notes", "flag", sensitive=True)
    record = DecisionRecord.for_test(cell=cell, input_value="tb.person+7@example.invalid")
    payload = record.to_dict()
    assert "tb.person+7@example.invalid" not in json.dumps(payload)
    assert payload["input"]["display"] == "[REDACTED]"
```

- [ ] Run `PYTHONPATH=src python -m pytest tests/truthbench/test_models_exact.py -q`; expect import/collection failure because the package does not exist.

- [ ] Implement `Disposition(StrEnum)`, frozen dataclasses with explicit `schema_version=1`, `GoldCell.create()` stable ID generation, `TypedValue`, and a `canonical_json()` serializer that rejects non-finite JSON numbers and unknown types.

```python
class Disposition(str, Enum):
    PRESERVE = "preserve"
    REPAIR = "repair"
    FLAG = "flag"
    REVIEW = "review"

def exact_equal(left: Any, right: Any) -> bool:
    return encode_typed(left) == encode_typed(right)

def stable_digest(value: Any, *, key: bytes) -> str:
    encoded = canonical_json(encode_typed(value)).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()
```

- [ ] Ensure every `DecisionRecord` includes expected/actual disposition, input/output type, confidence, rationale, audit, trust, requested/actual backend, fallback events, and repeat hash fields; represent non-applicable dimensions explicitly as `None`, never by omission.

- [ ] Rerun the focused test; expect all tests to pass.

- [ ] Commit: `git add benchmarks/truthbench tests/truthbench/test_models_exact.py && git commit -m "feat: add TruthBench oracle models"`

### Task 2: Add JSON schema and aggregate integrity validation

**Files:**
- Create: `benchmarks/truthbench/schema.py`
- Create: `tests/truthbench/test_schema.py`

**Interfaces:** Consumes serialized `RunResult`. Produces either a validated payload or a precise `TruthBenchSchemaError`. It must reject partial runs and inconsistent aggregates before gates are evaluated.

- [ ] Write failing tests for unknown schema versions, absent fixtures/backends/gates, duplicate record IDs, non-finite confidence/trust values, fixture-hash mismatches, incorrect aggregate counts, and an `overall_passed=True` claim with a failed gate.

- [ ] Run `PYTHONPATH=src python -m pytest tests/truthbench/test_schema.py -q`; expect failures for missing schema validation.

- [ ] Implement a Draft 2020-12 schema with `additionalProperties: false` at result, record, gate, failure, and environment levels. Add semantic post-validation:

```python
def validate_run(payload: Mapping[str, Any]) -> None:
    jsonschema.Draft202012Validator(RESULT_SCHEMA).validate(payload)
    ids = [record["record_id"] for record in payload["records"]]
    if len(ids) != len(set(ids)):
        raise TruthBenchSchemaError("duplicate decision record id")
    if payload["summary"]["records"] != len(ids):
        raise TruthBenchSchemaError("record aggregate does not match records")
    passed = all(gate["passed"] for gate in payload["gates"])
    if payload["summary"]["overall_passed"] is not passed:
        raise TruthBenchSchemaError("overall gate claim is inconsistent")
```

- [ ] Rerun the focused test; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/schema.py tests/truthbench/test_schema.py && git commit -m "feat: validate TruthBench result integrity"`

### Task 3: Build fixture infrastructure with complete physical-cell labels

**Files:**
- Create: `benchmarks/truthbench/fixtures/__init__.py`
- Create: `benchmarks/truthbench/fixtures/base.py`
- Create: `tests/truthbench/conftest.py`
- Create: `tests/truthbench/test_fixtures.py`

**Interfaces:** Consumes a domain name and fixed seed. Produces a `TruthFixture` holding pristine/adversarial pandas frames, one `GoldCell` for every physical adversarial-frame cell, schema, policy, protected columns, PII canaries, row cases, schema cases, and a deterministic fixture hash.

- [ ] Write failing fixture invariant tests. The label count must equal `rows * columns`; every `(row_id, column)` appears exactly once; injected cases overwrite the default `preserve` label; every repair has an exact typed output; every sensitive cell has a canary ID; row/schema expectations never stand in for cell labels.

```python
@pytest.mark.parametrize("domain", DOMAINS)
def test_every_physical_cell_has_exactly_one_label(domain):
    fixture = build_fixture(domain, seed=1729)
    expected = {(str(row), str(col)) for row in fixture.frame.index for col in fixture.frame}
    actual = {(cell.row_id, cell.column) for cell in fixture.cells}
    assert actual == expected
    assert len(fixture.cells) == len(expected)
    fixture.validate()
```

- [ ] Run `PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q`; expect import failure.

- [ ] Implement `FixtureBuilder` so it starts with a complete `preserve` label matrix and `inject()` atomically changes a value and replaces exactly one label. Reject missing/duplicate row IDs, unknown columns, non-synthetic PII domains, and contradictory repair outputs.

```python
class FixtureBuilder:
    def inject(self, row_id: str, column: str, value: Any, disposition: Disposition,
               *, expected: Any = UNSET, family: str, sensitive: bool = False) -> None:
        key = (row_id, column)
        if key not in self._labels:
            raise FixtureError(f"unknown cell {key}")
        self.frame.at[row_id, column] = value
        self._labels[key] = GoldCell.create(
            self.version, self.domain, row_id, column, disposition,
            expected_output=expected, family=family, sensitive=sensitive,
        )
```

- [ ] Make `fixture_hash` cover typed pristine/adversarial values, labels, schema, policy, row/schema cases, and protected columns, excluding object identity and build time.

- [ ] Rerun the focused test with a temporary minimal fixture registered in `conftest.py`; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/fixtures tests/truthbench/conftest.py tests/truthbench/test_fixtures.py && git commit -m "feat: add complete TruthBench fixture oracle"`

### Task 4: Add finance, healthcare, retail, and CRM gold datasets

**Files:**
- Create: `benchmarks/truthbench/fixtures/finance.py`
- Create: `benchmarks/truthbench/fixtures/healthcare.py`
- Create: `benchmarks/truthbench/fixtures/retail.py`
- Create: `benchmarks/truthbench/fixtures/crm.py`
- Modify: `benchmarks/truthbench/fixtures/__init__.py`
- Modify: `tests/truthbench/test_fixtures.py`

**Interfaces:** Each `build(seed: int) -> TruthFixture` is pure and deterministic. Cases use only reserved synthetic namespaces such as `.invalid`, `555-01xx`, and explicit `TB-*` identifiers.

- [ ] Add failing domain-content tests requiring each disposition in every domain and the following exact adversarial families:

| Domain | Required cases |
|---|---|
| finance | `apple` in price, `Apple` company, `AAPL` ticker; `0.00`; negative/extreme values; `₹1,23,456.70`; USD/EUR/INR conflict; `01/02/2025`; invisible PII in memo; tail-row account canary; protected ticker policy conflict |
| healthcare | valid rare ICD/LOINC; `98.6` in Celsius; `5 mg`/`5000 mcg`; partial/FHIR/impossible dates; decomposed Unicode name; MRN/phone/PHI in notes; protected DOB repair conflict |
| retail | leading-zero SKU/GTIN; free item and return quantity; mixed decimal/grouping/currency; HTML/entity/mojibake review; multilingual product; card/email in review; added/reordered/type-drifted columns |
| CRM | Unicode/combining names; `.invalid` email; spaced phone; ambiguous country/language/date; lifecycle contradiction; zero-width email and hidden SSN; `apple` lead source versus `Apple` employer; protected customer ID |

- [ ] Run `PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q`; expect four missing builders/content failures.

- [ ] Implement 16-row pristine frames per domain with stable string indexes and at least 12 injected cases per domain. Use fixed reference date `2026-01-15`, UTC timezone, and explicit locale metadata. Keep unusual valid values labelled `preserve`, deterministic representational fixes labelled `repair`, unsafe values labelled `flag`, and ambiguous/policy-conflicted values labelled `review`.

- [ ] Add row cases for exact duplicates and schema cases for added/removed/renamed/reordered/type-drifted columns. Do not label a removed row/column as a cell outcome.

- [ ] Assert byte-for-byte deterministic frame serialization and fixture hashes over two builds for seeds `1729` and `2718`.

- [ ] Rerun the fixture tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py && git commit -m "feat: add first TruthBench domain corpus"`

### Task 5: Add logistics, government, education, and insurance gold datasets

**Files:**
- Create: `benchmarks/truthbench/fixtures/logistics.py`
- Create: `benchmarks/truthbench/fixtures/government.py`
- Create: `benchmarks/truthbench/fixtures/education.py`
- Create: `benchmarks/truthbench/fixtures/insurance.py`
- Modify: `benchmarks/truthbench/fixtures/__init__.py`
- Modify: `tests/truthbench/test_fixtures.py`

**Interfaces:** Same builder contract as Task 4; registry order is the stable alphabetical order `crm, education, finance, government, healthcare, insurance, logistics, retail`.

- [ ] Add failing content tests for:

| Domain | Required cases |
|---|---|
| logistics | rare valid UN/LOCODE-like code; kg/lb and C/F; cross-timezone delivery window; 24:00-like transport time; address PII; late tracking canary; protected shipment ID conflict |
| government | leading-zero district/case IDs; Indian/international grouping; fiscal versus calendar year; multilingual agency names; restricted national ID in notes; mixed legacy encoding; contradictory retention/repair policy |
| education | student IDs; letter/percentage/GPA scales; school-year ambiguity; valid zero score; enrollment date ordering; guardian email/phone and FERPA notes; protected student ID and grade policy conflict |
| insurance | policy/claim IDs; premium/reserve currency conflict; negative reserve review; incident/report date ordering; state transition contradiction; claimant PII and medical loss text; protected policy number conflict |

- [ ] Run the fixture suite and expect four missing-builder/content failures.

- [ ] Implement the four deterministic builders using the same 16-row/fixed-reference conventions, complete labels, row cases, schema drift, mixed language/encoding, and PII tail cases.

- [ ] Add a corpus-level assertion that all required trap categories occur across the eight fixtures and every domain contains all four dispositions.

- [ ] Rerun `PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q`; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py && git commit -m "feat: complete eight-domain TruthBench corpus"`

### Task 6: Implement privacy-safe values and exhaustive sink scanning

**Files:**
- Create: `benchmarks/truthbench/privacy.py`
- Create: `tests/truthbench/test_privacy.py`

**Interfaces:** Consumes fixture canaries and arbitrary nested sink values (`str`, bytes, mappings, sequences, dataclasses, pandas objects). Produces redacted values/digests and precise leak locations without repeating leaked text.

- [ ] Write failing mutation tests for literal, case-folded, whitespace-stripped, punctuation-stripped, digit-only, URL-decoded, HTML-unescaped, UTF-8 bytes, NFKC/NFC/NFD, zero-width-removed, and JSON-escaped canary variants.

```python
@pytest.mark.parametrize("mutate", CANARY_MUTATORS)
def test_scanner_finds_every_normalized_variant(mutate):
    scanner = SinkScanner.from_canaries({"crm-email": "tb.person+7@example.invalid"})
    leaks = scanner.scan({"report": [mutate("tb.person+7@example.invalid")]})
    assert [(leak.canary_id, leak.path) for leak in leaks] == [("crm-email", "$.report[0]")]
    assert "tb.person" not in repr(leaks)
```

- [ ] Run the focused tests; expect import failure.

- [ ] Implement normalization as named transforms and scan both decoded text and byte hex/escape forms. Use run-scoped HMAC-SHA256 digests and `Leak(canary_id, variant, path)`; never store a matched substring.

- [ ] Add scanners for exception text, `CleanReport.to_dict()`, action metadata, `coerced_cells`, findings, plan JSON, validation/domain/privacy/Copilot reports, generated code, stdout/stderr, Markdown, HTML, JSON, and failure artifacts.

- [ ] Verify scanner self-test rejects a result that contains its own canary and passes a correctly redacted sink.

- [ ] Rerun focused tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/privacy.py tests/truthbench/test_privacy.py && git commit -m "feat: add TruthBench PII leak scanner"`

### Task 7: Classify every public surface and define the adapter protocol

**Files:**
- Create: `benchmarks/truthbench/inventory.py`
- Create: `benchmarks/truthbench/surfaces/__init__.py`
- Create: `benchmarks/truthbench/surfaces/base.py`
- Create: `tests/truthbench/test_inventory.py`
- Create: `tests/truthbench/test_surface_adapters.py`

**Interfaces:** Consumes `freshdata.__all__`, lazy export registries, domain registry, experimental Copilot export, and enterprise CLI parser. Produces one classification for every public name/command: `decision`, `sink`, `explicit-transform`, `data-model`, `configuration`, `registration`, or `out-of-scope-with-reason`. Decision/sink entries must reference an adapter.

- [ ] Write failing inventory tests comparing the manifest to `fd.__dir__()`, `_ENTERPRISE_EXPORTS`, `_INTEGRATION_EXPORTS`, `_LEARNING_EXPORTS`, `_VALIDATION_EXPORTS`, bundled `domains.available()`, and CLI subcommands. Fail on a new unclassified public name or an adapterless decision/sink.

- [ ] Run focused tests; expect missing manifest/protocol failures.

- [ ] Implement frozen `SurfaceSpec` and abstract `SurfaceAdapter.observe(fixture, context) -> SurfaceObservation`. `SurfaceObservation` must carry output frame, raw decisions, audit sinks, trust, backend disclosure, generated code, captured stdout/stderr, and unexpected exception details.

```python
@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    version: int
    classification: SurfaceClass
    adapter: str | None
    mutates: bool
    deterministic: bool
    backend_parity: bool
    rationale: str
```

- [ ] Explicitly classify low-level `fill_missing`, `remove_outliers`, `resolve_duplicates`, `group_aggregate`, display setters, plugin registration, token vault primitives, and explicit detokenization as caller-directed surfaces; they receive safety/input-output checks but no inferred disposition credit.

- [ ] Rerun focused tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/inventory.py benchmarks/truthbench/surfaces tests/truthbench/test_inventory.py tests/truthbench/test_surface_adapters.py && git commit -m "feat: inventory FreshData decision surfaces"`

### Task 8: Implement cleaning, validation, domain, text, and streaming adapters

**Files:**
- Create: `benchmarks/truthbench/surfaces/cleaning.py`
- Create: `benchmarks/truthbench/surfaces/validation.py`
- Modify: `benchmarks/truthbench/surfaces/__init__.py`
- Modify: `tests/truthbench/test_surface_adapters.py`

**Interfaces:** Calls public FreshData APIs only. Produces `SurfaceObservation` without grading it. Adapter-specific expected mappings are explicit: mutators repair only `repair`; read-only validators never mutate; `flag` and `review` remain unchanged by default.

- [ ] Write failing contract tests for `fd.clean`, `Cleaner.clean`, `clean_csv`, `CleanResult`, fluent `pipeline`, `suggest_plan`/`apply_plan`, `validate_fields`/`apply_field_policy`, `fd.validate`/`ValidationSuite`, context compile/validate, bundled domain validators, deterministic semantic assist/review/auto, text clean/lint, and fixed-partition `StreamingCleaner`/`clean_timeseries`.

- [ ] Run focused tests; expect missing adapters.

- [ ] Implement adapters with narrow exception capture at the top runner boundary only. Preserve the exact exception type and a scanner-sanitized message. Use public methods for reports and decisions; do not import internal cleaning functions.

- [ ] Map aggregate actions to cells only when metadata supplies a row, a documented value mapping identifies exact matching cells, or the returned validation/domain rule supplies violation rows. Never award a column-wide action to every labelled cell.

- [ ] Capture input snapshot, output, `report.actions`, findings, `coerced_cells`, domain reports/repairs, trust fields, plan hashes, recommendations, and rendered representations as separate scan sinks.

- [ ] For streaming, run identical data through fixed partitions `(8, 8)` and `(5, 5, 6)` and retain batch/rolling/cumulative decisions for later parity comparison.

- [ ] Rerun focused adapter tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/surfaces tests/truthbench/test_surface_adapters.py && git commit -m "feat: observe core FreshData surfaces"`

### Task 9: Implement privacy, trust, reporting, export, CLI, and Copilot adapters

**Files:**
- Create: `benchmarks/truthbench/surfaces/privacy.py`
- Create: `benchmarks/truthbench/surfaces/reporting.py`
- Create: `benchmarks/truthbench/surfaces/copilot.py`
- Modify: `benchmarks/truthbench/surfaces/__init__.py`
- Modify: `tests/truthbench/test_surface_adapters.py`

**Interfaces:** Calls `detect_pii`, anonymization/privacy policy, trust/quality/compliance/insight, findings/exporters/renderers/CLI, and `experimental.ai_copilot.analyze_dataset(provider=None)`. Produces observations plus every externally visible sink.

- [ ] Add failing adapter tests for PII detection, anonymization with a fixed test key, privacy policy, k-anonymity, `compute_trust_score`, quality/debt/insight/compliance/stakeholder reports, `to_dict`/`to_frame`/`to_findings`, JSON/Markdown/HTML/Peel/plain rendering, quality-ops/dbt/GX/exception exporters, CLI stdout/stderr, and Copilot provider-free analysis.

- [ ] Run focused tests; expect missing adapters.

- [ ] Implement adapters. Copilot must receive `provider=None`; monkeypatch a sentinel provider/network function that fails if called. Record prompt/model context, recommended code, audit, narrative, and all render forms as privacy sinks.

- [ ] Use a fixed per-test masking secret for deterministic parity, while separately asserting default random masking discloses randomness and never leaks raw PII.

- [ ] Capture trust on pristine, adversarial, cleaned, and deliberately destructive frames. A destructive control is a same-shape constant/null frame, not an empty frame that bypasses metrics.

- [ ] Rerun focused tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/surfaces tests/truthbench/test_surface_adapters.py && git commit -m "feat: observe privacy reporting and Copilot surfaces"`

### Task 10: Add required-backend execution and honest parity checks

**Files:**
- Create: `benchmarks/truthbench/surfaces/backends.py`
- Create: `tests/truthbench/test_backends_determinism.py`
- Modify: `benchmarks/truthbench/surfaces/__init__.py`

**Interfaces:** Consumes a fixture and common-native-subset `CleanConfig`. Produces pandas-normalized observations for requested backends `pandas`, `polars`, and `duckdb` with `fallback_policy="error"`, requested/actual backend, fallback events, row identity/order, and report differences.

- [ ] Write failing tests proving a missing required backend, unexpected fallback, requested/actual mismatch, row reorder, dtype/value divergence, action divergence, or undisclosed backend difference fails parity. Tamper adapters in tests; never rely on an actually missing dependency.

- [ ] Run focused tests; expect missing backend adapter.

- [ ] Implement backend preflight with `importlib.metadata.version`, public `fd.clean(..., engine=backend, fallback_policy="error", return_report=True)`, native-to-pandas conversion, and explicit equivalence rules limited to approved representation differences.

- [ ] Ensure pandas is the reference but not automatically correct: all three outputs are independently scored against gold before parity comparison.

- [ ] Add extended-profile contracts for Spark/FreshCore. Normal pytest tests adapter/gate behavior using fakes; the extended workflow requires real infrastructure.

- [ ] Rerun focused tests with installed pandas/Polars/DuckDB; expect pass and no fallback events.

- [ ] Commit: `git add benchmarks/truthbench/surfaces/backends.py tests/truthbench/test_backends_determinism.py benchmarks/truthbench/surfaces/__init__.py && git commit -m "feat: enforce TruthBench backend parity"`

### Task 11: Normalize decisions and implement all absolute gates

**Files:**
- Create: `benchmarks/truthbench/normalize.py`
- Create: `benchmarks/truthbench/gates.py`
- Create: `tests/truthbench/test_normalize.py`
- Create: `tests/truthbench/test_gates.py`

**Interfaces:** Consumes fixture gold plus a `SurfaceObservation`. Produces one normalized `DecisionRecord` per `(surface, backend, repeat, cell)` and `CaseRecord` per row/schema case, then evaluates gates independently of any baseline.

- [ ] Write failing normalization tests for exact output/dtype, detected versus mutated, quarantine/review routing, cell-level audit IDs, confidence/rationale/provenance extraction, trust delta, backend disclosure, sensitive values, and non-applicable fields.

- [ ] Write one independent tamper test for each mandatory gate:

```python
@pytest.mark.parametrize("mutator, gate", [
    (corrupt_preserve, "valid_value_corruption"),
    (modify_protected, "protected_column_modification"),
    (leak_canary, "raw_pii_leakage"),
    (diverge_backend, "backend_inconsistency"),
    (change_repeat, "default_nondeterminism"),
    (break_generated_code, "broken_generated_code"),
    (remove_high_confidence_explanation, "unexplained_high_confidence"),
    (invert_trust, "trust_inversion"),
])
def test_each_mandatory_gate_fails_independently(passing_run, mutator, gate):
    result = evaluate_gates(mutator(passing_run))
    assert failed_gate_names(result) == {gate}
```

- [ ] Run focused tests; expect missing normalizer/gates.

- [ ] Implement surface-aware disposition mapping. Mutators must exactly repair `repair`, preserve `preserve`, and not mutate `flag`/`review`. Validators receive detection credit without mutation. PII adapters are graded only on PII-labelled cells plus false positives. Explicit transforms are graded on requested behavior and protected/privacy invariants.

- [ ] Implement the eight named gates plus completeness, unexpected exception, required-backend, mutation-audit, review-routing, exact-repair, flag-mutation, input-mutation, and aggregate-consistency gates. High confidence means `>= 0.90`; substantive explanations require non-boilerplate rationale, non-empty rule/model provenance, and a matching audit record.

- [ ] Make gate evaluation fail closed when records are absent, a surface is unexecuted, schema validation failed, or a run is partial. Baseline comparison may add failures but cannot clear one.

- [ ] Rerun focused tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/normalize.py benchmarks/truthbench/gates.py tests/truthbench/test_normalize.py tests/truthbench/test_gates.py && git commit -m "feat: enforce TruthBench release gates"`

### Task 12: Add decision determinism and controlled generated-code execution

**Files:**
- Create: `benchmarks/truthbench/determinism.py`
- Create: `benchmarks/truthbench/generated_code.py`
- Modify: `tests/truthbench/test_backends_determinism.py`
- Create: `tests/truthbench/test_generated_code.py`

**Interfaces:** Consumes normalized observations and Copilot code. Produces stable hashes/diffs and a subprocess result that proves AST parse, compilation, offline execution, expected output, and PII safety.

- [ ] Write failing tests showing hashes ignore duration, peak memory, timestamp, run/lineage IDs, and documented salt bytes but detect decision/order/output/confidence/rationale/audit/trust changes.

- [ ] Write failing generated-code tests for syntax error, compile error, missing input/output contract, filesystem escape, imports outside the allowlist, `socket`/HTTP access, subprocess creation, raw PII literal, runtime failure, timeout, and wrong cleaned output.

- [ ] Run the focused tests; expect missing modules.

- [ ] Implement recursive normalization with an explicit excluded-field set and sorted canonical JSON. Reject callers attempting to exclude a decision-bearing field.

- [ ] Parse code with `ast.parse`, reject unsafe nodes/imports/calls, compile in-process, then execute in `python -I` with a 10-second timeout, a temporary working directory, environment allowlist, network-denial bootstrap, and serialized synthetic fixture input. Compare the produced frame/report to the adapter contract and scan code/stdout/stderr/artifacts for canaries.

- [ ] Rerun focused tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench/determinism.py benchmarks/truthbench/generated_code.py tests/truthbench/test_backends_determinism.py tests/truthbench/test_generated_code.py && git commit -m "feat: verify deterministic decisions and generated code"`

### Task 13: Implement runner, deterministic minimizer, CLI, and atomic reports

**Files:**
- Create: `benchmarks/truthbench/runner.py`
- Create: `benchmarks/truthbench/minimize.py`
- Create: `benchmarks/truthbench/report.py`
- Create: `benchmarks/truthbench/cli.py`
- Create: `benchmarks/truthbench/__main__.py`
- Create: `tests/truthbench/test_minimize.py`
- Create: `tests/truthbench/test_runner_cli.py`

**Interfaces:** CLI accepts `run --profile release|extended --backends ... --require-backends --repeats N --check`. Runner returns a complete `RunResult`; minimizer consumes one reproducible `GateFailure` and predicate; reporter writes validated, PII-scanned JSON/Markdown atomically.

- [ ] Write failing reducer tests proving removal order is fixtures/policies, columns, rows, mutations, value/schema/policy simplification, then backend/repeat. The target cell and expected disposition must survive and the same failure ID must still reproduce.

- [ ] Write failing CLI tests for exact option parsing, unknown backend/profile, missing required backend, partial-run failure, nonzero `--check`, successful atomic replacement, and refusal to write a leaking artifact.

- [ ] Run focused tests; expect missing runner/CLI/minimizer.

- [ ] Implement the approved 13-stage runner flow. Use a fresh adapter context per repeat, deterministic fixture/surface ordering, no broad exception suppression, and an infrastructure-failure record when observation cannot complete.

- [ ] Implement one-minimal reduction with an evaluation budget of 250 calls and a cache keyed by typed fixture/config/surface/backend hash. Generate failure IDs from fixture version, surface, backend, cell/case, gate, and normalized evidence.

- [ ] Implement `write_atomic()` with same-directory temporary files, `fsync`, schema validation, sink scan, then `os.replace`. Render Markdown only from already-redacted JSON.

- [ ] Add CLI defaults exactly matching the release command:

```bash
python -m benchmarks.truthbench run --profile release \
  --backends pandas,polars,duckdb --require-backends --repeats 2 --check
```

- [ ] Rerun focused tests; expect pass.

- [ ] Commit: `git add benchmarks/truthbench tests/truthbench/test_minimize.py tests/truthbench/test_runner_cli.py && git commit -m "feat: run and minimize TruthBench failures"`

### Task 14: Reproduce and resolve the initial audit hypotheses

**Files:**
- Create/modify only the regression and source files proven necessary by each reproduction.
- Expected focused files: `tests/regressions/test_truthbench_domain_guard.py`, `tests/regressions/test_truthbench_domain_report.py`, `tests/regressions/test_truthbench_quarantine.py`, `tests/regressions/test_truthbench_semantic_review.py`, `tests/regressions/test_truthbench_findings_audit.py`, `src/freshdata/api.py`, `src/freshdata/report.py`, `src/freshdata/steps/dtypes.py`, `src/freshdata/engine/missing.py`, `src/freshdata/semantic/policy.py`, `src/freshdata/findings.py`.

**Interfaces:** Consumes real TruthBench failures. Produces minimized privacy-safe reproductions, root-cause notes, permanent regressions, minimal source fixes, and green affected suites. A hypothesis that does not reproduce gets a passing adversarial coverage test and no source change.

- [ ] Run the release profile without `--check`, capture all failures, and run the minimizer for each:

```bash
PYTHONPATH=src python -m benchmarks.truthbench run --profile release \
  --backends pandas,polars,duckdb --require-backends --repeats 2
```

- [ ] For domain protection, add a focused test where a compiled policy marks a repairable domain column immutable. Expect byte-identical output or `ProtectedColumnError`; reproduce before changing `api.py`. If reproduced, snapshot the resolved protected columns before `run_domain`, verify the repaired frame afterward, and record the guard action.

- [ ] For post-domain report state, assert `rows_after`, `cols_after`, `memory_after`, and `missing_after` describe the returned domain-repaired frame. If reproduced, centralize final metric refresh and call it after `_fold_domain_outcome`.

- [ ] For late coerced cells, build 1,020 parse casualties where a row after index 1,000 would otherwise be imputed. If reproduced, add a non-serialized `_coerced_rows: dict[str, set[Any]]` to `CleanReport`, populate every lost index in `_record_coerced`, keep `coerced_cells` recovery values capped, and make `_quarantined_rows` use the complete internal index set.

- [ ] For semantic review, inject a high-confidence low-risk proposal from `embedding`, `profile`, `memory`, and `plugin:*` backends. If any applies in review mode, restrict review-mode auto-application to deterministic built-in provenance; keep non-default proposals suggested with human review. Preserve `auto` behavior subject to its existing safety gates.

- [ ] For finding audit projection, assert a medium/high-risk action retains confidence, rationale, model/rule ID, status, reversible, human-review, and safe metadata in `QualityFinding.extra`. If lost, pass complete action dictionaries from `CleanReport.to_findings()` and explicitly copy those safe audit fields in `findings_from_dict()`.

- [ ] For every reproduced defect, follow this exact loop separately: run the failing focused test; save minimized redacted failure JSON; implement one root-cause fix; run the focused test; run the affected module suite; rerun its TruthBench case; commit with a concise `fix:` message naming the observed root cause.

- [ ] Exercise the remaining hypotheses—Gauntlet row mapping, domain cell evidence, audit over-attribution, raw recovery leakage, random salt normalization, and tail-only plan drift—through TruthBench. Fix FreshData only if the new exact benchmark reproduces a public-contract violation; otherwise retain the adversarial test as coverage.

- [ ] After each fix, scan the patch for broad catches/skips/changed gold:

```bash
git diff origin/main -- tests src benchmarks/truthbench | rg "pytest\.skip|xfail|except Exception|COERCED_CELLS_CAP|expected|Disposition"
```

- [ ] Commit each validated fix independently; do not combine unrelated root causes.

### Task 15: Commit benchmark artifacts and document every result

**Files:**
- Create: `benchmarks/truthbench/results/README.md`
- Create: `benchmarks/truthbench/results/baseline.json`
- Create: `benchmarks/truthbench/results/latest.json`
- Create: `benchmarks/truthbench/results/latest.md`
- Create/update: `benchmarks/truthbench/results/failures/*.json`
- Create: `docs/truthbench.md`
- Modify: `README.md`

**Interfaces:** Consumes the clean release run. Produces versioned, schema-valid, PII-free evidence and user documentation. Baseline detects regressions but cannot waive an absolute gate.

- [ ] Add failing tests that committed artifacts validate, match current fixture hashes/surface manifest, contain all eight domain names and all gate results, contain no canary variants, and do not claim success for unresolved failures.

- [ ] Run those tests; expect missing artifact failures.

- [ ] Run the full release profile with `--check`; write `latest.json`/`latest.md`, then intentionally copy the verified result to `baseline.json` only after every absolute gate passes.

- [ ] Document architecture, disposition meanings, surface mapping, exact comparator, privacy model, failure reproduction, minimization, baseline update policy, local commands, and the explicit prohibition on LLMs in the runtime path.

- [ ] In `latest.md`, list implemented files, each discovered failure and root cause, each fix/regression, exact commands, per-domain/per-surface/backend results, every gate's evidence counts, remaining limitations, and the two pre-existing timing flakes with their passing exact reruns.

- [ ] Rerun artifact tests and `rg` every planted canary across committed result/docs paths; expect zero matches outside fixture source and scanner test data.

- [ ] Commit: `git add benchmarks/truthbench/results docs/truthbench.md README.md tests/truthbench && git commit -m "docs: publish TruthBench release evidence"`

### Task 16: Make TruthBench a mandatory PR and production release gate

**Files:**
- Modify: `Makefile`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release.yml`
- Create: `.github/workflows/truthbench-extended.yml`
- Modify: `pyproject.toml` only if a dedicated benchmark extra is required after verification.

**Interfaces:** CI/release consumes the repository and required dependencies. Produces a hard pass/fail before packaging/publication plus scheduled extended artifacts.

- [ ] Add failing workflow/Makefile contract tests asserting exact required command, no `continue-on-error`, no skip-on-missing-backend branch, and TruthBench execution before `python -m build`/publication.

- [ ] Add targets:

```make
.PHONY: truthbench truthbench-release truthbench-extended
truthbench:
	PYTHONPATH=src python -m benchmarks.truthbench run --profile release --backends pandas,polars,duckdb --require-backends --repeats 2
truthbench-release:
	PYTHONPATH=src python -m benchmarks.truthbench run --profile release --backends pandas,polars,duckdb --require-backends --repeats 2 --check
truthbench-extended:
	PYTHONPATH=src python -m benchmarks.truthbench run --profile extended --backends pandas,polars,duckdb,spark,freshcore --require-backends --repeats 2 --check
```

- [ ] Install dev/out-of-core dependencies in PR/release jobs and add `make truthbench-release` after fast pytest and before build. The scheduled extended job provisions JVM/native infrastructure and uploads `latest.json`, `latest.md`, and minimized failures on success or failure.

- [ ] Run workflow syntax/contract tests plus `make -n truthbench-release`; expect the exact command.

- [ ] Commit: `git add Makefile .github/workflows tests pyproject.toml && git commit -m "ci: require TruthBench release gates"`

### Task 17: Final verification and self-review

**Files:**
- Review all changes from `origin/main...HEAD`.
- Update `benchmarks/truthbench/results/latest.json` and `latest.md` only if verification evidence changed.

**Interfaces:** Consumes the finished branch. Produces reproducible evidence that the specification and every release gate are satisfied without test weakening.

- [ ] Run focused TruthBench tests:

```bash
PYTHONPATH=src python -m pytest tests/truthbench tests/regressions/test_truthbench_*.py -q
```

Expected: all pass, no skips.

- [ ] Run the mandatory release benchmark:

```bash
PYTHONPATH=src python -m benchmarks.truthbench run --profile release \
  --backends pandas,polars,duckdb --require-backends --repeats 2 --check
```

Expected: exit 0; all eight mandatory and additional completeness gates pass; no backend fallback; zero raw-PII findings.

- [ ] Run existing correctness systems unchanged:

```bash
PYTHONPATH=src python -m benchmarks.gauntlet run --check
PYTHONPATH=src python benchmarks/public_benchmark.py
PYTHONPATH=src python -m pytest -m "not online and not large"
```

Expected: Gauntlet gates pass, public/CleanBench benchmark passes, pytest passes apart from no accepted failures. If the two recorded timing tests flake, rerun their exact node IDs and report both outputs; do not alter thresholds.

- [ ] Run static/package checks:

```bash
ruff check .
mypy src/freshdata
python -m build
python -m twine check dist/*
```

Expected: all exit 0.

- [ ] Perform specification traceability review: map every approved design section and acceptance criterion to an implemented file plus a passing test/result field.

- [ ] Scan for incomplete implementation and policy violations:

```bash
rg -n "TODO|FIXME|NotImplemented|pass$|pytest\.skip|xfail|continue-on-error" benchmarks/truthbench tests/truthbench tests/regressions .github/workflows
rg -n "openai|anthropic|provider=|requests\.|httpx\.|urllib" benchmarks/truthbench src/freshdata
```

Expected: no placeholders, hidden skips, release `continue-on-error`, or network/model calls in the benchmark/default path; the Copilot adapter contains only the explicit `provider=None` assertion and network-denial test harness.

- [ ] Review `git diff --check`, `git status --short`, and the full commit list. Confirm the original checkout remains unchanged.

- [ ] If verification changed result evidence, regenerate it, rerun schema/PII checks, and commit `test: finalize TruthBench release evidence`.

- [ ] Prepare the final handoff with implemented files, discovered failures, fixes, commands and outputs, benchmark totals, limitations, and per-gate evidence.
