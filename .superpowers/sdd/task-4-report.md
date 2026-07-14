# Task 4 Report: Add finance, healthcare, retail, and CRM gold datasets

## Status

DONE. The four deterministic 16-row domain builders are registered and covered by focused and adjacent TruthBench tests.

## TDD evidence

### RED

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q --no-cov
```

Observed 16 failures before implementation: each new domain was absent from the fixture registry (`FixtureError: unknown fixture domain`).

### GREEN

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q --no-cov
```

Result: `28 passed`.

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py tests/truthbench/test_models_exact.py tests/truthbench/test_schema.py -q --no-cov
```

Result: all focused, model, and schema tests green.

Static checks:

```text
python -m ruff check benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py
python -m ruff format --check benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py
mypy --ignore-missing-imports --explicit-package-bases benchmarks/truthbench/fixtures tests/truthbench/conftest.py tests/truthbench/test_fixtures.py
git diff --check
```

All passed.

## Files

- `benchmarks/truthbench/fixtures/finance.py` — finance frame and oracle labels for Apple semantic traps, numeric/currency/date defects, protected ticker conflict, and synthetic PII canaries.
- `benchmarks/truthbench/fixtures/healthcare.py` — healthcare frame and oracle labels for ICD/LOINC, temperature and dose units, FHIR/partial/impossible dates, Unicode, and PHI canaries.
- `benchmarks/truthbench/fixtures/retail.py` — retail frame and oracle labels for leading-zero identifiers, free/return quantities, locale currency formats, mojibake/HTML, multilingual values, and review PII.
- `benchmarks/truthbench/fixtures/crm.py` — CRM frame and oracle labels for Unicode/contact ambiguity, lifecycle contradiction, zero-width/hidden PII, Apple semantic traps, and protected IDs.
- `benchmarks/truthbench/fixtures/__init__.py` — registry entries and stable domain order.
- `tests/truthbench/test_fixtures.py` — domain completeness, disposition, deterministic-seed, and adversarial marker tests.

## Self-review

- Every builder emits exactly 16 rows with stable string indexes, fixed `2026-01-15` UTC reference metadata, explicit locale metadata, complete physical-cell labels, and at least 12 injected adversarial cells.
- All four dispositions are represented per domain. Row cases cover exact duplicates and removed rows; schema cases cover added, removed, renamed, reordered, and type-drifted columns without mislabeling absent cells.
- Sensitive values are whole-value synthetic `.invalid`, `TB-*`, or `555-01xx` forms, preserving the base builder's redaction and canary invariants. Zero-width examples are intentionally non-sensitive representation traps.
- Seed values are included only in a deterministic batch marker; same-seed builds produce byte-stable serialized oracle payloads and identical fixture hashes for seeds `1729` and `2718`.
- No FreshData runtime, LLM/provider, network, or external data dependency is used.

## Concerns

- The legacy `minimal` fixture remains registered for backwards compatibility; the four Task 4 domains are appended in stable order. A later registry task may choose to retire `minimal` once its callers migrate.
- Row/schema expectations are metadata cases (the physical frame remains rectangular), consistent with the oracle contract that removed rows/columns cannot have cell labels.

