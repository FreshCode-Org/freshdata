# Task 5 Report: Complete eight-domain TruthBench corpus

## Status

DONE. Logistics, government, education, and insurance now have deterministic 16-row gold fixtures. The registry is the stable alphabetical eight-domain order (`crm`, `education`, `finance`, `government`, `healthcare`, `insurance`, `logistics`, `retail`) with the temporary minimal registry removed.

## TDD evidence

### RED

After adding domain-content tests, the focused fixture suite failed with the expected missing-builder errors for all four new domains (`FixtureError: unknown fixture domain`). The failure run was:

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q --no-cov
```

### GREEN

The focused suite now passes:

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q --no-cov
```

Result: `47 passed`.

Adjacent fixture/model/schema verification:

```text
PYTHONPATH=src python -m pytest \
  tests/truthbench/test_fixtures.py \
  tests/truthbench/test_models_exact.py \
  tests/truthbench/test_schema.py -q --no-cov
```

Result: `112 passed`.

Static checks:

```text
python -m ruff check benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py
python -m ruff format --check benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py
mypy --ignore-missing-imports --explicit-package-bases \
  benchmarks/truthbench/fixtures tests/truthbench/test_fixtures.py
git diff --check
```

All passed.

## Coverage

- `logistics.py` covers valid UN/LOCODE-like references, kg/lb and C/F units, cross-timezone windows, 24:00 transport values, address PII, late tracking, and protected shipment IDs.
- `government.py` covers leading-zero IDs, Indian/international grouping, fiscal/calendar ambiguity, multilingual labels, restricted national IDs, mixed legacy encoding, and retention/repair policy contradiction.
- `education.py` covers student IDs, letter/percentage/GPA scales, school-year ambiguity, zero scores, enrollment ordering, guardian contacts, FERPA notes, and protected grade-policy conflict.
- `insurance.py` covers policy/claim IDs, premium/reserve currency mismatch, negative reserve review, incident/report ordering, state contradiction, claimant/medical PII, and protected policy numbers.
- All four builders emit complete physical-cell labels, all four dispositions, row duplicate/removal cases, five schema drift cases, fixed UTC/reference metadata, deterministic seed batches, and privacy-safe synthetic canaries.
- Corpus-level tests assert all required trap categories occur across the eight domains.

## Concerns

None. No FreshData runtime, LLM/provider, network, or external data dependency is used.

## Review follow-up: explicit contract coverage

### RED

The new required-family contract test intentionally used the repaired numeric output (`95`) as the adversarial frame value for education `edu-07`. The focused test failed because the actual frame value is the required raw value `"95%"` while the repair oracle separately stores `95.0` as its expected output.

```text
PYTHONPATH=src python -m pytest \
  tests/truthbench/test_fixtures.py::test_required_domain_families_match_actual_values_and_dispositions \
  -q --no-cov
```

Observed: one failure at `edu-07` (`'95%' != 95`).

### GREEN

The contract now asserts the raw value, family, disposition, and typed repair output. It also uses an explicit per-domain family mapping covering every required category (including logistics lb/F units, government IDs/grouping/language/fiscal/protected case, education scales/contacts/protected grade, and insurance IDs/grouped premium/PII/protected policy).

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_fixtures.py -q --no-cov
```

Result: `48 passed`.
