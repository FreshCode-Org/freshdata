# Task 6 Report: Privacy-safe values and exhaustive sink scanning

## Status

DONE. `SinkScanner` now scans normalized canary variants across nested TruthBench
sinks and emits only `Leak(canary_id, variant, path)` metadata. Redaction markers
contain run-scoped HMAC-SHA256 digests and never matched text.

## TDD evidence

### RED

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_privacy.py -q --no-cov
```

Observed collection failure before implementation:

```text
ModuleNotFoundError: No module named 'benchmarks.truthbench.privacy'
```

### GREEN

Focused privacy suite:

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_privacy.py -q --no-cov
```

Result: `18 passed`.

Adjacent TruthBench suites:

```text
PYTHONPATH=src python -m pytest \
  tests/truthbench/test_privacy.py \
  tests/truthbench/test_fixtures.py \
  tests/truthbench/test_models_exact.py \
  tests/truthbench/test_schema.py -q --no-cov
```

Result: `131 passed`.

Static checks:

```text
python -m ruff check benchmarks/truthbench/privacy.py \
  benchmarks/truthbench/__init__.py tests/truthbench/test_privacy.py
python -m ruff format --check benchmarks/truthbench/privacy.py \
  benchmarks/truthbench/__init__.py tests/truthbench/test_privacy.py
mypy --ignore-missing-imports --explicit-package-bases \
  benchmarks/truthbench/privacy.py tests/truthbench/test_privacy.py
git diff --check
```

All passed.

## Files

- `benchmarks/truthbench/privacy.py` — `Leak`, `PrivacySafeValue`, named
  normalizers, run-scoped HMAC scanner, recursive redaction, typed redaction
  handling, pandas/dataclass/bytes support, self-test, and named sink entry points.
- `benchmarks/truthbench/__init__.py` — exports privacy scanner primitives.
- `tests/truthbench/test_privacy.py` — mutation matrix for every required
  normalized form, nested sink coverage, redaction/self-test behavior, and typed
  redaction digest safety.

## Self-review

- Leak objects contain only identifiers, transform labels, and JSONPath-like
  locations; `repr(leaks)` cannot repeat canary text.
- Literal, case-folded, whitespace-stripped, punctuation-stripped, digit-only,
  URL-decoded, HTML-unescaped, UTF-8/hex/escape bytes, NFKC/NFC/NFD,
  zero-width-removed, and JSON-escaped forms are covered.
- Mapping, sequence, dataclass, pandas DataFrame/Series/Index, exception,
  report, plan, generated-code, stream, markup, JSON, and failure-artifact sinks
  are traversed. Exact redacted `TypedValue` payloads are treated as safe and
  their HMAC digests are not re-scanned as plaintext.
- Redaction is recursive, emits `[REDACTED:<digest>]`, and `self_test` raises on
  an unredacted result while accepting the scanner's own redacted output.

## Concerns

- The scanner intentionally treats digit-only normalized forms conservatively;
  very short numeric canaries can match unrelated text. Fixtures use synthetic
  identifiers and email/phone canaries, so this does not affect the bundled
  corpus.
- Redacting a pandas object may change a sensitive column to object/string dtype,
  which is preferable to retaining a raw value in an audit sink.

## Follow-up RED/GREEN evidence

A follow-up regression test used the exact sensitive `TypedValue` redaction shape
with a one-digit canary. Before the redacted-payload guard, the digest's hex text
was incorrectly reported as a digit-only leak. After adding the guard:

```text
PYTHONPATH=src python -m pytest \
  tests/truthbench/test_privacy.py::test_scanner_accepts_exact_typed_redaction_without_scanning_digest \
  -q --no-cov
```

Result: `1 passed`.

## Review follow-up: marker, key, and label hardening

### RED

The review regression matrix was run before the hardening changes:

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_privacy.py -q --no-cov
```

Result: `4 failed, 17 passed` for forged redaction markers, sensitive mapping
keys, pandas labels, and the digest compatibility alias.

### GREEN

After validating markers against the scanner's own 64-hex HMAC digest set,
redacting mapping keys and pandas column/index/name labels, scanning arbitrary
key stringifications, and adding `digest` as an alias:

```text
PYTHONPATH=src python -m pytest tests/truthbench/test_privacy.py -q --no-cov
```

Result: `23 passed`.

Adjacent TruthBench verification:

```text
PYTHONPATH=src python -m pytest \
  tests/truthbench/test_privacy.py \
  tests/truthbench/test_fixtures.py \
  tests/truthbench/test_models_exact.py \
  tests/truthbench/test_schema.py -q --no-cov
```

Result: `136 passed`.

Static checks were rerun after the follow-up and remained clean:

```text
python -m ruff check benchmarks/truthbench/privacy.py \
  tests/truthbench/test_privacy.py
python -m ruff format --check benchmarks/truthbench/privacy.py \
  tests/truthbench/test_privacy.py
mypy --ignore-missing-imports --explicit-package-bases \
  benchmarks/truthbench/privacy.py tests/truthbench/test_privacy.py
git diff --check
```
