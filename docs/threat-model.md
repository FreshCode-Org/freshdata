# Threat model

What FreshData defends against, what it deliberately does not, and where the
residual risks live. [Limitations](limitations.md) is capability honesty;
this page is security and privacy honesty. Every claim here is backed by a
test, a measurement, or a pointer to the line of code that enforces it.

## Trust boundaries

### 1. Input data → cleaning engine

Input is untrusted. Cleaning never executes cell contents: there is no
`eval`, no `exec`, and no pickle loading anywhere in `src/freshdata`,
including the format parsers (HL7v2 / GPX / SDMX / EDIFACT / FHIR), which
are hand-rolled text parsers. Inputs are read only through
pandas / pyarrow / DuckDB readers.

### 2. Cleaned data → CSV exports (formula injection)

Cells starting with `= + - @ <tab> <cr>` execute as formulas when a CSV is
opened in Excel / Google Sheets / LibreOffice (OWASP CSV injection).
FreshData's stance follows the artifact's purpose:

| Surface | Default | Why |
|---|---|---|
| `export_review_queue` (csv) | sanitize **on** (`sanitize_formulas=False` to opt out) | built to be opened by humans in spreadsheets |
| `fd.clean_csv(output_path=...)` | off, opt-in `sanitize_formulas=True` | pipeline artifact; byte-exact fidelity by default |
| streaming CLI | off, opt-in `--sanitize-formulas` (also covers the quarantine export) | same |

JSONL and Parquet are never altered. **Residual risk:** a pipeline CSV
opened directly in a spreadsheet keeps the risk unless the flag was passed.

### 3. DataFrame → AI Copilot `model_context`

`report.model_context` is the only payload a `provider` hook ever sees, and
it is SHA-256 fingerprinted in `report.audit` so you can prove after the
fact what was shared. Per privacy mode:

- **`mask_pii_before_reasoning` (default)** — every string-like column
  (object / string / categorical) in the sample rows is hash-masked: declared
  `must_mask` columns, regex-detected PII columns, and everything else
  string-like. This is deliberate defense-in-depth: regex detection cannot
  see names, addresses, or free text, so no string value is trusted to be
  safe. `allow_unmasked_columns` is an explicit per-column opt-out that
  never exempts a declared or detected PII column and rejects unknown
  names. Detected-problem details entering `model_context` are value-free
  in **every** mode (`category_noise` spelling previews stay local).
- **`schema_only`** — no cell values at all.

These guarantees are enforced by adversarial regression tests registered in
the `CLAIM_REGISTRY` (`tests/test_experimental_ai_copilot.py`), which CI
re-verifies against the README wording.

**Residual risk (by design, documented):** numeric values pass through
unmasked. Numeric quasi-identifiers — an exact salary plus age plus a
postcode-like code — can re-identify a person. Drop such columns first or
use `schema_only`.

### 4. Provider hook failure

Provider exceptions are caught, recorded as `report.audit["provider_error"]`,
and the deterministic report survives. Stated plainly: the prompt has
already been sent when a provider fails — a failing provider does not
un-send data. The failure mode is fail-closed for the *report*, not a
retroactive privacy guarantee.

### 5. PII detection is regex-only

The dependency-free detector covers EMAIL / PHONE / SSN / credit card / IP.
It does **not** detect names, addresses, or free-text PII (install the
`privacy` extra for NER via presidio). This is the #1 practitioner gotcha:
if you call `anonymize()` with only auto-detected columns, undetected PII
passes through. The copilot does not inherit this gap because it masks all
string-like columns regardless of detection (see boundary 3).

### 6. Masking tokens

Hash masking is HMAC-SHA256 with a configurable salt. **The default salt is
a public constant** in the source, so tokens are stable across runs and
joinable — and an attacker holding tokens can confirm guesses of
low-cardinality values. Supply your own `salt` on `MaskingRule` when
unlinkability matters. The copilot's internal masking uses the default
deterministic path on purpose: a per-run random salt would break the
documented reproducibility of `model_context` and its audit fingerprint.
This trade-off is tracked as a roadmap item, not silently changed.

## Non-goals

- **Not a sandbox.** FreshData reads tabular files; hostile *file formats*
  are the parser libraries' concern (pandas / pyarrow / duckdb versions are
  therefore floor-pinned).
- **Not DLP.** PII detection is best-effort convenience. The compliance
  packs state their own scope and do not turn detection into a guarantee.
- **No network in any cleaning path.** The only network call in the package
  is the explicit `fd.models.pull(...)`; offline behavior is CI-enforced
  (`-m "not online"` is the default gate).
