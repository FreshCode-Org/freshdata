# Threat model

What FreshData defends against, what it deliberately does not, and where the
residual risks live. [Limitations](limitations.md) is capability honesty;
this page is security and privacy honesty. Every claim here is backed by a
test, a measurement, or a pointer to the line of code that enforces it.

## Trust boundaries

### 1. Input data → cleaning engine

Input is untrusted. Cleaning never executes cell contents: there is no
`eval`, no `exec`, and no pickle loading anywhere in `src/freshdata`,
including the format parsers (HL7v2 / GPX / SDMX / EDIFACT / FHIR). HL7v2
and EDIFACT are hand-rolled text parsers and FHIR uses `json`. GPX and
SDMX use the standard library's `xml.etree.ElementTree` (expat), behind a
guard in `Parser.open_safe_xml_binary`: input is capped at 10 MB, the
document encoding is detected (byte-order mark, UTF-16/UTF-32 prefix, or XML
declaration; EBCDIC is refused), and any `DOCTYPE` or entity declaration is
rejected, both by a marker scan of the decoded text and by an expat pass that
sees the document exactly as ElementTree will. No DTD reaches ElementTree, so
there is no entity expansion ("billion laughs") in any encoding. Tabular
inputs are read only through pandas / pyarrow / DuckDB readers.

### 2. Cleaned data → CSV exports (formula injection)

Cells starting with `= + - @ <tab> <cr>` — including behind leading
whitespace — execute as formulas when a CSV is opened in Excel / Google
Sheets / LibreOffice (OWASP CSV injection). Every first-party CSV surface is
now safe by default; byte-exact fidelity is the explicit opt-out:

| Surface | Default | Opt-out |
|---|---|---|
| `export_review_queue` (csv) | sanitize **on** | `sanitize_formulas=False` |
| `fd.clean_csv(output_path=...)` | sanitize **on** | `sanitize_formulas=False` |
| `fd.clean_excel(output_path=...)` (xlsx) | sanitize **on** | `sanitize_formulas=False` |
| `freshdata clean` / `apply-plan` CLI csv output | sanitize **on** | `--no-sanitize-formulas` |
| streaming CLI (incl. quarantine export) | sanitize **on** | `--no-sanitize-formulas` |
| HTML-report ledger CSV download | sanitize **on** | none (spreadsheet-bound artifact) |

Sanitizing covers every place input text reaches the file: cells, column
labels at every level of a multi-row header (`read_csv_kwargs={"header":
[0, 1]}`), index labels at every level, and column/index names (written with
`index=True`). Header aliases a caller passes to the writer
(`to_csv_kwargs={"header": [...]}`) are caller-supplied and written as given.

JSONL and Parquet are never altered. **Residual risk:** a consumer that
opted out and opens the CSV in a spreadsheet re-accepts the injection risk.

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

Hash masking is HMAC-SHA256 keyed by the rule's `salt`. When `salt` is
empty, `MaskingRule` generates a random salt per rule, so default hash
tokens are not stable across runs and cannot be recomputed from the source.
Set your own `salt` when you need stable, joinable tokens, and keep it
secret: anyone holding the salt and the tokens can confirm guesses of
low-cardinality values.

`tokenize`, `surrogate` and `fpe` masking rules and the policy
`pseudonymize` action (the GDPR pack's default, also used by the HIPAA
date-of-birth and FERPA grade rules) are keyed by `key` / `key_env`. They
never fall back to a constant from the source. Without a key, each
`anonymize` / `apply_privacy_policy` call uses a random key and emits
`EphemeralKeyWarning`, and `report.metadata["ephemeral_key_rules"]` names
the rules. Output is then consistent within one call but not across calls.
Pass a secret key for stable, joinable pseudonyms; anyone holding that key
and the output can recompute the pseudonym of a guessed value. Before 2.1.0
these keyless paths used public constants, so keyless output from earlier
releases can be reversed by enumerating candidate values: re-pseudonymise
it with a secret key (see [Compliance](compliance.md#pseudonymisation-keys)).

The copilot's internal masking uses the default
deterministic path on purpose: a per-run random salt would break the
documented reproducibility of `model_context` and its audit fingerprint.
This trade-off is tracked as a roadmap item, not silently changed.

Report stand-ins for declared `sensitive_columns` are a separate case. They
are the `[SENSITIVE:xxxxxxxx]` tokens in `CleanReport` warnings, coerced
cells, semantic action text and metadata, and `validate_fields`
`normalized_cells`. Each one is a truncated HMAC-SHA256 under a random key
made once per process, never a constant. Within one run the same value
gives the same token, so records can be matched up. A guess list of SSNs,
phone numbers or dates cannot be hashed to find a match, and tokens from
different runs are not joinable.

### 7. Local temporary/spill files

When a DuckDB run exceeds `memory_limit_gb`, DuckDB writes intermediate
relation data (rows of the dataset being cleaned) to disk with the process
umask. Each run therefore spills into its own `tempfile.mkdtemp` directory
(mode 0700) that no other local account can list or read, and concurrent runs
never share file names. The directory lives under `EngineConfig.temp_directory`
if set, else `$FRESHDATA_SPILL_DIR`, else the per-user cache directory
(system temp only when that is not writable), and is removed when the run's
connection closes, or when a returned `output_format="duckdb"` relation is
released (`execution/_spill.py`). The base directory must be owned by the
current user and not group/other-writable, or be sticky and owned by the user
or root; otherwise the run raises `PermissionError`. **Residual risk:** a
process killed mid-run (`SIGKILL`, power loss) leaves its private run
directory behind until it is deleted; root and the same user can always read
it.

### 8. Token vault files

`JsonTokenVault` and `SqliteTokenVault` store the plaintext token-to-value
mapping; so do `MaskingRule(token_vault_path=...)` and policy vaults with
`vault_backend: json` or `sqlite`. Anyone who can read a vault file, or a
backup of it, can reverse every token in it.

On POSIX, FreshData creates vault files with mode **0600** (owner read/write
only) in the same `os.open` call that creates them, so a new vault is never
readable by other users, whatever the umask. SQLite gives its `-journal`,
`-wal` and `-shm` files the database file's mode. A missing parent directory
is created with mode 0700; only the last path component gets that mode, and
intermediate directories follow the umask.

An existing vault file keeps its mode. If it is group- or other-accessible,
FreshData emits a `UserWarning` and still uses it; it never changes the mode
of a file it did not create. On Windows, access to vault files follows the
ACLs of the directory. Keep vaults out of shared directories either way.

### 9. Persisted baselines and profiles

Drift baselines (`fd.build_baseline` / `save_baseline`) are meant to be
committed and shared, so they must not carry the data they summarise. With the
default `include_samples=False` a baseline never stores raw sample values, and
category labels are protected one of two ways:

- **No key (default): label-free.** Each categorical column stores only its
  frequency profile in descending order (`r:0000`, `r:0001`, …). Categorical
  PSI still catches shape and cardinality drift, but it cannot see two
  categories swapping shares.
- **`label_key=` or `FRESHDATA_BASELINE_KEY`: keyed.** Labels are
  HMAC-SHA256 pseudonyms and only a short key identifier is stored. Compare
  with the same key; a missing or different key skips categorical PSI with a
  `drift.categorical_drift_skipped` warning. Anyone holding the key can confirm
  guessed labels, so keep it out of the repository that holds the baseline.

Baselines built inside one call (`compare_to_baseline(df, other_df)`, the
inline baseline in `clean_enterprise`) use a random key that is never stored.
**Residual risk:** exact category shares are still visible, and
`freshdata-baseline-v1` files (unkeyed SHA-1 labels, reversible by hashing a
guess list) still load with a warning; rebuild and delete them.

Learned `.fdprofile` archives are meant to be saved, diffed and shared.
Under the default `privacy="mask"`, literal value-map entries and examples on
sensitive columns are stored as HMAC tokens. A column is sensitive when its
name matches a hint or when `detect_pii` finds *any* PII type in it — payment
cards, IBANs, IP addresses and health identifiers included; unmapped types
fail closed to free text. `freshdata profile audit` flags checksum-valid card
numbers and IBANs left raw by older versions. **Residual risk:** PII that
neither a name hint nor the regex scanner recognises (for example a bare
account number in a column called `ref`) is stored raw; pass
`protected`/`context` policies or drop such columns before `fd.learn`.

## Non-goals

- **Not a sandbox.** FreshData reads tabular files; hostile *file formats*
  are the parser libraries' concern (pandas / pyarrow / duckdb versions are
  therefore floor-pinned).
- **Not DLP.** PII detection is best-effort convenience. The compliance
  packs state their own scope and do not turn detection into a guarantee.
- **No network in any cleaning path.** The only network call in the package
  is the explicit `fd.models.pull(...)`; offline behavior is CI-enforced
  (`-m "not online"` is the default gate).
