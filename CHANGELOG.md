# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Added a runnable PyJanitor interoperability example that demonstrates both
  tool orderings while keeping PyJanitor optional.
- A dependency-optional Great Expectations recipe demonstrating the
  repair-then-validate workflow with an in-memory checkpoint.

## [2.0.0] - 2026-07-20

Remediation of the July 2026 v1.2.0 production-readiness audit: the unsafe
defaults it confirmed are now safe-by-default, with every old behavior still
available as an explicit opt-in. These default changes are breaking, hence
the major version. v1.2.0 was tagged but never published, so PyPI users
upgrade directly from 1.1.1 and should read the [1.2.0] section below as
part of this release.

### Changed — safety defaults (breaking)
- **`drop_duplicates` now defaults to `False` under every strategy.** Exact
  duplicate rows are detected and reported but never removed until you opt in
  with `drop_duplicates=True`. A duplicate ratio above `duplicate_threshold`
  still raises the strong report warning; the new
  `duplicate_ratio_action="error"` escalates it to
  `DuplicateRatioError` for pipelines that must stop on a suspicious join.
- **`outlier_action="auto"` now flags under every strategy, including
  `"aggressive"`, and never rewrites values.** Winsorizing requires an
  explicit `outlier_action="cap"`; explicitly-requested capping is now
  skew-aware (log-space Tukey fences for strongly skewed positive data) so
  legitimate heavy tails are not flattened.
- **`dayfirst="auto"` never infers a column-wide day/month order.** Values
  whose day-first and month-first readings are both valid are quarantined
  into `report.coerced_cells` (originals preserved, audit action recorded)
  unless `dayfirst=True/False` is set explicitly; a single unambiguous
  sibling value no longer flips the interpretation of a whole column.
- **Column-name outlier heuristics are opt-in.** The `_DOMAIN_SENSITIVE`
  name match (fraud/amount/risk/…) no longer changes outlier decisions by
  default; identically-distributed data gets identical treatment regardless
  of column name. Opt back in with `domain_sensitive_names=True`.
- **`fd.anonymize()` fails closed.** Called with no `rules` and no
  `detection_config` it now raises `ValueError` instead of returning the
  data unchanged with a warning.
- **CSV outputs neutralize spreadsheet formula injection by default**
  (`fd.clean_csv`, `freshdata clean` / `apply-plan` CLI CSV output, the
  streaming CLI including its quarantine export, and the HTML-report ledger
  download). Cells and labels starting with `= + - @ <tab> <cr>` — including
  behind leading whitespace — are prefixed with `'`. Use
  `sanitize_formulas=False` / `--no-sanitize-formulas` where byte-exact
  round-trips matter.
- **PyPI Development Status classifier downgraded to `4 - Beta`** until the
  project's own absolute release gates (CleanBench T5 runtime gate included)
  hold on release infrastructure.

### Fixed
- PII detection no longer misreports dates, UUID fragments, licence IDs,
  Aadhaar-style 4-4-4 digit groups, semver strings, or ordinary numbers as
  `PHONE`, and no longer misreports IBANs as `CREDIT_CARD`. Regex matches
  for PHONE / CREDIT_CARD / IBAN must now pass post-match validators:
  structural plausibility and boundary checks for phones, Luhn (13–19
  digits) for cards, ISO 13616 mod-97 for IBANs (spaced and compact).
  Checksum-verified findings report `source="checksum"` with score ≥ 0.95.
- Restored the nightly perf-regression workflow's CleanBench T5 gate, which
  a temporary profiling probe (PRs #152/#153) had left neutered with
  `|| true` — the gate and its failure-alert issue automation are enforcing
  again.
- Refreshed the committed TruthBench `baseline.json`, which still recorded
  the four pre-release-audit KNOWN-RED gates (`cleaning:raw_pii_leakage`,
  `cleaning:review_routing`, `cleaning:exact_repair`,
  `generated_code_sandbox`) even though the release-audit fixes shipped in
  v1.2.0 made them pass; a regression on any of them now fails the PR
  ratchet.

## [1.2.0] - 2026-07-18

> **Note:** v1.2.0 was tagged (`v1.2.0`) but never published to PyPI or
> GitHub Releases — the release was stopped on a benchmark runtime gate.
> Everything below first shipped to users in 2.0.0.

### Added
- **TruthBench release runner** (`benchmarks/truthbench/`): the semantic
  red-team foundation gained its missing production pieces — an end-to-end
  runner, normalized decision hashing with fail-closed repeat verification
  (`determinism.py`), full generated-code verification (parse → strict AST
  allowlist → compile → isolated `python -I` execution with module poisoning,
  timeout, input contract, protected-cell and PII-canary checks), a
  deterministic failure minimizer, atomic schema-validated result artifacts,
  and a CLI. `make truthbench-release` (equivalently
  `PYTHONPATH=src python -m benchmarks.truthbench run --profile release
  --backends pandas,polars,duckdb --require-backends --repeats 2 --check`)
  now gates PR CI and the release workflow; every decision/sink surface maps
  to a concrete behavioral adapter (a contract test rejects placeholders).
- Adversarial regression suites for the twelve release-risk hypotheses
  (`tests/test_release_hypotheses.py`) and the TruthBench components.
- **Validation Gauntlet** (`benchmarks/gauntlet/`, `docs/validation-gauntlet.md`):
  a gold-labelled disposition benchmark for the validation, domain and
  text-cleaning surfaces. Five deterministic fixtures (finance, healthcare,
  CRM, e-commerce, adversarial text) label every injected defect with the
  disposition FreshData should choose (preserve / repair / flag / review) and
  the harness scores detection P/R/F1, repair accuracy, review routing,
  preservation, corruption, escapes, false positives, audit completeness,
  determinism, trust monotonicity and runtime/memory. Runs on every PR
  (`gauntlet.yml`) with absolute gates plus no-regression checks against the
  stored `baseline.json`.
- `CleanReport.coerced_cells`: per-cell record (`{column: {row: original}}`)
  of values that `fix_dtypes` nulled because they did not parse as the
  column's inferred type — the recovery source for quarantined cells, also
  included in `report.to_dict()`.
- Date-field range validation in `fd.validate_fields`: `FieldSpec.min_value`
  / `max_value` now accept a date string or timestamp for `date`/`datetime`
  fields, so a future date of birth or an 1875 admission date is flagged as a
  `domain_mismatch` (gauntlet finding).
- Case-variant vocabulary suggestions in `fd.validate_fields`: a value that
  matches an `allowed_values` entry except for case (`ACTIVE` vs `active`) is
  no longer silently accepted — it gets a warning-severity issue with the
  canonical form as `suggestion` and action `accept_with_warning` (gauntlet
  finding).
- `docs/trust-claims.md` (claim-to-evidence map for every trust-relevant
  README/docs claim, superset of the machine-enforced `CLAIM_REGISTRY`) and
  `docs/threat-model.md` (trust boundaries, per-privacy-mode guarantees,
  ranked residual risks), both linked in the docs nav.
- `benchmarks/bench_outofcore.py`: subprocess-isolated peak-RSS evidence for
  the four engine/output-format combinations on a generated parquet fixture
  (per-scenario `ru_maxrss`, wall time, and the `materialized` flag).
- **AI Copilot (experimental)** — `freshdata.experimental.ai_copilot.analyze_dataset`:
  deterministic, fully offline dataset analysis that returns a ranked problem
  list (PII, policy violations, duplicates, missing values, mixed date
  formats, near-duplicate category spellings), a PII warning, an ordered
  explainable cleaning plan, and copy-ready freshdata code generated for the
  analyzed dataset. Privacy-first: raw string values never enter the report's
  `model_context` (every string-like sample column is hash-masked, numeric
  values pass through as-is, or samples are omitted entirely with
  `privacy="schema_only"`); the payload is SHA-256 fingerprinted in the
  audit. An optional `provider` hook (plain `Callable[[str], str]`) allows
  plugging in an LLM later — no built-in provider ships, no API key is
  needed, and provider failures never break the deterministic report.
- **Flagship demo**: `examples/freshdata_ai_copilot_demo.py` plus the bundled
  `examples/data/messy_customers.csv` — the full messy-to-audit-ready story
  (analyze → mask → clean under a compiled policy → merge category variants →
  re-score trust), and a new docs guide (`docs/ai-copilot.md`).
- `CITATION.cff` so the project can be cited from GitHub's "Cite this
  repository" button, and a documentation issue template alongside the
  existing bug/feature templates.

### Changed
- Lint now covers the whole repository (`ruff check .` in CI, closing #54):
  benchmark and notebook lint debt fixed, dead code removed
  (`harness_metrics` unused gold-labels block), and the ASV-managed
  `freshdata-benchmarks/` sub-project excluded as tool-generated. No
  runtime behavior changes.
- All CI workflows now declare least-privilege `GITHUB_TOKEN` permissions
  (read-only by default; the nightly-alert and coverage-badge jobs keep their
  scoped write grants).
- Contributor docs (`CONTRIBUTING.md`, `README.md`, `QUALITY_OPS.md`) now
  quote the exact commands CI runs; the pre-commit config no longer ships a
  `ruff-format` hook the codebase and CI never enforced.

### Removed
- Dead packaging/CI config: `MANIFEST.in` (ignored by the hatchling build
  backend — the sdist is shaped by `pyproject.toml`) and
  `freshdata-benchmarks/.github/workflows/` (nested workflow directories are
  never executed by GitHub Actions).
- Committed AI-assistant working artifacts (`.superpowers/`, now
  git-ignored) and internal planning notes that were being published to the
  documentation site (`docs/superpowers/`).

### Fixed
- **Default-path slowdown from the scientific-notation segfault guard**
  (release blocker, nightly issue #147): the guard that masks huge-exponent
  tokens (`"1e999"`) before `pd.to_numeric` — protection against a pandas
  2.3.x segfault — screened text columns cell by cell through a Python
  predicate, roughly doubling `fix_dtypes` time on 50k-row frames in CI.
  Each column is now screened with a single C-level joined-blob regex scan
  and the per-cell predicate runs only on columns that screen positive.
  Masking semantics are unchanged; the CleanBench T5 runtime gate is back
  within its ±20 % baseline envelope.
- `dir(freshdata)` no longer lists `Action` twice: the privacy-policy engine's
  `Action` enum was listed in the lazy enterprise exports but was unreachable
  there — `fd.Action` is (and remains) the audit action from
  `freshdata.report`. Use `freshdata.enterprise.Action` for the privacy enum.
- **Sensitive-column masking across all report surfaces**
  (`fd.clean(sensitive_columns=...)`, `fd.validate_fields(sensitive_columns=...)`,
  `analyze_dataset(sensitive_columns=...)`): values from declared-sensitive
  columns never appear verbatim in report warnings, coerced-cell payloads,
  action rationales/metadata/evidence, `normalized_cells`, or
  Copilot-recommended pipelines (which now always mask declared columns
  before printing report summaries). A deterministic `[SENSITIVE:xxxxxxxx]`
  digest token keeps records correlatable without disclosure.
- **Ambiguous and partial dates are quarantined, never interpreted**: a
  short-form date whose day/month order cannot be resolved (no explicit
  `dayfirst`, no disambiguating sibling) and partial ISO dates (`"2025-01"`)
  now coerce to missing with originals preserved in `coerced_cells` for
  review instead of being silently resolved month-first / to a fabricated
  day; time-range strings (`"09:00-17:00"`) no longer parse to bogus
  offset-bearing timestamps.
- **Corroboration-gated semantic mutations**: unit strips auto-apply only
  with a declared column unit (inferred units demote to suggestions;
  unit-mismatched values become high-risk review items), and a new
  dataset-level `semantic_context["currencies"]` declaration routes
  out-of-policy currency values to review instead of silently dropping the
  denomination.
- **CSV formula injection** in `write_exception_table`: observed values such
  as `=HYPERLINK(...)` were written verbatim to the exception-table CSV and
  would execute when opened in a spreadsheet. The CSV path now routes through
  the same `sanitize_csv_formulas` guard every other spreadsheet export uses.
- **Trust-score monotonicity**: `compute_trust_score` rated corrupted frames
  *higher* than pristine ones because constant columns counted as structural
  inconsistency (corrupting one made it vary, clearing the flag). Constant
  columns are now surfaced as per-column issues instead of lowering the
  consistency dimension.
- **Semantic overconfidence**: an isolated unit value (`"10 lb"` among plain
  numbers) is no longer auto-stripped at 0.97 confidence — inferred unit
  consistency now requires majority support and demotes to a suggestion
  otherwise; already-canonical ISO dates are no longer rewritten to
  timestamps when unparseable values keep the column as text.
- **Calibration (nightly issue #139)**: the CleanBench full-suite ECE gate
  failed at 0.0384 > 0.03 from systematic *under*confidence of the measured
  deterministic canonicalization families. `calib-default-2` maps their raw
  0.96–0.97 scores to measured rule-of-three lower bounds (email_format
  148/148, phone_format 168/168, reference_value 128/128 across seeds 0–9)
  while staying identity below the 0.95 auto threshold, so no
  apply/suggest/review decision changes. ECE is now 0.0217.
- **Default-path performance regression (nightly issues #139/#140)**: the
  pandas-segfault exponent guard and the relative-date guard in `fix_dtypes`
  scanned whole columns per value; both are now vectorized (candidate
  prefilter / unique-first) restoring the T5 runtime gate (slowdown 0.27 →
  0.03 vs the v1.0 baseline) with unchanged semantics.
- **Nightly online/large lane (issue #138)**: the lane inherited the
  repo-wide `--cov-fail-under=93` while deliberately selecting ~21 tests, so
  it could never pass; it now runs with `--no-cov` (coverage stays enforced
  on the full fast lane).
- **Unparseable values are quarantined, never fabricated** (gauntlet finding,
  the `'apple'`-in-a-price-column case): when `fix_dtypes` converts a
  mostly-numeric (or datetime) text column, cells that fail to parse used to
  become `NaN` and then be silently imputed by the auto engine — turning
  junk into a fabricated median. They now stay missing, are excluded from
  auto-imputation, keep their originals in `report.coerced_cells`, and the
  decision is a `human_review` action in the audit trail. Genuine missing
  values (true `NaN`, sentinels like `"N/A"`) keep the documented
  auto-impute behaviour, and an explicit `impute=` request still fills
  everything.
- Formatted-number stragglers (`"$1,234.56"`, `"1,200,500.00"`) in a
  mostly-plain numeric column are now parsed by the existing locale-aware
  rescue instead of being coerced to missing — the rescue previously only
  engaged when the plain parse failed the threshold entirely (gauntlet
  finding).
- `fd.validate_fields` consensus inference now honours the same
  contamination boundary as the `fix_dtypes` warning that points users at it
  (dominant share ≥ 60% with at most a handful of stragglers). Previously
  the warning fired from a 60% parse share but the consensus gate required
  80%, so the exact frame the warning named sailed through
  `validate_fields` silently (gauntlet finding).
- Explicitly allowed values are no longer swallowed by null-marker
  heuristics in `fd.validate_fields`: with
  `FieldSpec(allowed_values={"US", "DE", "NA"})`, `"NA"` is Namibia, not a
  missing value (gauntlet finding).
- `clean_text` / `validate_fields` text normalization no longer rewrites
  typography in content-bearing fields: for `free_text`, `text` and entity
  name types, the punctuation→ASCII mapping (curly quotes, em-dashes, prime
  marks — `12″` became `12"`) is withheld, matching the field-aware safety
  contract. Untyped columns keep the existing behaviour (gauntlet finding).

- `anonymize()` called with no `rules` and no `detection_config` now emits
  a `UserWarning` instead of silently returning the data unchanged — a
  privacy call that does nothing must say so. Behavior is otherwise
  unchanged; pass an empty rule set intentionally by suppressing the
  warning (found by the installed-wheel matrix audit).
- `pip install "freshdata-cleaner[polars]"` now actually enables the
  advertised polars round-trip: the extra was missing `pyarrow`, which
  `fd.clean(polars_df)` needs for the polars→pandas interchange, so the
  natural install crashed with polars' internal ModuleNotFoundError. The
  extra now ships pyarrow, and the adapter raises an actionable message
  naming the fix when pyarrow is absent (found by the installed-wheel
  matrix audit).
- `explain_clean` cell-change reporting: when cleaning removed rows (for
  example duplicate removal), every shared column previously reported the
  whole surviving row count as "cells changed". Frames are now aligned on
  their shared index labels and only genuinely differing cells are counted;
  cells missing on both sides are unchanged, value↔missing transitions
  count, and a dtype conversion alone no longer marks untouched values as
  changed. The elementwise fallback also no longer uses a Python-3.10-only
  `zip(strict=...)` argument, which crashed on Python 3.9 when reached (#30).
- `memory_bytes` sampled estimation (frames above 200k rows) no longer counts
  the index payload once per string-like column; a string-heavy index is now
  measured once, matching the exact path used for smaller frames (#35).
- Integer finalization now checks the exact int64 range in integer space
  instead of a float magnitude threshold: `-2**63` and `2**63 - 1024` (the
  largest float64 below `2**63`) convert to int64/Int64 exactly instead of
  being demoted to float64, and values at or above `2**63` can never be
  admitted by float rounding (#34).
- **AI Copilot privacy hardening**: sample rows in `model_context` now
  hash-mask *every* string-like column, not only declared/regex-detected PII
  columns — names, addresses, free text, and obfuscated identifiers in
  undeclared columns no longer leave the machine raw. A new explicit
  `allow_unmasked_columns` opt-out exists but never exempts declared or
  detected PII. `category_noise` problem details are stripped of raw value
  previews before entering `model_context` in **all** privacy modes
  (including `schema_only`); the local `report.problems` keeps the rich
  previews.
- Out-of-core docs now match measured behavior: keeping a native handle
  requires `fix_dtypes=False` **in addition to** `strategy="conservative"`
  (dtype fixing runs sampled pandas heuristics and forces the recorded
  fallback), and `output_format="polars-lazy"` defers only the *final*
  materialization — pipeline stages currently collect intermediates
  eagerly, so peak memory during cleaning matches eager output. The DuckDB
  handle path is the measured lower-peak-memory route (#52, #53).
- **CSV formula-injection protection** (OWASP): `export_review_queue` now
  neutralizes spreadsheet formula payloads in CSV exports **by default**
  (string cells and column labels starting with `= + - @ <tab> <cr>` get a
  leading `'`; opt out with `sanitize_formulas=False`) — review queues are
  built to be opened by humans in spreadsheets. `fd.clean_csv` and the
  streaming CLI keep byte-exact output by default and gain an explicit
  opt-in (`sanitize_formulas=True` / `--sanitize-formulas`) covering the
  cleaned output and the quarantine export. JSONL/Parquet are never altered.
- `SECURITY.md` supported-versions table updated to the current 1.1.x line.
- Source distribution now contains exactly the documented file set: the
  hatchling `include` patterns are anchored to the repo root, so unanchored
  names no longer pull in stray matches at any depth (`docs/examples/*.html`,
  nested `README.md`s).
- `freshdata-benchmarks/README.md` no longer claims CI execution or
  published results the repository never produced; it now documents the ASV
  suite as a locally-run comparative benchmark, separate from the CleanBench
  CI workflow.

## [1.1.1] - 2026-07-06

### Fixed
- **README rendering on PyPI**: the logo and several links (`LICENSE`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `examples/*`) used
  paths relative to the repository, which resolve on GitHub but not on the
  PyPI project page (rendered with no repository context). All now point to
  absolute `github.com`/`raw.githubusercontent.com` URLs. No code changes.

## [1.1.0] - 2026-07-06

### Added
- **Interactive output layer** (`freshdata.render`, lazy-imported): `to_html()` /
  `_repr_html_()` / `.show()` on `CleanReport` (collapsible action timeline +
  filterable audit ledger), `Profile` (inline quality cockpit), `CleanPlan`
  (decision cards / strategy diff grid), `ExplainReport` (before/after diff
  explorer), and the `compare_plans` / `compare_clean` / `infer_roles` frames
  (via a transparent `ReportFrame` DataFrame subclass). Self-contained HTML needs
  **no** optional deps; new `[viz]` / `[notebook]` extras (itables, plotly,
  great-tables, anywidget) only *upgrade* the output. `Action` gains
  `status` / `reversible` / `memory_influenced` / `human_review` metadata.
- **Cleaning memory**: `fd.learn_cleaning_memory` / `fd.load_cleaning_memory` /
  `CleaningMemory` (JSON + server-free SQLite storage, `to_dict` / `to_json` /
  `diff` / `summary`) and `fd.clean(df, memory=...)` replay — applies accepted
  decisions when the dataset signature matches and **blocks + explains** unsafe
  replay when the data drifts too far.
- **Baseline drift convenience**: `fd.compare_to_baseline` now accepts a raw
  DataFrame baseline plus `key=` / `event_time=` for key-level change counts;
  `DriftReport` gains `what_likely_matters()` and an interactive view.
- **Quality-debt ledger**: `fd.evaluate_quality_debt` scores nine debt dimensions,
  persists history to SQLite, and escalates warn→fail on repeated/worsening issues.
- **Dirty-join assistant**: `fd.suggest_join_keys` proposes exact + fuzzy join
  keys with confidence, blocking, per-field explanations, and an ambiguous/review
  section — never auto-joining low-confidence matches.
- **Text/encoding lint**: `fd.lint_text_encoding` detects mixed scripts, mojibake,
  NFC/NFD inconsistency, RTL/LTR risk, locale-ambiguous dates/numbers, and
  replacement/control characters (diagnostic-only, with safe-repair flags).
- **Stakeholder summaries**: `fd.stakeholder_summary` exports business-language
  Markdown / HTML.
- **Honest out-of-core handles**: new `output_format="duckdb"` /
  `"polars-lazy"` return an un-materialized DuckDB relation / Polars `LazyFrame`;
  `CleanReport.materialized` flags it. Streaming Polars dedup is now streaming-safe
  (no forced `maintain_order`) and discloses the order trade-off.
- **Benchmarks**: `benchmarks/bench_report.py` (100MB CSV ingest, 1M-row profile,
  10M-row null-fill, import-time, memory; balanced vs aggressive) with reproducible
  commands and honest "not yet measured" placeholders in the docs.
- New **CDC / event-time quality gate** `fd.cdc_profile(df, event_time=..., key=...)`
  (module `freshdata.cdc`, also exporting `CDCReport` / `CDCDefect`): classifies
  change-data-capture defects that are *not* nulls — stale, late (past-watermark),
  out-of-order, duplicate-key, invalid-operation, missing-event-time, and
  replay-risk batches — with per-key ordering, an explicit-watermark mode, and
  freshness/ordering/CDC trust penalties (each `0..1`). Read-only; never imputes.
  `CDCReport` supports `.summary()` / `.to_dict()` / `.to_json()` / `.to_frame()` /
  `.passed` / `.trust_penalties` / `.freshness_seconds`.
- New **provenance-aware cleaning** for document/OCR-extracted tables (module
  `freshdata.provenance`): `fd.clean(df, source_provenance=..., return_report=True)`
  and `clean_enterprise(..., source_provenance=...)` preserve per-column
  `source_file` / `page` / `region` / `parser_confidence` / `extracted_at` and
  **warn when a low-confidence field is coerced or repaired**
  (`provenance_confidence_threshold`, default `0.7`). The summary lands at
  `CleanReport.source_provenance` and in `.to_dict()`. FreshData is the
  post-extraction normalization/audit layer, not a PDF parser.
- New **baseline-free contract schema diff** (`fd.diff_schema(df, contract=...)`,
  exposed lazily from `freshdata.enterprise.contracts`): explains structural schema
  drift *before* any repair runs, with no persisted baseline required. Reports
  added/unexpected, removed, **renamed**, dtype, nullability, and semantic-domain
  drift, returning a `DriftReport` with a structured `contract_results`
  categorization and `.summary()` / `.to_dict()` / `.to_json()` / `.to_frame()`
  exports. Policies `on_unexpected` (`fail|warn|preserve`) and `on_missing`
  (`fail|warn|ignore`) control the gate. Rename detection is **evidence-based**
  (matching semantic type or high name similarity over a dtype-compatible pair),
  so unrelated same-dtype columns are never reported as renames. `fd.profile(df,
  contract=...)` attaches the same diff at `profile.schema_diff`. `DriftReport`
  also gains a `.to_frame()` exporter shared with `monitor_contract` /
  `compare_to_baseline`. Read-only; never mutates input.
- **Contract gate in `fd.clean` and `fd.suggest_plan`** (`contract=`, `on_unexpected=`,
  `on_missing=`): runs `diff_schema` on the input *before* repair. A failing gate
  (errors in the diff) raises `ContractViolation` (carrying the `DriftReport` at
  `.report`); otherwise the diff is attached to the `CleanReport` as a JSON-friendly
  `contract_violations` section that surfaces in `.summary()` and `.to_dict()`.
  `fd.suggest_plan(df, contract=...)` exposes the same diff at `plan.schema_diff`.
  In-memory pandas engine only; never auto-renames or drops on the basis of a diff.
  `CleanReport` gains a `contract_violations` field.
- New **wide-schema / large-frame perf controls on `fd.profile`**: `profile_sample=N`
  profiles a deterministic N-row sample (stats become estimates), `max_columns=M`
  caps profiling to the first M columns, and `lazy_report=True` skips the expensive
  full-frame duplicate-row scan. When any is used the `Profile` describes the
  profiled *subset* and records the totals at `profile.materialization` (also in
  `.to_dict()`). `build_profile` gains matching `sample=` / `max_columns=` / `lazy=`
  keyword-only parameters; default behaviour is unchanged.
- New **two-frame entity-resolution wrapper** `fd.link(left, right, keys=...,
  strategy="exact"|"fuzzy"|"external")` (also `freshdata.enterprise.link`): the
  ergonomic front door over `link_entities`. Builds the resolution config from
  `keys` + `strategy`, returns an `EntityResolutionReport` with candidate pairs,
  confidence scores, per-field explanations, and a steward-reviewable structure.
  `strategy="external"` formats an adapter callable's pairs (e.g. Dedupe) without
  re-implementing it. Defaults to the pandas backend (no optional deps); supports
  a `blocking=` override and `return_linked=`.
- **Privacy/regulated-pipeline hardening on `MaskingRule`**: `strategy="token"` is
  now accepted as an alias for the reversible `tokenize` strategy, and rules gain
  `retention_days`, `policy_id`, and `policy_reason` fields. `MaskReport` (from
  `mask_dataframe`) now surfaces per-column `retention` and an auditable
  `policy_provenance` list (which rule masked each column, with what strategy,
  under which policy id, and why), both exported via `.to_dict()`. FreshData
  records the declared retention policy for audit; it does not enforce deletion
  and makes no automatic compliance claims.
- New **compliance-grade privacy policy engine** (`freshdata.enterprise.privacy_policy`,
  exposed as `fd.PrivacyPolicy` / `fd.PrivacyRule` / `fd.CompliancePack` / `fd.Jurisdiction`
  / `fd.apply_privacy_policy` / `fd.load_privacy_policy` / `fd.load_compliance_pack`): turns
  the masking primitives into a declarative, **jurisdiction-aware** (US / EU / UK / India /
  Global) policy with actions `classify` / `tokenize` / `pseudonymize` / `redact` / `drop`
  / `minimize` / `quarantine` / `preserve_with_reason`. Ships built-in **HIPAA, FERPA, PCI
  and GDPR** rule packs (YAML under `freshdata/compliance/packs`) combining column-name,
  value-regex, context and entity/domain-pack classifiers; PCI card numbers are gated by a
  Luhn check. Policies load from YAML/JSON. Reversible tokenisation uses pluggable vault
  backends (`memory` / `json` / `sqlite`, via `fd.make_vault`) and requires an explicit vault
  **and** key; `detokenize_series` reverses only with both. The returned `PrivacyReport` gains
  a **Data-Trust privacy dimension** (`sensitive_fields_detected` / `_touched`,
  `unprotected_sensitive_fields`, `policy_violations`, 0–100 score), per-column audit fields
  (`rule_id`, `action`, `legal_basis_or_reason`, `jurisdiction`, `compliance_pack`), plus
  `to_frame()` / `to_json()`. Reports redact previews and never expose vault secrets by
  default. The legacy `detect_pii` / `anonymize` / `check_k_anonymity` / `MaskingRule` /
  `PrivacyReport` API is unchanged.
- New **schema-drift & data-contract monitoring** (`freshdata.enterprise.contracts`,
  exposed as `fd.build_baseline` / `fd.save_baseline` / `fd.load_baseline` /
  `fd.compare_to_baseline` / `fd.monitor_contract`): record a versioned, PII-safe
  `DatasetBaseline` (schema + numeric/categorical/datetime statistics) for a trusted
  dataset, persist it as JSON (`"schema_version": "freshdata-baseline-v1"`), then detect
  schema drift, distribution drift (dependency-free **KS** statistic and **PSI** over
  baseline quantile/frequency bins), `DataContract` violations (dtype/nullable/unique/
  allowed-values/min-max/regex/cardinality), and a **trust-score quality gate**. Baselines
  never store raw sample values unless `include_samples=True`; category labels are hashed
  by default. Configured via `DriftConfig`. Findings are JSON-serialisable and the input
  frame is never mutated.
- New **stronger PII detection + reversible / format-preserving anonymization**
  (`freshdata.enterprise.privacy`, exposed as `fd.detect_pii` / `fd.anonymize` /
  `fd.check_k_anonymity`): a Presidio-style but dependency-free detector (regex + context
  keywords, optional Presidio NER behind the `[privacy]` extra) across 15+ entity types
  with HIPAA/GDPR context boosting; reversible **tokenization** with an in-memory or JSON
  `TokenVault` (`tokenize_value` / `detokenize_value`); **surrogate**/`fpe`
  format-preserving anonymization (clearly flagged as *not cryptographic FPE* unless
  `pyffx` is installed); HIPAA/GDPR-tagged `MaskingEvent` audit records that redact raw
  previews by default (`audit_include_pii=True` to include them); and a `check_k_anonymity`
  re-identification report. `MaskingRule` gains `tokenize`/`fpe`/`surrogate` strategies plus
  `entity_types`/`reversible`/`key`/`key_env`/`token_vault_path`/`preserve_format`/
  `hipaa_tags`/`gdpr_tags`; all existing strategies keep working unchanged.
- New **probabilistic entity resolution at scale** (`freshdata.enterprise.entity_resolution`,
  exposed as `fd.resolve_entities` / `fd.link_entities`): a Splink-style, **DuckDB-backed**
  record-linkage backend (with a pandas fallback) that blocks candidate pairs via SQL
  predicates, scores them with weighted comparisons (exact / Jaro–Winkler / Levenshtein /
  numeric & date distance / phonetic Soundex / custom SQL — all pure-Python primitives),
  and builds entity clusters via connected components with a completeness-based canonical
  record. A hard `max_pairs` gate prevents cartesian explosions. Configured via
  `EntityResolutionConfig` / `BlockingRule` / `ComparisonLevel`. Documented as
  rule-weighted probabilistic linkage (not full EM-trained Splink parity).
- `EnterpriseConfig` gains `drift` / `privacy` / `anonymization` / `k_anonymity` /
  `entity_resolution` sub-configs and `enable_contracts` / `enable_privacy_detection` /
  `enable_entity_resolution` toggles; `clean_enterprise` accepts `baseline=` / `contract=`
  and `EnterpriseResult` now carries `drift_report` / `privacy_report` /
  `k_anonymity_report` / `entity_resolution_report`. New optional extras `[privacy]` and
  `[entity-resolution]`, plus examples `schema_drift_monitoring.py`,
  `privacy_anonymization.py`, and `entity_resolution_duckdb.py`.
- New **FHIR R4 JSON parser** (`fd.parse_domain(source, format="fhir")`): flattens a
  Bundle, a single resource, a list of resources, a JSON string, or a file path into
  `patient`/`observation`/`encounter`/`condition`/`medication_request` frames whose
  columns line up with the healthcare validators. The **healthcare pack now validates
  Condition and MedicationRequest** (FHIR R4 clinical-status / status / intent value sets,
  ICD-10 codes against a documented common sample, ISO-8601 dates), adds **UCUM** unit
  validation on Observations via the reference layer, and auto-detects all five resources.
  Resource IDs are never imputed; `patient_id` stays PHI-masked unless
  `audit_include_phi=True`; unsupported resource types are recorded as warnings, not
  dropped. The **HL7 v2 parser** now also parses the `OBR` segment (an `order` frame, with
  each `OBX` linked to its order).
- New **format parsers** (`freshdata.parsers`) and `fd.parse_domain` /
  `fd.clean_domain_file`: structural readers that turn HL7 v2 ER7 (MSH/PID/PV1/OBX →
  patient/encounter/observation, with LOINC/SNOMED/ICD-10 code-system URIs), GPX
  (waypoints/routes/tracks), SDMX-ML (audit-only observations), and UN/EDIFACT
  (segments/elements, honoring `UNA` delimiters + the release character) into DataFrames.
  Parsers register via a `freshdata.parsers` plugin registry; malformed input is recorded
  in `ParseResult.warnings` rather than raising.
- New **centralized reference-data layer** (`freshdata.domains.reference`): one cached,
  normalizer-aware `load_reference(...)` over the bundled code sets (ISO-4217, ISO-3166,
  UN/CEFACT units, plus new **UCUM** and **UN/LOCODE** samples), each with a `_meta`
  version/disclaimer block. Supports case-sensitive/insensitive matching, synonym
  coercion, and an `invalid_mask` for validators.
- New finance **tick mode** (`fd.clean(df, domain="finance", finance_mode="tick")`):
  validates market tick/trade data — ISO-8601 non-future timestamps, positive price/size,
  ISO-4217 currency (via the reference layer), non-crossed quotes (`bid <= ask`),
  duplicate-tick detection, and BCBS-239 / SOX-style completeness controls. Symbol and
  exchange are IDs and are never imputed; the default `finance_mode="ledger"` is unchanged.
- New **energy (SCADA / Modbus)** domain pack: `fd.clean(df, domain="energy")` validates
  point-level telemetry — one row per `(asset_id, register_address, timestamp)` reading —
  against common Modbus/SCADA conventions: the 16-bit register-address range (0–65535),
  the public Modbus function codes (1, 2, 3, 4, 5, 6, 15, 16), OPC/SCADA point quality
  (`good`/`bad`/`uncertain`/`stale`/`null`, with synonym coercion), engineering units, and
  non-future ISO-8601 timestamps. Asset IDs are never imputed; bad/stale/uncertain readings
  and function/register-class mismatches are flagged for audit rather than dropped. Bundled
  reference data ships with `_meta` version/disclaimer notes documenting that these are
  common public conventions, not exhaustive vendor specifications. The validator is
  stateless per frame, so it composes with micro-batch streaming.
- New `freshdata.streaming` subpackage and `fd.StreamingCleaner` for **streaming /
  micro-batch cleaning** of datasets larger than memory. It consumes pandas (and,
  when installed, PyArrow `Table`/`RecordBatch` and polars `DataFrame`/`LazyFrame`)
  batches, keeps **bounded** running statistics across them — Welford mean/variance,
  reservoir-sampled medians, Space-Saving top-k categories — and emits the same
  explainable `CleanReport` per micro-batch, now carrying a `streaming` block with
  `batch_id`, rows seen, and per-batch / rolling / cumulative trust scores plus a
  schema-drift flag. Imputation runs in a warmup phase (collect stats, defer and
  audit) then a stable phase (impute from running stats), preserving every
  leakage-aware safety gate (ID/target/free-text). Optional source connectors
  (`clean_kafka`, `clean_arrow_flight`) sit behind new `freshdata[kafka|flight]`
  extras and raise a clear `ImportError` when absent. New CLI subcommands
  `freshdata stream`, `stream-kafka`, and `benchmark-stream` process CSV/Parquet
  batch-by-batch with per-batch + summary reports and a trust-gate exit code, and
  `benchmarks/bench_streaming.py` proves stable memory across a lazily-generated
  100M-row stream. `CleanReport` serialization stays backward compatible (no
  `streaming` key for normal in-memory cleans).
- New `freshdata.execution` subpackage: a pluggable, out-of-core / Arrow-native
  execution engine. `fd.clean()` gains keyword-only `engine` (`"pandas"` |
  `"polars"` | `"duckdb"` | `"auto"`), `output_format` (`"pandas"` | `"polars"` |
  `"arrow"`), and `engine_config` (`EngineConfig`) arguments — all backward
  compatible; default callers are unchanged. The **Polars** backend cleans
  `LazyFrame`/Parquet sources with projection/predicate pushdown and streaming
  collection; the **DuckDB** backend cleans via staged SQL with spill-to-disk
  under a configurable `memory_limit`. Both reproduce the deterministic
  representation-repair + structural-reduction + full-row-dedup subset natively
  (identical `CleanReport` to the pandas pipeline) and transparently fall back to
  pandas for the accuracy-first decision engine, dtype heuristics, and opt-in
  impute/outliers. `engine="auto"` picks a backend from the source type and row
  count, and `fd.clean("data.parquet")` now also reads a file path directly. New
  optional extras: `freshdata[polars|duckdb|pyarrow|outofcore|bench]`.
- New `freshdata.benchmarks` harness (`python -m freshdata.benchmarks.run_benchmarks`)
  that generates synthetic Parquet at a target row count without materialising it,
  then times `fd.clean` across the pandas/polars/duckdb backends (wall time, peak
  resident memory, throughput, Data Trust Score). See
  `src/freshdata/benchmarks/RESULTS.md` for a 10k–10M reference run.
- New `freshdata.integrations` subpackage with first-class orchestration hooks for
  **Dagster** (`freshdata_asset_check`, `FreshDataResource`), **Airflow**
  (`FreshDataCleanOperator`), and **dbt** (`FreshDataDbtTransform`, the `dbt-gate`
  CLI, and a `freshdata_trust_gate` macro). A framework-agnostic core,
  `evaluate_trust_gate(df, ...) -> (DataFrame, TrustGateResult)`, cleans a frame and
  gates it on the 0-100 Data Trust Score, reacting to a low score with
  warn / fail / skip. Each adapter is an opt-in extra
  (`freshdata[dagster|airflow|dbt|integrations]`) and imports cleanly without its
  framework; a compliance bundle is attached to the gate report when
  `freshdata.compliance` is available.
- New `freshdata.compliance` subpackage that maps a `CleanReport` onto regulatory
  control frameworks and emits standards-grade audit artifacts via
  `generate_compliance_report(report, frameworks=[...]) -> ComplianceBundle`.
  Five frameworks ship: `21cfr_11` (21 CFR §11.10(e) audit trail), `gdpr_30`
  (Article 30 + 17), `alcoa_plus` (ALCOA+ data integrity), `sox_404`
  (transformation controls), and `hipaa_safe_harbor` (18-identifier coverage).
  Reports are purely additive and report-only (never mutate the input). Optional
  `dataframe=` recovers column roles/missing ratios via `infer_roles`, and
  `enterprise_result=` folds in the Data Trust Score, PII-masking events, and
  clustering lineage. `ComplianceConfig.strict_cfr_normalization` (default
  `False`) toggles whether lossless normalising rewrites count as obscuring for
  the 21 CFR gate.
- Four new domain validator packs: `healthcare` (FHIR/US Core — `Patient`,
  `Observation`, `Encounter` with `fhir_resource=`/auto-detection), `education`
  (Ed-Fi), `agriculture` (ADAPT, with area/yield unit coercion), and `media`
  (EIDR/DDEX via `media_type=`/auto-detection, with tested EIDR Mod 37,2 and ICPN
  GS1 mod-10 check digits). Healthcare/education redact PHI in the audit trail as
  `[PHI]` unless `audit_include_phi=True`. `fd.clean` gains optional `fhir_resource`,
  `media_type`, and `audit_include_phi` keyword arguments.
- P1 repair-layer primitives for validator bridges, schema drift
  harmonization, duplicate/replay defense, and human review queues.
- Top-level bridge adapters: `freshdata.from_gx`, `freshdata.from_dbt_failures`,
  `freshdata.from_pandera_errors`, `freshdata.emit_gx_expectations`, and
  `freshdata.emit_dbt_tests`.

### Fixed
- **Packaging: the distribution is `freshdata-cleaner` again.** A recent commit
  renamed the project back to `freshdata`, a name PyPI rejects as too similar
  to the existing `fresh-data` project (the exact collision that forced the
  original rename). `pyproject.toml`, every in-source install hint, the docs,
  and the packaging tests now agree on `pip install freshdata-cleaner`
  (import name unchanged: `import freshdata`).
- **MissForest: `<col>_was_missing` indicators are no longer all-False.** The
  indicator was computed *after* the column had been imputed, so it never
  marked any row (and `missforest_add_indicators="auto"` never fired at all).
  Indicators now come from the pre-fill missing mask and the pre-computed
  column context, matching the standard imputation engine.
- **Outliers: an explicit `outlier_action` is now honored.** Under the default
  `strategy="balanced"`, `outlier_action="cap"` (and `"remove"`) was silently
  downgraded to `"flag"`, so capping never happened despite being the documented
  default — extreme values were returned unchanged. Explicit
  `"cap"` / `"remove"` / `"flag"` are now applied to every eligible numeric
  column.
- **Small frames no longer skip outlier handling.** The engine's minimum
  non-null threshold dropped from 10 to 4 (the floor at which IQR / z-score
  fences are defined), so outliers in small DataFrames are detected and handled.

### Changed
- The default `outlier_action` is now `"auto"` (context-aware: flags under
  `balanced`, caps under `aggressive`, flags heavy-tailed >15%-outlying
  columns). The default *behavior* under `balanced` is unchanged (still flags);
  only the explicit-directive path changed. An explicit `cap` / `remove` on a
  heavy-tailed column now caps / removes and emits a warning instead of silently
  flagging.

## [1.0.0] - 2026-06-14

First stable release. The public API is now considered **stable under Semantic
Versioning** — breaking changes will require a 2.0.

### Changed
- Promoted the package to **Production/Stable** (`Development Status :: 5`).

### Notes
- No behavioral changes versus 0.5.0. The stable public surface is
  `fd.clean`, `fd.profile`, `fd.suggest_plan`, `fd.compare_plans`,
  `fd.compare_clean`, `fd.explain_clean`, `fd.infer_roles`, `fd.Cleaner`,
  `fd.CleanConfig`, `fd.CleanReport`/`fd.Action`, `fd.Profile`, and the lazily
  imported `freshdata.enterprise` layer.
- Install: `pip install freshdata-cleaner`; import: `import freshdata as fd`.

## [0.5.0] - 2026-06-14

### Added
- **Documentation site** built with MkDocs Material and deployed to GitHub
  Pages (<https://freshcode-org.github.io/freshdata/>): installation,
  quickstart, cleaning-engine, profiling, feature overview, benchmarks,
  auto-generated API reference (mkdocstrings), FAQ, and contributing guides,
  with search, dark/light mode, OpenGraph metadata, `sitemap.xml`, and
  `robots.txt` for SEO/AI discoverability.
- **`examples/`** — 8 runnable scripts (missing values, outliers,
  normalization, profiling, ML pipeline, large datasets, pandas integration,
  CSV automation) and **`notebooks/`** — 3 reproducible Jupyter walkthroughs.
- **Packaging governance**: `MANIFEST.in`, `SECURITY.md`, `RELEASE.md`,
  `.pre-commit-config.yaml`, a tag-triggered PyPI release workflow
  (`release.yml`) using Trusted Publishing, a docs-deploy workflow
  (`docs.yml`), and an issue-template chooser config.
- Expanded PyPI keywords and classifiers and a `Documentation` project URL for
  better search ranking and discoverability.

## [0.4.0] - 2026-06-14

### Added — enterprise layer (`freshdata.enterprise`)
- **`clean_enterprise(df)`** and the reusable **`FreshDataEnterprise`** pipeline:
  core cleaning → fuzzy value clustering → semantic validation → PII masking, returning
  an `EnterpriseResult` (cleaned frame + trust scores + quality report + lineage). Accepts
  and returns **pandas *or* polars** — Polars-native on the hot path when installed, with a
  vectorized pandas fallback otherwise.
- **Data Trust Score** (`compute_trust_score`, `TrustScore`): a 0–100 score from
  completeness, validity, uniqueness, and structural consistency, with per-column detail
  and a JSON/Markdown **`QualityReport`** (`build_quality_report`).
- **Value clustering** (`merge_clusters`, `cluster_column`): OpenRefine-style fingerprint
  key-collision and n-gram merging of variants/typos, built from native Polars string
  expressions (pandas fallback), with `most_frequent` / `longest` / `shortest` / `first`
  canonicalisation.
- **PII masking** (`mask_dataframe`, `MaskingRule`): salted SHA-256 `hash`, `redact`,
  `partial`, `regex_scrub` (built-in email/phone/SSN/credit-card/IP/IBAN patterns), and
  `drop`; null-preserving and frame-type-preserving.
- **Semantic validation** (`SemanticValidator` + `ReferenceSetValidator` / `RegexValidator`
  / `CallableValidator` / `APISemanticValidator`, `run_semantic_validation`), including a
  built-in ISO-3166 `iso_country_validator`.
- **Lineage** (`LineageTracker`, `schema_of`): records who/when/input-schema/output-schema/
  rule per step and exports OpenLineage-compatible `START`/`COMPLETE` RunEvents (schema +
  column-lineage facets) with no hard dependency on the OpenLineage client.
- **Optional Cleanlab wrappers** (`detect_label_issues`, `detect_outliers`) with a clear
  install-hint error when cleanlab is absent.
- **CLI** (`freshdata`): `clean` / `profile` / `trust` subcommands reading CSV/Parquet/JSON,
  emitting JSON quality + OpenLineage reports, with a non-zero exit code on trust-gate
  failure — suitable as an Airflow/Prefect batch step. Config via JSON/YAML files.
- New optional-dependency extras: `pyarrow`, `semantic`, `cli`, `cleanlab`, aggregate
  `enterprise`, and `all`. Polars/PyArrow/requests/cleanlab are imported lazily, so plain
  `import freshdata` stays dependency-light.

## [0.3.0] - 2026-06-12

### Changed (breaking)
- **Default strategy is now `"balanced"`** — accuracy-first cleaning that
  preserves high-missing columns, flags outliers instead of capping, and
  skips KNN imputation. Use `strategy="aggressive"` for v0.2-style scrubbing
  (KNN, column drops, winsorization).
- `strategy="auto"` is deprecated (alias for `"aggressive"`; emits
  `DeprecationWarning` once per process).

### Added
- `fd.suggest_plan(df)` and `fd.compare_plans(df)` — dry-run previews of
  engine model choices per column, with ranked alternatives.
- Model selection router (`engine/model_select.py`) scoring imputation and
  outlier actions; `Action.model_id` records the chosen model.
- Expanded target/label heuristics (`aqi`, `*_bucket`, `score`, …) and
  domain-sensitive outlier preservation (pollutants, prices, latency, …).
- `profile(df, include_plan=True)` attaches a `CleanPlan` at `profile.plan`.
- `src/freshdata/py.typed` marker for PEP 561 typing support.
- Multi-dataset regression suite (`tests/fixtures/`, `test_regressions.py`,
  `test_realworld.py`, `test_model_select.py`, `test_plan.py`).
- Golden report snapshots (`tests/fixtures/golden/`, `pytest --update-golden`).
- Benchmark tests (`test_benchmark.py`) and `benchmarks/bench.py --fixtures`.
- CI enforces ≥93% coverage and treats `freshdata` warnings as errors.
- README migration guide for 0.2 → 0.3.

### Fixed
- KNN imputation: collinearity pruning, row-count gate (10k), warning
  suppression, index alignment on fill.
- Re-cleaning idempotency for outlier flag columns.

### Added (0.3.1 validation pass)
- `fd.compare_clean()` — side-by-side quality + efficiency metrics per strategy.
- Four new scenario fixtures: `large_panel` (3k rows), `duplicate_heavy`,
  `locale_numbers`, `mixed_roles`.
- Performance baselines (`tests/fixtures/perf/baselines.json`) with 25% regression gate.
- `@pytest.mark.large` optional full AQI.csv benchmark (`FRESHDATA_AQI_PATH`).
- Engine perf: one-pass `EngineCache` (contexts + correlation matrix), lazy
  informative-missing checks, sampled skew on large columns.
- `benchmarks/bench.py --compare` table output.

## [0.2.0] - 2026-06-12

`fd.clean(df)` now performs real, context-aware automatic cleaning by
default, driven by a rule-based decision engine.

### Added
- **Decision engine** (`strategy="auto"`, the new default): profiles every
  column (missing ratio, dtype, skewness, cardinality, inferred role,
  informative missingness) and applies threshold rules for missing values
  and outliers. Every action — including deliberately preserving a column —
  is logged with a rationale, risk level, and confidence score.
- Missing-value bands with configurable thresholds
  (`missing_threshold_low/medium/high`, defaults 0.05/0.30/0.60): contextual
  mean/median/mode/sentinel/ffill imputation, KNN imputation for correlated
  numeric features (scikit-learn optional), column drops for
  high/extreme missingness with logged reasons, `<col>_was_missing`
  indicator columns when missingness is informative.
- Column-role inference: targets are never modified, IDs are never imputed,
  free text is never force-filled, datetimes use time-aware fills.
- Outlier engine: `outlier_action="cap"` (default) / `"remove"` / `"flag"` /
  `None`; `outlier_method="auto"` (z-score for ~normal, IQR for skewed) and
  `"isolation_forest"`; heavy-tail protection (flag instead of cap);
  domain-sensitive columns (fraud/anomaly/risk) keep their extremes.
- Duplicate rules: `duplicate_keep="first"/"last"/"drop"/"aggregate"`,
  `duplicate_threshold` data-quality warning, time-indexed frames protected
  unless `allow_timeseries_duplicates=True`; count and percentage reported.
- New `clean()` parameters: `strategy`, the threshold options,
  `outlier_action`, `preserve_original`, `return_report`, `verbose`,
  `preserve_columns`, `target_column`, `id_columns`, `advanced_imputation`,
  `missing_indicators`.
- Report upgrades: per-action `rationale`/`risk`/`confidence`, missing cells
  before/after, duplicates removed, outliers handled, columns
  dropped/imputed/preserved, `warnings`, `recommendations`, and a compact
  `brief()` used by `verbose=True`.
- Optional extra: `pip install "freshdata-cleaner[ml]"` for scikit-learn.

### Changed
- **Default behavior**: statistical cleaning now runs by default. Pass
  `strategy="conservative"` for the 0.1.x representation-only behavior;
  explicit `impute=` / `outliers=` still override the engine.
- `report.to_frame()` gained `rationale`, `risk`, and `confidence` columns.
- `verbose=True` (default) prints a one-line summary per clean.

## [0.1.0] - 2026-06-12

Initial release.

### Added
- `freshdata.clean()` — automatic, audited cleaning: column-name
  normalization, whitespace stripping, sentinel-string normalization,
  empty row/column pruning, validated dtype inference (numeric incl.
  currency/thousands separators, datetime, boolean), and exact duplicate
  removal.
- Opt-in steps: imputation (`auto`/`mean`/`median`/`mode`), outlier
  clipping/flagging (IQR or z-score), constant-column dropping, memory
  optimization (numeric downcasting + category conversion), index reset.
- `freshdata.profile()` — read-only profiling whose dtype suggestions are
  produced by the same inference code `clean` uses.
- `freshdata.Cleaner` — reusable configured pipeline with `report_`.
- `freshdata.CleanConfig` — frozen, self-validating configuration;
  unknown options raise with a "did you mean" suggestion.
- `freshdata.CleanReport` / `freshdata.Action` — structured audit trail
  with `summary()`, `to_dict()`, `to_frame()`.
- Type hints throughout (`py.typed`), zero dependencies beyond
  pandas/numpy, support for Python 3.9–3.13.
