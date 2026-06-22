# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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
