# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `fd.clean_excel()`, the Excel companion to `fd.clean_csv()`: reads one sheet,
  cleans it, and optionally writes the result, with formula sanitization on by
  default. Needs the new `excel` extra (`openpyxl`).
- `CleanReport.to_json()` and `CleanReport.write_json()` for first-class audit
  report serialization without manual `json.dumps(...)` calls.
- Added a runnable PyJanitor interoperability example that demonstrates both
  tool orderings while keeping PyJanitor optional.
- A dependency-optional Great Expectations recipe demonstrating the
  repair-then-validate workflow with an in-memory checkpoint.
- `TimeSeriesCleanConfig.timestamp_unit` (`"s"`, `"ms"`, `"us"` or `"ns"`) sets
  the epoch unit for numeric timestamp and event-time columns in time-series
  streaming. Without it the unit is inferred from the values (#227).
- `apply_review_decisions()` accepts a keyword-only `queue=` (the
  `ReviewQueueReport` the reviewer worked from), so decisions that carry only
  an `item_id` can be resolved to a record pair (#267).
- The `FRESHDATA_MODEL_TIMEOUT` environment variable sets the network timeout,
  in seconds, for `fd.models.pull` downloads (default 60) (#341).
- `cdc_profile` accepts `event_time_unit` (`"s"`, `"ms"`, `"us"` or `"ns"`) to
  set the epoch unit for numeric event-time columns. Without it the unit is
  inferred from the values (#327).
- `FreshDataDbtTransform` accepts `audit_name`, the stem of its audit file, and
  `run()` accepts a keyword-only `raise_on_fail`. With `raise_on_fail=False` a
  failing gate returns its result instead of raising (#343, #344).
- `load_cleaning_memory` accepts a keyword-only `dataset_id` that selects one
  memory from a SQLite store holding several (#306).
- `ModelConfig.file_sha256` pins a checksum for each file of a model, alongside
  the existing `sha256` pin for the primary file (#346).

### Changed
- `explain_clean()` now profiles only post-clean columns that can contribute a
  decision narrative, avoiding a redundant full-width context pass.
- The `polars`, `outofcore`, `enterprise`, `all` and `dev` extras now require
  `polars>=1.0`. The Polars engine uses `LazyFrame.collect_schema()`, which
  0.20.x lacks (#214).
- The Polars engine reads float `NaN` from native Polars, Arrow and file sources
  as null, as pandas input already was. Polars output for those sources shows
  null where it used to show `NaN` (#200).
- Requesting another engine's native handle (for example `engine="duckdb"` with
  `output_format="polars-lazy"`) now raises `ValueError` instead of returning a
  different type. `engine="auto"`, or no `engine`, picks the engine that owns
  the format (#205).
- When every column is dropped as empty, the DuckDB and Polars engines return a
  zero-column frame that keeps the row count, as the pandas pipeline does.
  Polars and Arrow output cannot hold rows without columns; the report records
  that difference (#201).
- `freshdata validate` exits 2 ("could not load rules") instead of 1 when a
  `--suite` or `--contract` file is not an object, and
  `ValidationSuite.from_dict` raises `ValueError` for non-mappings (#289).
- `dbt-gate` no longer passes when nothing was gated. A file without a `nodes`
  mapping (such as `run_results.json`) raises `ValueError`, `all_passed` is
  false when no models were processed, and `--fail` exits 1. Ephemeral and
  disabled models are skipped and listed under a new `skipped` summary key
  instead of counting as failures (#296, #249).
- Semantic repairs that could rewrite a valid value are now suggested for
  review instead of auto-applied: fuzzy cleaning-memory matches, percent values
  in `rate`/`ratio` columns whose scale is fractional or unknown, and shape
  alignments of unseparated values. Shape alignments whose groups do not match
  the template are no longer proposed (#252, #253, #254).
- The EIDR check character (MD-C002) now uses the hybrid ISO 7064 MOD 37,36
  system from the EIDR ID Format spec, so published EIDR IDs validate. IDs
  whose check character came from the old MOD 37-2 code are flagged, and a `*`
  check character is always rejected (#259).
- The finance FIN-003 guard leaves ambiguous DD/MM vs MM/DD dates unresolved
  even when a time follows the date, instead of reading them month-first
  (#260).
- HIPAA Safe Harbor reports now rest on column evidence. `fd.clean` and
  `fd.apply_plan` record the input columns, so identifier columns that
  cleaning did not touch are detected without `dataframe=` and reports that
  used to pass can fail. Reports with no full column list (synthetic reports,
  native backends, streaming and multi-file runs) set
  `coverage_verifiable: False`, add a warning and do not pass (#245).
- HIPAA identifier hints match whole column-name tokens, so `ip` no longer
  matches `description` or `ship_address`. Hints of four characters or fewer
  no longer match inside run-together lowercase names such as `visitdate`;
  separated and camelCase forms still match (#283).
- Blocking rules the pandas entity-resolution backend cannot evaluate (`OR`,
  comparison operators, literals, arithmetic, `BETWEEN`, parenthesised
  predicates, unquoted names with spaces or hyphens) now raise
  `EntityResolutionError` instead of returning zero or wrong pairs. Quote such
  names, for example `l."first name" = r."first name"` (#237).
- Entity resolution treats NaT and `pd.NA` as missing, so they no longer count
  as agreement and records previously merged on missing datetimes can split
  (#238).
- `clean_enterprise` raises `ValueError` when `EnterpriseConfig.anonymization`
  is non-empty, instead of silently ignoring a setting it does not apply.
  `EnterpriseConfig` still accepts the field, and the CLI prints a one-line
  error and exits 1 (#247).
- `TrustScoreWeights` rejects NaN and infinite weights with `ValueError`
  instead of producing a `nan` trust score (#277).
- Time-series streaming reads numeric timestamp and event-time columns as
  epochs instead of 1970 dates, recording the unit in a
  `timeseries_timestamp_parse` action. Unparseable timestamps keep their row
  and are reported in `coerced_cells`/`coerced_rows` with a warning, and
  mixed-offset batches become a UTC-aware column (#227, #250).
- `TimeSeriesCleanConfig(anomaly_window_size=1)` is rejected when the config
  is built (#290).
- `cdc_profile` measures freshness against the current UTC time by default,
  and naive `now=` and `watermark=` values are read as UTC (#233, part 4).
- A constant baseline column now produces `drift.ks` findings when current
  values move away from the constant. Saved baseline JSON stores statistics at
  full precision instead of six decimal places; existing files load unchanged.
  Some point-mass shifts score lower than before, because the higher scores
  came from the tie handling fixed in #234 (#235, #275).
- `load_review_decisions` reads CSV ids as strings (`"007"` stays `"007"`), and
  `apply_review_decisions` raises `ValueError` for a decision it cannot
  resolve to a pair instead of dropping it. `feedback_summary` gains an
  `n_unmatched` count, and clusters created by an apply get ids numbered past
  `n_records` (#239, #267, #268).
- Learned clean values in cleaning-memory JSON are plain JSON types: integers
  stay numbers, and Timestamps and Decimals are strings (#256).
- With `apply_plan(allow_drift=True)`, actions whose raw value is no longer in
  the column are recorded as skipped (`frame drift`) with count 0, and applied
  actions record the observed cell count instead of the plan-time
  `n_affected` (#258).
- `Parser.read_text` defaults to `encoding="utf-8-sig"` and strips a leading
  UTF-8 BOM from text input (#314).
- `mostly` thresholds are inclusive: a rule with exactly the allowed share of
  violating rows (for example 1 of 10 under `mostly=0.9`) warns and passes
  instead of failing (#307).
- `RepairPlan.decisions_hash` and `to_json` change for plans whose params hold
  sets (members are now sorted) or numpy scalars (now JSON numbers and bools),
  so the value no longer depends on `PYTHONHASHSEED` or the numpy version.
  Plans without such params keep their hash (#312).
- The context compiler no longer splits allowed values or dedup keys on `/`,
  and a bare number is no longer read as a confidence gate (`only if 3
  neighbours agree`); such phrases stay unparsed and raise under `strict`
  (#301, #303).
- `fd.clean(..., policy=..., strict=True)` with a schema-free policy now
  raises `protection_conflict`, as the `columns=` and `context=` flows do
  (#304).
- Phone validation rejects numbers with more than 15 digits (E.164) (#318).
- Quality-debt `duplicates` now counts duplicate rows detected in the cleaned
  output, not only rows removed, so frames with duplicates can warn or fail the
  gate under default options. `pii_risk` counts distinct PII columns instead of
  matching cells, so scores drop for tall PII columns (#264, #286).
- Network plugins registered without `allow_network=True` re-read
  `FRESHDATA_ALLOW_NETWORK_PLUGINS` at call time, so setting the variable after
  registration activates them and unsetting it deactivates them again (#299).
- Cleaning raises `ValueError` when an `impute_strategy` key, including one set
  by `Pipeline.impute(columns=)`, names no column after renaming, instead of
  silently imputing nothing. The message lists the unknown keys and available
  columns and suggests the normalized name for a pre-rename key such as `Age`.
  Keys for columns that a later step drops are still accepted (#310).
- `ExplainReport.cell_changes` is always keyed by `str(label)`. `explain_clean`
  and `infer_roles` raise `ValueError` naming duplicated column labels instead
  of `AttributeError` or `TypeError`, and `explain_clean` also raises for
  labels whose string forms collide, such as `1` and `"1"` (#232, part 3;
  #265, part 3).
- `suggest_join_keys` scores a field 0 when either value is missing (None, NaN,
  NaT, `pd.NA` or `""`) and leaves missing values out of exact-key overlap.
  Integer keys and NaN-promoted float keys render alike (`101` and `101.0`), so
  they overlap and share blocks (#272, #273).
- `is_valid_icpn` rejects values with surrounding text or separators other than
  spaces and hyphens, such as `"tel: 036000291452"`. Formatted UPC/EAN values
  such as `0-36000-29145-2` still pass (#321).
- `cdc_profile` reads numeric event-time columns such as Debezium `ts_ms` as
  epochs in an inferred unit, as `clean_timeseries` does, instead of as
  nanoseconds that gave 1970 dates. Small integers such as row numbers are read
  as seconds (#327).
- `cdc_profile` checks rows with a null CDC key for ordering as one group and
  counts them in a new `missing_key` warning, which does not affect `passed` or
  penalties. A negative `stale_after` raises `ValueError` (#325, #326).
- `build_baseline`, `compare_to_baseline`, `enforce_contract` and `diff_schema`
  raise `ValueError` for duplicate column labels or labels that collide as
  strings, and so do `fd.validate(suite=...)`, `fd.clean(contract=...)` and the
  enterprise drift step (#265, part 2).
- On pandas 2, `min_datetime`/`max_datetime` contracts parse string columns
  with mixed date formats instead of dropping values in a second format as
  `NaT`. Values that still fail to parse produce a warning-level
  `contract.unparseable_datetime` finding, and contracts that passed can now
  fail (#242).
- `FreshDataDbtTransform(on_low_score="fail").run()` raises `TrustGateError`
  when the gate fails, after writing the audit file, instead of returning a
  result with `should_fail=True`. `gate_manifest` still records failing models
  and gates every model (#343).
- `freshdata clean`, `freshdata stream` and `fd.clean_csv` load numeric-looking
  CSV columns with zero-padded values as text, so `02134` keeps its leading
  zero. Detection reads a 10,000-row sample (the first chunk when streaming)
  and is skipped when `read_csv_kwargs` sets `dtype` or `converters`, or with
  `preserve_leading_zeros=False` (#228).
- `freshdata stream` exits 1 without writing output when a later batch adds a
  column or cannot be cast to the first batch's Parquet schema (#248, part a).
- `load_cleaning_memory` on a SQLite store raises `ValueError` listing the
  stored ids when the store holds several memories and no `dataset_id` is
  given, `KeyError` for an unknown `dataset_id`, and `FileNotFoundError` for a
  missing path instead of creating an empty database (#306).
- Loading a `.fdprofile` raises `ProfileFormatError` for a truncated or
  malformed manifest or member, and for a manifest that does not cover every
  required member (#311).
- Model checksum pins cover every file. Once a model has a pin, every file
  needs one, or `ModelChecksumError` names the unpinned file before any
  download. `fd.models.pull` also checks files that are already installed and
  raises on a mismatch with a `force=True` hint (`freshdata models pull` exits
  2), and `status()` reports `verified: True` only when every file matches.
  Models without pins behave as before (#346).
- Rewriting `24:00` to `00:00` in time-only columns is suggested for review in
  every semantic mode instead of auto-applied, and is not learned into cleaning
  memory. The TruthBench logistics oracle moves log-06 `transport_time` `24:00`
  from REPAIR to REVIEW and adds log-07 `tracking_status` `on time` to
  `on-time` as the domain's exact repair (#305).
- `anonymize`, `detect_pii` and `apply_privacy_policy` leave `pd.NA` and `NaT`
  cells missing instead of rewriting them as `"<NA>"` or `"NaT"`, and
  `cells_changed` no longer counts them (#243).
- `detect_pii` raises `ValueError` naming duplicated column labels instead of
  `AttributeError`, and so does `anonymize` when detection is enabled or a rule
  targets a duplicated label (#265, part 1).
- `detect_pii` sets `metadata["ner"]` from whether NER actually ran, and new
  `ner_requested`, `ner_active` and `ner_error` keys report the request and
  outcome. When the Presidio analyzer cannot start, one `UserWarning` is
  emitted, the NER pass is skipped and start-up is not retried within the
  process (#282).
- When `fpe` cells used more than one mode, `metadata["fpe_mode"]` is `"mixed"`
  and `metadata["fpe_modes"]` maps each column to its per-mode cell counts.
  Single-mode runs report as before (#281, parts 1-2).
- `apply_privacy_policy` and `classify_columns` classify each column from every
  distinct non-null value instead of the first 200 cells, so columns longer
  than 200 rows may now be classified. The report metadata gains
  `classification_values_scanned` (#246).
- An in-scope inline `PrivacyPolicy` rule now takes priority over every pack
  rule; classifier specificity still decides within each group (#284).
- A policy rule's own key is resolved before the policy defaults:
  `rule.key_env`, then `rule.key`, then `policy.key_env`, then `policy.key`.
  Rules without a key resolve as before (#285).
- `apply_privacy_policy` and `classify_columns` raise `ValueError` for
  duplicate column labels or labels that collide as strings, such as `1` and
  `"1"` (#232, part 5).
- A `MaskingRule` column name matches every column with the same name or the
  same snake_case form, so `"First Name"` masks `first_name` and `"email"`
  masks both `email` and `Email`. Listed names that match no column are
  recorded under a new `MaskReport.unmatched_columns` key; they raise
  `ValueError` only with `MaskingRule(strict=True)` or
  `mask_dataframe(..., strict=True)`. `freshdata clean --mask COLUMN:STRATEGY`
  rules are strict, so it exits 1 with a one-line error when `COLUMN` matches
  no column, instead of masking nothing and exiting 0 (#251).

### Fixed
- Trust-gate integrations now validate `on_low_score` policies at configuration
  boundaries, rejecting typos instead of silently skipping failure handling (#345).
- The minimum supported numpy is now 1.22. The numpy 1.21.6 wheel bundles an
  OpenBLAS that segfaults on BLAS-backed matrix multiplies on current Apple
  Silicon Macs regardless of `OPENBLAS_NUM_THREADS`, so installs at the old
  floor could crash in any code path that multiplies float matrices
  (for example semantic similarity scoring on larger inputs).
- Median imputation no longer crashes with `OverflowError` on nullable integer
  columns (`Int8`/`Int16`/`Int32`/`UInt*`) containing missing values under
  numpy 2.5+. Affects `impute="median"`/`"auto"`, the default missing-value
  engine, `fill_missing`, MissForest seeding and seasonal time-series
  imputation.
- `engine="duckdb"` with `output_format="arrow"` or `"polars"` now fetches the
  result directly in that format instead of materializing a pandas frame with
  `fetchdf()` and converting it again (#52). Arrow output keeps DuckDB's
  column types (for example `DECIMAL` stays `decimal128`).
- The Polars engine's duplicate-detection and dedup row counts now run through
  the streaming collect path instead of re-evaluating the whole plan in memory
  (#53).
- `fd.clean_timeseries` and `StreamingCleaner` no longer crash with
  "cannot convert to 'float64'-dtype NumPy array with missing values" on
  nullable integer columns (`Int*`/`UInt*`) with gaps under pandas < 2
  (running statistics and short-gap interpolation). MissForest's convergence
  check uses the same NA-safe conversion.
- `engine="duckdb"` no longer fails with "STDDEV_SAMP is out of range" on float
  columns holding `inf` or `NaN`. Missing counts are exact instead of rebuilt
  from a rounded percentage, and outlier fences use finite values only (#199).
- The Polars engine counts float `NaN` as missing, so empty rows and columns
  are dropped as with pandas, and `±inf` no longer skews outlier fences (#200).
- The DuckDB engine really drops all-empty columns when every column is empty,
  instead of reporting the drop and returning them (#201).
- DuckDB full-row deduplication keeps the input row order and honours
  `duplicate_keep="first"`/`"last"` (#202).
- Polars `LazyFrame` inputs work on the default path and when a run falls back
  to pandas, instead of raising `TypeError` (#203).
- The DuckDB and Polars engines strip every Unicode whitespace character that
  Python's `str.strip()` removes (for example NBSP and `\x1c`–`\x1f`) (#204).
- Pandas inputs with a mixed-type object column or duplicate column labels take
  a recorded pandas fallback on the DuckDB and Polars engines. They no longer
  crash (Polars) or silently turn numbers into text (DuckDB) (#206).
- `CleanReport.revert()` no longer writes reverted values into the frame
  passed to it (#207).
- Imputation no longer corrupts nullable integers beyond ±2**53 by casting
  them to float64. `impute="mean"`/`"median"`, the default engine, `fill_missing`
  and `StreamingCleaner` keep the integer dtype and present values exact (#208).
- `StreamingCleaner.clean_batch` and `fd.clean_timeseries` no longer raise
  `KeyError` on non-string column labels (#209).
- `fd.fill_missing` no longer crashes on nullable integer columns with a
  fractional mean or median, or on nullable boolean columns. Duplicate column
  labels raise a clear `ValueError` (#210).
- `fd.detect_outliers` returns a plain boolean mask on nullable columns, and
  `fd.remove_outliers` no longer drops rows that are only missing.
  `remove_outliers`/`resolve_duplicates` with `inplace=True` raise on a
  non-unique index instead of dropping extra rows (#211).
- `pd.ArrowDtype` string columns and categorical text columns get whitespace
  and sentinel normalization. Categoricals keep their dtype (#212).
- Docs: the ydata-profiling comparison example, the CSV sanitization row in
  trust claims, and the README quickstart output now match current behaviour
  (#213).
- The missing-pyarrow error names the feature that needs it (for example Arrow
  output), and Parquet metadata reads no longer fail with `AttributeError` in a
  fresh process (#215).
- `freshdata clean --config` reports invalid YAML or JSON, non-object
  sections and unknown keys as a one-line error naming the file (exit 1)
  instead of a traceback, and `dbt-gate` does the same for malformed manifests
  and directory paths (#289).
- `freshdata clean` and `freshdata validate` no longer exit 1 after a passed
  gate when stdout cannot encode UTF-8 (#295).
- Semantic memory replay looks up the expert that learned a repair, so learned
  Unicode normalization and shape-alignment repairs replay instead of being
  checked as dates (#300).
- Retail GTIN checks read float-loaded integral cells as integer text, so a
  blank cell no longer causes a valid GTIN to be rewritten into a different
  one (#229).
- Healthcare date checks compare offset-aware FHIR `dateTime` values with naive
  dates or mixed offsets in UTC instead of raising `TypeError` (#233, part 2).
- The GDPR Article 30 report lists only the measures the run actually applied
  (#287).
- The DuckDB entity-resolution backend accepts non-equality blocking SQL such
  as `jaro_winkler_similarity(l.name, r.name) > 0.8` (#236).
- `fd.link(backend="duckdb")` works on keys containing spaces or hyphens
  (#266).
- `link_entities` and external `fd.link` reports record their thresholds, so
  `build_review_queue` orders items around the configured midpoint (#271).
- `StreamingCleaner(global_duplicates=True)` keeps a bounded window of recent
  rows instead of the first `window_size` rows forever, so duplicates of recent
  rows are removed for the whole stream (#292).
- Streaming cross-batch deduplication no longer misses duplicates when a
  column flips between integer and float dtypes (#293).
- Streaming distribution drift fires for a column that had been constant and
  then changes (#294).
- MAD anomaly detection no longer flags the minority value of two-valued or
  sparse series (#291), and anomaly columns stay stable when a batch's dtype
  changes, with text cells never scored or capped (#248, part).
- CSV review queues round-trip: the formula guard added on export is removed
  from ids on load, blank decision cells are skipped, and applying decisions
  keeps existing cluster ids and canonical records (#239, #240, #268).
- A frame no longer fails drift checks against its own baseline when values
  tie across stored quantiles (#234).
- `pd.ArrowDtype` columns (double, decimal, large string, dictionary) are
  recognised by contracts and baselines, and Arrow decimals no longer crash
  numeric profiling (#241).
- Semantic cross-field checks no longer raise or miss findings on frames with
  duplicate row labels (#231, part 1), integer column labels no longer raise
  `KeyError` in semantic repair (#232, part 1), and date-ordering checks
  compare tz-aware and naive values instead of raising `TypeError` (#233,
  part 1).
- `save_profile` no longer fails with `TypeError` on profiles learned from
  numpy or pandas clean values (#256), and `LearningProfile.merge` no longer
  modifies either parent's memory (#257).
- The test suite runs from an unpacked sdist and in isolation, and the
  streaming docs describe `rolling_trust_score` as an unweighted mean
  (#347, #348, #349).
- The FHIR parser records malformed resources (unexpected list or object
  shapes, non-string `resourceType`) as per-resource warnings instead of
  raising (#313).
- HL7v2, FHIR and EDIFACT parsers handle input that starts with a UTF-8 BOM
  (#314).
- The GPX parser skips points with NaN, infinite or out-of-range coordinates
  with a warning (#320).
- The context compiler keeps quoted values and values such as
  `Trinidad and Tobago` whole (#301), and dotted column names such as
  `file.name` no longer split the sentence (#302).
- `clean_text_value` is idempotent and returns text in the configured Unicode
  normal form (#316).
- `lint_text_encoding` no longer reports Japanese text such as `コーヒー`, or
  `Nº 5`, as mixed script, or uppercase Portuguese such as `MANHÃ DE SOL` as
  mojibake (#317).
- `validate_fields` accepts international phone numbers such as
  `+49 (0) 30 12345678` and punycode email TLDs (#318).
- `insight_report` issue ids are unique for columns whose names slug the same
  (#329).
- HTML report filter boxes work; the generated script was a JavaScript syntax
  error (#336).
- `stakeholder_summary` and the per-column view no longer describe preserved
  missing values as changes (#337), and no longer claim 100% completeness when
  every column was dropped (#339).
- `export_dbt_tests` quotes YAML scalars that would change type or lose
  characters on load (dates, `0x1F`, trailing newlines), and floats such as
  NaN and infinity load back as floats (#342).
- `OnnxEncoder` no longer fails when several threads trigger the lazy model
  load at once (#340).
- `fd.models.pull` times out stalled downloads and rejects a truncated file
  instead of installing it (#341).
- The FreshCore backend falls back to pandas for configurations its native
  kernels got wrong: `impute="missforest"`, per-column `impute_strategy`,
  outlier detection on float columns holding infinity, and mode imputation of
  nullable boolean columns. Under `fallback_policy="error"` these runs raise
  `FallbackError` (#322, #334, #335).
- The FreshCore adapter reports native duplicate detections and applies
  `duplicate_ratio_action` to native drop counts, as the pandas pipeline does
  (#323, part).
- The Spark engine renames columns without collisions, honours
  `duplicate_keep` and input order when deduplicating, reads float `NaN` as
  null with outlier fences from finite values only, and no longer treats
  interval columns as numeric (#330, #331, #332, #333).
- A malformed plugin proposal or entry point is dropped or skipped with a log
  message instead of crashing the clean or stopping registration (#297).
- Reusing a plugin name within one kind logs a warning naming the replaced
  plugin (#298).
- MissForest fills a column that has no predictor columns once, through its
  fallback, recording one `missforest_fallback` action and one
  `columns_imputed` entry, and no longer needs scikit-learn when every column
  falls back (#324).
- `fd.validate` compares `allowed_values` by type on numeric and boolean
  columns, so `1` matches `1.0` and `true` matches `True` instead of every row
  being a violation. `export_gx_suite` and `export_dbt_tests` emit typed
  numbers and booleans for these sets (#255).
- `explain_clean` reports changed cells, HTML dtypes and narratives for
  integer-labelled columns, MultiIndex labels no longer break `to_dict()` and
  `to_html()`, and `explain_clean` and `infer_roles` accept mixed integer and
  string labels instead of raising `TypeError` (#232, parts 3 and 6).
- The HL7 v2 parser uses the delimiters each MSH segment declares, takes the
  first repetition for single-valued fields while OBX-5 keeps every repetition
  joined with `~`, and decodes `\F\`, `\S\`, `\T\`, `\R\` and `\E\` escapes
  (#261).
- `clean_text`, `validate_fields` and `suggest_join_keys` handle frames with
  duplicate row labels, such as `pd.concat` output, instead of writing one
  row's value to all of them or raising (#231, parts 2-4).
- `validate_fields` compares tz-aware values with naive `min_value` and
  `max_value` bounds, or the reverse, in UTC instead of raising `TypeError`
  (#233, part 3).
- MissForest keeps integer and nullable integer column dtypes instead of
  returning `object` columns. Imputed values are rounded half-to-even, noted in
  the action rationale and flagged by a new `rounded_to_integer` metadata key
  (#263).
- GTFS-ST004 flags only stop_times rows that repeat an earlier `stop_sequence`
  within a trip, so a valid trip whose rows are not sorted no longer raises an
  error (#319).
- `cdc_profile(stale_after=0)` no longer raises `ZeroDivisionError`, and the
  `replay_threshold` docs state that `replay_risk` needs at least one
  duplicate-key row (#326, #328).
- `diff_schema` accepts polars frames (#274), and baseline key-level changes
  treat a `NaN` key as one value and accept mixed integer and string keys, so
  identical frames report no changes (#276).
- Contracts and baselines find integer column labels declared as `0` or `"0"`
  (#232, part 4), and baselines compare tz-aware and naive timestamps in UTC
  with a `drift.timezone_change` warning instead of raising `TypeError` (#233,
  part 5).
- The FreshCore backend falls back to pandas for datetime, timedelta,
  categorical, period and interval columns, integer columns beyond ±2**53, and
  column labels that collide once stringified, instead of changing dtypes or
  overwriting columns. Native results restore integer dtypes and non-string
  labels; integer columns that cannot be restored stay `float64` and are
  recorded in `backend_differences` (#262; #232, part 2).
- The privacy extras (`privacy` and `all`) install on Python 3.9 from wheels.
  On Python 3.9 they cap spaCy below 3.8.8, thinc below 8.3.5 and blis below
  1.2.1; Linux aarch64 still builds thinc and blis from source (#278).
- `gate_manifest` with `output_dir` no longer overwrites the audit file of a
  model that shares its alias with another. Such models write
  `<schema>.<alias>_audit.json`, or `<unique_id>_audit.json` when the schema is
  missing or still not unique (#344).
- `freshdata stream` output keeps the first batch's layout: later CSV batches
  are written under its columns, so a missing flag column is left empty instead
  of shifting values, and later Parquet batches are cast to its schema. Output
  goes to a `.partial` file that is moved into place on success, so a failed
  run leaves no truncated file (#248, part a).
- Replaying a decisions table read back from CSV applies table-level steps
  such as `drop_duplicates`, and `CleaningMemory.to_json` writes `null` instead
  of bare `NaN` or `Infinity` (#309).
- Replaying a merged `LearningProfile` on its training frame no longer reports
  drift or skips text columns, because the merge keeps the parents' source
  schema (#308).
- The GPX and SDMX parsers return their usual invalid-XML warning and empty
  frames when the XML declares an unknown encoding, instead of raising
  `LookupError` (#315).
- `check_k_anonymity` ignores empty combinations of categorical
  quasi-identifiers, so unused categories no longer give
  `smallest_class_size=0` and fail the check, including through
  `clean_enterprise` with `KAnonymityConfig` (#244).
- The `reversible` flag in `fpe` audit metadata follows the mode each cell
  actually used, so cells that took the surrogate fallback are no longer
  reported as reversible (#281, parts 1-2).
- `JsonTokenVault` instances that share a path keep each other's entries:
  writes re-read and merge the file under a lock instead of rewriting it from
  a stale copy, and an empty vault file loads as an empty vault.
  `SqliteTokenVault` works when used from threads other than the one that
  created it (#279).
- `apply_privacy_policy` works on frames with non-string column labels, such
  as integers; report keys stay strings (#232, part 5).
- The `quarantine` privacy-policy action works on nullable integer, boolean
  and categorical columns instead of raising `TypeError`; those columns come
  back as object dtype and missing cells stay missing.

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

## [1.0.1] - 2026-06-15

### Fixed
- **Single-string config fields no longer split into characters.** Passing a
  bare string such as `id_columns="sku_num"` went through `tuple()` and became
  one entry per character, so ID protection, `preserve_columns`,
  `duplicate_subset` and `extra_sentinels` silently matched nothing. These
  fields now accept either one name or a sequence of names.
- An explicit `outlier_action="cap"` / `"remove"` is honored instead of being
  downgraded to `"flag"` under `strategy="balanced"`. This shipped in 1.0.1
  and is described in full under [1.1.0].

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
