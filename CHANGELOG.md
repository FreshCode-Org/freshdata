# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

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
