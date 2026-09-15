# FreshCore architecture

FreshCore is FreshData's optional native backend for cleaning-first workloads.
It is designed to plug into the existing execution layer rather than replace the
public API or the pandas reference implementation.

## Integration model

- Users opt in with `fd.clean(df, engine="freshcore")`.
- `CleanConfig` still controls what cleaning is requested.
- `EngineConfig` still controls how execution is routed.
- The default `fd.clean(df)` path remains unchanged.
- Unsupported work delegates to the pandas reference pipeline and is recorded in
  `CleanReport.fallback_events`.

The Python adapter lives at `freshdata.execution.backends._freshcore`. It
materializes pandas-compatible inputs, sends compact columns and plan parameters
to the Rust module, then maps native output back into a pandas DataFrame and the
standard `CleanReport` audit contract.

## Native engine

The Rust crate is in `crates/freshcore` and exposes the `freshdata_freshcore`
PyO3 module. Its internal model is deliberately smaller than a general-purpose
DataFrame:

- typed nullable arrays (`Float`, `Bool`, `Utf8`)
- separate validity/null representation through `Option<T>`
- a compact physical cleaning plan
- native kernels for strings, missing values, casts, duplicates, outliers, and
  simple profiles
- operation-level timing records for benchmarks

FreshCore does not depend on pandas, Polars, DuckDB, Spark, Dask, Modin, Vaex,
or any existing DataFrame execution engine.

## V1 support boundary

FreshCore v1 runs native kernels for conservative, deterministic cleaning:

- column name normalization
- whitespace trimming
- optional text case normalization via `string_case=None|"lower"|"upper"`
- sentinel-to-missing normalization
- empty row/column drops
- full-row duplicate detection with `duplicate_keep="first"` or `"last"`
- boolean and numeric string casts where safe
- mean/median/mode imputation
- IQR/z-score outlier clipping or flagging
- simple per-column profile metadata

FreshCore falls back for semantic cleaning, context/policy protection, cleaning
memory, domain packs, contracts, non-default indexes, duplicate subsets,
aggregate/drop duplicate modes, model-based outliers, constant-column dropping,
memory downcasting, and the balanced/aggressive decision engine. It also falls
back for `impute="missforest"`, per-column `impute_strategy`, outlier handling
on float columns holding `±inf`, and mode/auto imputation of nullable boolean
columns with missing values. Detection-only dedup under
`duplicate_ratio_action="error"` falls back unless the native module reports
`duplicates_detected`.

The native arrays carry only float, bool and string values, so FreshCore also
falls back for datetime, timedelta, categorical, period and interval columns,
for integer columns holding values beyond ±2\*\*53 (float64 cannot represent
them exactly), and for column labels that collide once stringified (such as
`1` and `"1"`). Integer columns (`int64`, nullable `Int64`, and other widths)
are cast back to their input dtype when every returned value is integral and
in range; otherwise they come back as `float64` and the change is recorded in
`report.backend_differences`. Non-string column labels such as `0` and `1`
come back unchanged.

## Benchmarking

Build the native module before benchmarking:

```bash
pip install -e ".[dev,freshcore]"
maturin develop --manifest-path crates/freshcore/Cargo.toml --features extension-module
python benchmarks/bench_freshcore.py --rows 10000 100000 1000000 --workload full
```

The benchmark compares:

- a handwritten pandas baseline
- FreshData's pandas reference path
- FreshCore through `engine="freshcore"`

The output includes timing, peak memory, fallback events, parity shape checks,
and FreshCore stage timings when the native module is installed.
