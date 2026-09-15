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
- IQR/z-score outlier clipping or flagging, including the pandas zero-IQR
  fallback to mean-absolute-deviation fences
- simple per-column profile metadata

FreshCore falls back for semantic cleaning, context/policy protection, cleaning
memory, domain packs, contracts, non-default indexes, duplicate subsets,
aggregate/drop duplicate modes, model-based outliers, constant-column dropping,
memory downcasting, and the balanced/aggressive decision engine. It also falls
back for `impute="missforest"`, per-column `impute_strategy`, outlier handling
on float columns holding `±inf`, and mode/auto imputation of nullable boolean
columns with missing values.

The same two gaps can open inside a native run. `fix_dtypes` casts text columns
before imputation and outlier handling, so a `"yes"`/`"no"` column with missing
values becomes a boolean column the kernels do not impute. A numeric text
column holding `"inf"` becomes a float column holding `±inf`, which leaves the
outlier fences undefined. The input frame shows neither, so the adapter checks
the returned column dtypes. When a cast column hits a gap, it reruns the frame
on pandas and records a fallback event naming the column
(`fallback_step="impute"` or `"outliers"`). Under `fallback_policy="error"` it
raises `FallbackError` instead. `fd.plan()` checks only the input, so it
cannot predict these fallbacks, and the discarded native run still costs time.
Frames without such casts stay on the native path.

With `drop_duplicates=False` (the default), the native module counts full-row
duplicates at the same stage as the pandas step: after string cleaning,
empty-row removal and casts, and before imputation and outliers. It returns
the count as `duplicates_detected` and records a `detect_duplicates` stage
timing. The adapter reports it like pandas: a "detected N duplicate row(s)"
action, a warning above `duplicate_threshold`, and `DuplicateRatioError` under
`duplicate_ratio_action="error"`. Native modules built before this count
existed don't report `duplicates_detected`; with those, detection is skipped,
and under `duplicate_ratio_action="error"` the adapter falls back to pandas.

The native arrays carry only float, bool and string values, so FreshCore also
falls back for datetime, timedelta, categorical, period and interval columns,
for integer columns holding values beyond ±2\*\*53 (float64 cannot represent
them exactly), and for column labels that collide once stringified (such as
`1` and `"1"`). Integer columns (`int64`, nullable `Int64`, and other widths)
are cast back to their input dtype when every returned value is integral and
in range; otherwise they come back as `float64` and the change is recorded in
`report.backend_differences`. Non-string column labels such as `0` and `1`
come back unchanged.

## Building and testing

The CI `freshcore-native` job runs the same steps:

```bash
pip install -e ".[dev,freshcore]"
cargo test --manifest-path crates/freshcore/Cargo.toml
maturin develop --manifest-path crates/freshcore/Cargo.toml --features extension-module
pytest tests/test_execution -k freshcore
```

`tests/test_execution/test_freshcore_native_parity.py` runs the real extension
against the pandas reference and is skipped when `freshdata_freshcore` is not
installed. The other FreshCore tests use a fake native module.

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
