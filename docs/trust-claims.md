# Trust claims — evidence map

Every trust-relevant claim FreshData makes, mapped to the thing that proves
it. The three product-defining claims are additionally machine-enforced: the
`CLAIM_REGISTRY` (`benchmarks/cleanbench/reproducibility.py`) pins their
README wording verbatim to named tests, and CI fails if either side drifts.

## Machine-enforced claims (CLAIM_REGISTRY)

| Claim (verbatim README wording) | Evidence |
|---|---|
| protected columns are never modified | `tests/test_semantic_cleaning.py::test_id_columns_protected`, CleanBench `T2.protected_column_violation_rate` |
| nothing happens silently | `tests/test_semantic_cleaning.py::test_assist_records_without_mutating` |
| raw PII never enters the copilot's model context | 4 tests in `tests/test_experimental_ai_copilot.py`, incl. adversarial cases: undeclared string-like columns masked, `category_noise` previews withheld |

## Other README / docs claims

| Claim | Status | Evidence / boundary |
|---|---|---|
| 93% coverage gate enforced in CI | **holds** | `--cov-fail-under=93` in pyproject addopts; CI runs it on every PR (recent runs: 93.5%) |
| Safe defaults (never imputes identifiers, never touches targets, no blind outlier removal) | **holds** | protected-column guard + role inference tests; `strict=True` escalates ambiguity to errors |
| Fully offline; only network call is `fd.models.pull` | **holds** | default test gate is `-m "not online"`; no other network code paths |
| Polars/DuckDB/Spark backends scale beyond pandas | **holds with measured boundary** | see [backends](backends.md): DuckDB native handle peaks at ~40% of eager RSS (measured); `polars-lazy` defers only final materialization — pipeline intermediates still collect eagerly. Reproduce: `python benchmarks/bench_outofcore.py` |
| Native handle kept with `strategy="conservative"` | **corrected** | also requires `fix_dtypes=False`; dtype fixing forces the recorded pandas fallback |
| StreamingCleaner cleans arbitrarily tall data with stable memory | **holds** | bounded reservoirs/counters, RSS benchmark `benchmarks/bench_streaming.py`; dedupe window is recent-window, not global (see [limitations](limitations.md)) |
| CSV exports safe to open in spreadsheets | **holds for review queues by default; opt-in elsewhere** | see [threat model](threat-model.md); `tests/test_csv_formula_sanitize.py` |
| Typed (`py.typed`), mypy-clean | **holds** | `mypy src/freshdata` in CI quality gate (187 files) |
| `fallback_policy="error"` prevents any unrequested pandas materialization | **holds** | enforced *before* the pandas pipeline runs at every backend delegation point; `tests/test_execution/test_fallback_policy.py` |
| Polars subset dedup matches pandas keep semantics byte-for-byte | **holds** | order-preserving `unique(subset=...)`; parity tests incl. nulls/unicode/multi-column in `tests/test_execution/test_fallback_policy.py` (order preservation makes the stage eager, disclosed on the report) |
| Validation never mutates data | **holds** | `tests/test_validation_suite.py::test_validation_never_mutates`; non-pandas inputs record their materialization on `ValidationResult.execution` |
| Entity-resolution accuracy is measured, not asserted | **holds with named method** | committed labelled dataset + `benchmarks/er_results.json` (P/R/F1, FP/FN, false merges per `null_policy` × `mode`); scoring is rule-weighted (no EM/Fellegi–Sunter — no Splink-parity claim) |
| Out-of-core evidence covers 10M+ rows | **holds to 25M** | `src/freshdata/benchmarks/RESULTS.md` (10k–25M, machine + versions + command recorded); 100M/1B size keys exist in the harness but have no committed evidence |

## Benchmark numbers

- The CleanBench table in [benchmarks](benchmarks.md) carries full run
  provenance (commit, platform, library versions) and includes the
  **disclosed miss**: ECE 0.0384 against the ≤ 0.03 target. The perfect
  1.0000 precision/recall rows are real results *on the public CleanBench
  corruption suite* — a suite the engine was developed against; treat them
  as regression floors, not field expectations.
- **Staleness note:** the published table was produced at freshdata 1.0.0.
  Numbers should be regenerated (same commands, pinned in the doc) for the
  next release, or the table header should state its version explicitly.
- Throughput numbers ("modern laptop") are order-of-magnitude guidance
  backed by `tests/fixtures/perf/baselines.json`, not a promise; the
  reproducible path is running `benchmarks/bench_report.py` yourself.

## Known gaps this page owns

- ECE 0.0384 > 0.03 target (calibration artifact that closes it exists in
  the dev training pipeline but is not published).
- Numeric quasi-identifier residue in copilot samples (see threat model).
- Polars pipeline collects intermediates eagerly (out-of-core boundary
  above).
