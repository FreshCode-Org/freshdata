# CleanBench full-suite results

Generated: 2026-07-05T07:07:36+00:00
Tracks: T1, T2, T3, T4, T5 | trained calib-v1 used: no
Command: `python -m benchmarks.cleanbench --tracks T1,T2,T3,T4,T5 --report site --reproduce-headline`

## Environment

| key | value |
|---|---|
| freshdata_version | 1.0.0 |
| git_commit | 635abe5f280707ad9aede97460c493aa6523e365 |
| python_version | 3.13.9 |
| platform | macOS-15.5-arm64-arm-64bit-Mach-O |
| processor | arm |
| polars_version | 1.40.0 |
| duckdb_version | 1.5.4 |
| pandas_version | 2.3.3 |
| metrics_version | cleanbench-metrics-v1 |

## T1

| metric | value |
|---|---|
| cell_repair_precision | 1.0000 |
| cell_repair_recall | 1.0000 |
| cell_repair_f1 | 1.0000 |
| false_modification_rate | 0.0000 |
| protected_column_violation_rate | 0.0000 |
| explainability_rubric_score | 1.0000 |

## T2

| metric | value |
|---|---|
| cell_repair_precision | 1.0000 |
| cell_repair_recall | 1.0000 |
| cell_repair_f1 | 1.0000 |
| false_modification_rate | 0.0000 |
| protected_column_violation_rate | 0.0000 |
| confidence_ece | 0.0384 |
| precision_at_conf_95 | 1.0000 |
| coverage_at_precision_99 | 1.0000 |
| n_confidence_pairs | 61 |
| explainability_rubric_score | 1.0000 |

## T3

| metric | value |
|---|---|
| false_modification_rate | 0.0000 |
| protected_column_violation_rate | 0.0000 |
| context_exact_policy_accuracy | 1.0000 |
| slot_f1 | 1.0000 |
| explainability_rubric_score | 1.0000 |

## T4

| metric | value |
|---|---|
| profile_replay_lift | 0.7778 |
| false_modification_rate | 0.0000 |
| false_modification_rate_without_profile | 0.0000 |
| profile_fmr_non_increase | yes |
| drift_block_rate | 1.0000 |
| privacy_leak_count | 0 |
| protected_column_violation_rate | 0.0000 |
| explainability_rubric_score | 1.0000 |

## T5

| metric | value |
|---|---|
| rows | 50000 |
| speed_rows_per_sec | 133115.1000 |
| seconds_default | 0.3756 |
| seconds_with_semantic | 0.5657 |
| semantic_overhead_ratio | 0.5060 |
| peak_rss_bytes | 207749120 |
| peak_rss_delta_bytes | 27836416 |
| peak_rss_delta_semantic_bytes | 638976 |
| protected_column_violation_rate | 0.0000 |
| false_modification_rate | — |
| runtime_slowdown_vs_baseline | -0.0314 |
| memory_overhead_vs_baseline | 0.0149 |

## Baselines

| baseline | status | detail |
|---|---|---|
| pandas | ran | cell_repair_precision=1.0000, cell_repair_recall=1.0000, cell_repair_f1=1.0000, false_modification_rate=0.0000, authored_lines=26, network_call_count=0 |
| pyjanitor | skipped | pyjanitor is not installed; install with `pip install pyjanitor` to run this baseline (the harness skips it otherwise). |
| great_expectations | ran | engine=great_expectations, cells_validated=72, cells_failing=49, cells_repaired=0, manual_fix_cost_cells=49, cell_repair_f1=—, protected_column_violation_rate=—, false_modification_rate=—, network_call_count=0, notes=GE flags dirt but repairs nothing; every failing cell is manual work. Validation success is not repair success. |
| llm_agent | skipped | set FRESHDATA_LLM_BASELINE=1 plus provider env vars to run (never enabled in CI; benchmark-only, isolated from runtime) |

## Release gates

**GATE FAILURES:**
- ECE 0.0384 > 0.03
