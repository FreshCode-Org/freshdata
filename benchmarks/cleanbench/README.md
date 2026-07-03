# CleanBench (Phase-2 skeleton)

A deterministic corruption/repair benchmark for freshdata's semantic layer.
The premise: before an auto-cleaner earns credit for what it fixes, it must
prove what it never breaks.

## Layout

```
benchmarks/cleanbench/
  corruptors.py   # 10 deterministic, seedable corruption functions
  metrics.py      # false-modification / protection / precision-recall-F1
  fixtures/       # CI-safe in-code mini fixtures (tiers T2, T3)
```

## Protocol

1. Build a known-clean frame (`truth`).
2. Corrupt it deterministically (`corrupted = corruptor(truth, seed=...)`).
3. Clean it (`repaired, report = fd.clean(corrupted, **kwargs)`).
4. Score `(truth, corrupted, repaired)` with the metrics.

Everything is seeded and generated in code — a CI failure reproduces
locally byte-for-byte.

## Metrics

| metric | question it answers |
| --- | --- |
| `protected_column_violation_rate` | did any protected column change at all? |
| `false_modification_rate` | how many already-correct cells were changed? |
| `cell_repair_precision` | of the cells changed, how many became the truth? |
| `cell_repair_recall` | of the corrupted cells, how many were restored? |
| `cell_repair_f1` | harmonic mean of the two above |

## Release gates (Phase 2)

| gate | value |
| --- | --- |
| protected-column violation rate | **= 0** (hard, no exceptions) |
| false modification rate (Phase-2 fixtures) | **<= 0.1%** |
| runtime slowdown vs. semantic-off clean | <= 20% *(needs the timing harness; informational until it lands)* |

The first two gates run in CI via `tests/test_cleanbench.py`.

## Tiers

- **T2 — semantic values** (`make_t2_semantic_fixture`): email repairs
  (`a@@b.com`, `a @ b.com`, padding), Indian phone normalization to
  `+91XXXXXXXXXX`, reference-list repairs against allowed status values.
- **T3 — context compliance** (`make_t3_context_fixture`): a protected
  revenue column whose deliberate dirt must **survive**, an age imputation
  threshold (>95%) that must hold fills back, allowed status values, and a
  duplicated `cust_id` that must be *reported* by `fd.validate`, not
  silently repaired.

Later phases add larger corpora, more corruptors, and the timing harness;
this skeleton intentionally stays small enough for every CI run.
