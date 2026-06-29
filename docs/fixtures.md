---
title: Benchmark fixtures
description: >-
  Schemas, defect manifests and gold labels for the freshdata benchmark fixture
  library, and how to contribute a new fixture.
keywords: data cleaning benchmark fixtures, synthetic data, defect injection
---

# Benchmark fixtures

The benchmark fixture library (`benchmarks/fixtures/`) provides six
deterministic, seed-controlled generators shaped like real enterprise tables.
Each injects only defects within FreshData's declared repair scope and documents
them so the harness can score repair fidelity against known ground truth.

Every fixture module exposes:

- `generate(n_rows, seed=42, defect_rate=None) -> pd.DataFrame` — deterministic;
  the same `(n_rows, seed, defect_rate)` always yields identical data. With
  `defect_rate=None` each defect family is injected at its documented base rate;
  with a float in `[0, 1]` every family is injected at that uniform rate (the
  knob the trust-monotonicity metric sweeps).
- `GOLD_LABELS: dict[str, dict]` — per column: `role`, `expected_dtype`,
  `fill_action`, `preserve`, and an optional `reference` set.
- `DEFECT_MANIFEST: list[dict]` — one record per injected defect family.
- `ID_COLUMNS`, `TARGET_COLUMN`, `TEXT_COLUMNS`, `SCALE_VARIANTS`, `N_COLS`.

The `gold` fixture additionally returns a `GoldBundle` (dirty/clean frames and
preservation/repair/false-repair masks) instead of a bare DataFrame.

## Fixture schemas

| fixture | cols | id column(s) | protected (never mutated) | scale variants |
|---|--:|---|---|---|
| **crm** | 40 | `customer_id` | id, `full_name`/`email`/`phone` (text) | 10k / 100k / 1M / 5M |
| **finance** | 60 | `transaction_id`, `account_code` | ids, `gl_account`/`cost_center` (text) | 10k / 500k / 5M / 25M |
| **event_log** | 25 | `event_id`, `entity_key` | ids | 10k / 1M / 10M / 50M |
| **wide_schema** | 100–5000 | `row_uuid` | id, `y_target` (target), `text_*` | 1k / 10k / 100k rows × 100/500/1k/5k cols |
| **provenance** | 18 | `record_id`, `invoice_number` | ids, `vendor_name` (text) | 1k / 10k / 100k |
| **gold** | 7 | `gid` | `gid`, `gtarget` (target), `gtext` (text) | 10k |

### Defect families (in scope only)

- **Sentinel normalization** — recognised tokens (`N/A`, `null`, `--`, `n.a.`)
  → missing. `999` is *excluded*: FreshData treats it as a number, so it is not
  an in-scope sentinel.
- **Dtype coercion** — numeric strings (incl. `$` and thousands commas) → float;
  ISO date strings → datetime64. Foreign-currency words (`EUR 500`) and
  accounting negatives (`(1234.56)`) are out of generic scope (finance domain
  pack territory) and intentionally not relied on.
- **Median fill** — low-missingness numeric columns; skewed distributions make
  the engine's robust median the deterministic choice.
- **Categorical normalization** — sentinels → mode (when a dominant mode ≥ 50%
  exists) or the canonical `Missing`/`Unknown` marker.
- **Duplicate removal** — exact full-row duplicates.
- **Reference-set violations / CDC / provenance** — values outside an approved
  set, late arrivals, replay, out-of-order versions, low parser confidence: these
  are *flagged*, not silently rewritten.
- **Preservation checks** — null ids, missing targets, missing free text: must be
  flagged and **never** filled or mutated.

## DEFECT_MANIFEST structure

Each entry (built from `fixtures._common.Defect`) is a JSON-friendly dict:

```python
{
  "id": "crm-ltv-missing",          # stable family id
  "column": "lifetime_value",        # column, "a|b", "prefix_*", or "*" (row-level)
  "defect_type": "missing_numeric",  # human-readable defect kind
  "rate": 0.06,                       # base injection rate (defect_rate=None)
  "in_scope_repair": "median_fill",   # the repair the harness expects
  "preservation": false,              # True => correct behaviour is "do not change"
  "note": ""                          # optional scope caveat
}
```

The harness expands `column` patterns (`"score_*"`, `"a|b"`, `"*"`) and maps
`in_scope_repair` to an observable post-condition for the family-level repair
fidelity metric. The `test_fixtures.py` suite asserts the actual injected counts
match each manifest `rate` within ±10%.

## GOLD_LABELS structure

```python
{
  "lifetime_value": {
    "role": "numeric",            # id|target|text|numeric|categorical|datetime|bool
    "expected_dtype": "float64",  # dtype after cleaning
    "fill_action": "median_fill", # the expected action
    "preserve": false,            # True for id/target/text invariants
    "reference": ["US", "CA"]     # optional approved set (categoricals)
  }
}
```

For `wide_schema`, labels depend on the column count, so use
`wide_schema.gold_labels(n_cols)`; the module-level `GOLD_LABELS` is the
100-column default.

## The gold fixture

`gold.generate(n_rows=10_000, seed=42, defect_rate=None)` returns a `GoldBundle`:

| field | shape | meaning |
|---|---|---|
| `dirty_df` | (n + dupes, 7) | input with injected defects |
| `clean_df` | (n, 7) | analytically derived expected output (deduped, repaired) |
| `preservation_mask` | (n, 7) | True where a protected cell must be byte-identical |
| `repair_mask` | (n, 7) | True where a cell must change to a known value |
| `false_repair_traps` | (n, 7) | True where changing the cell is a failure |

`clean_df` is derived from first principles (independent oracle), never by
running the library under test, so the repair-fidelity and false-repair metrics
are grounded in known-correct ground truth.

## Contributing a new fixture

1. Add `benchmarks/fixtures/<name>.py` using the `_common.py` helpers; expose
   `generate`, `GOLD_LABELS`, `DEFECT_MANIFEST`, and the role/scale constants.
2. Inject only in-scope defects (HARD CONSTRAINT 5). Verify against real
   `fd.clean` behaviour — encode what the library actually does, don't assume.
3. Register it in `fixtures/__init__.py` and add a default size in
   `bench.py`'s `DEFAULT_SIZES`.
4. Add a determinism + manifest-count case to `tests/benchmark/test_fixtures.py`.
