# Architecture

A map for contributors: what lives where in `src/freshdata`, how a call flows
through the code, and where to add behavior. Read it once before your first
change. Module docstrings and tests hold the detail.

For user-facing docs, see the
[documentation site](https://freshcode-org.github.io/freshdata/).

## The one-sentence version

`fd.suggest_plan(df)` builds an immutable, diffable `CleanPlan` of
`ActionRecord`s. `fd.apply_plan(df, plan)` executes that plan with the backend
that matches the frame type. `fd.clean(df)` runs both in sequence.

## Layout

```
src/freshdata/
├── __init__.py        public API re-exports
├── _core/
│   ├── models.py      frozen dataclasses: ColumnRule, CleanConfig, ActionRecord,
│   │                  CleanPlan (+ diff), ColumnProfile, ProfileReport, CleanReport
│   ├── engine.py      public functions + Cleaner; config defaults; report assembly
│   ├── backends.py    PandasBackend, PolarsBackend, resolve_backend()
│   └── stages.py      pandas planning stages + shared helpers
│                      (_normalize_name, normalize_config, _is_protected)
└── cli/migrate.py     freshdata-migrate: libcst codemod for 2.x -> 3.x scripts
```

Dependency direction is strictly one way:
`engine -> backends -> stages -> models`. Inside `_core`, no module imports `engine`.

## Data flow

```
fd.clean(df, cfg)
  ├─ suggest_plan(df, cfg)
  │    └─ resolve_backend(df)            dispatch on frame type
  │         ├─ PandasBackend.suggest_plan
  │         │    normalize_config(cfg)
  │         │    for stage in (column_names, strings, dtypes,
  │         │                  duplicates, missing, outliers):
  │         │        df, actions = stage(df, cfg)   eager; each stage sees
  │         │                                       the previous stage's output
  │         └─ PolarsBackend.suggest_plan
  │              schema only + at most ONE 1-row aggregate query
  │              (IQR quantiles for clip columns)
  │    -> CleanPlan(actions)
  └─ apply_plan(df, plan, cfg)
       ├─ PandasBackend.apply_plan   renames, then actions in plan order (eager)
       └─ PolarsBackend.apply_plan   renames -> one fused with_columns(...) ->
                                     optional unique(); never collects
```

## Invariants

1. **Plans are data.** `CleanPlan` and `ActionRecord` are frozen. A plan can be
   serialized (`to_dict`), diffed (`diff`), and edited before it is applied.
2. **Lazy stays lazy.** `PolarsBackend.apply_plan` must not call `.collect()`,
   `.to_pandas()`, or Python UDFs. `tests/test_backend_honesty.py` enforces this.
3. **Protection lives in one place.** `stages._is_protected` decides which
   columns the missing and outlier logic must skip. Do not copy the predicate.
4. **pandas is required. Polars is optional.** `resolve_backend` imports Polars
   only for Polars frames.

## Where to add behavior

| Change | Edit |
|---|---|
| New pandas planning rule | a `stage_*` function in `stages.py`, then add it to the tuple in `PandasBackend.suggest_plan` |
| Execute a new action | both `PandasBackend.apply_plan` and `PolarsBackend.apply_plan` |
| New config option | `models.py`, then every backend that must honor it |
| New frame type | a backend class with `profile` / `suggest_plan` / `apply_plan`, plus a branch in `resolve_backend` |

## Known issues

These are real defects in the current behavior. They are recorded here, not
fixed, because each fix changes output.

- **pandas: plan and apply disagree on strings.** `stage_strings` collapses
  inner whitespace and nulls `"nan"`/`"none"`/`"null"` tokens. The
  `strip_strings` executor in `apply_plan` only strips. Plan statistics
  (mode values, medians, `rows_affected`) come from data that `apply_plan`
  does not produce.
- **pandas: `apply_plan` nulls non-string cells in object columns.** A mixed
  column `[1, " a "]` becomes `[None, "a"]`.
- **pandas: protection is partial.** `stage_strings` and `stage_dtypes` modify
  ID, target, and `protected=True` columns. Polars skips them.
- **Backends resolve rules differently.** pandas falls back to the defaults
  per field (`rule.impute or default_impute`). Polars falls back only when the
  column has no rule at all. `ColumnRule(outliers="clip")` therefore disables
  default imputation on Polars only.
- **Polars ignores some actions without warning.** `drop_nulls`,
  `drop_outliers`, `parse_currency`, and `impute_mode` are not executed.
- **Accepted but not implemented:** `ColumnRule.role`, `impute="ffill"`,
  `outliers="flag"`, `CleanConfig.strategy`, `CleanConfig.strict_backend`.
- **Polars `LazyFrame` profiles report zeros** for row, null, and unique
  counts instead of "unknown".
- **`rows_affected` overcounts** for `normalize_name`, `strip_strings`,
  `parse_currency`, and `clip_iqr` (it reports the row count, not the cells
  changed). Polars reports 0 for every action.

## Performance notes

- The pandas path is eager. `clean()` transforms the frame twice (once to plan,
  once to apply), and every stage copies the frame. For large data, use a
  Polars `LazyFrame`.
- The pandas string stages run one Python call per cell. `_clean_str` is the
  hottest function in `clean()`. Keep it allocation-light.
