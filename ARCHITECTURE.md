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
3. **One definition per decision.** `stages._is_protected` decides which
   columns no stage may touch. `stages._impute_strategy` and
   `stages._outlier_strategy` resolve rules against defaults. `stages._strip_str`
   is the only pandas string executor. Do not copy these.
4. **Every action runs everywhere or fails loudly.** Both executors handle the
   same action set. An action a backend cannot run raises `ValueError` when
   `strict_backend=True` (the default), and warns and skips otherwise.
5. **pandas is required. Polars is optional.** `resolve_backend` imports Polars
   only for Polars frames.

## Where to add behavior

| Change | Edit |
|---|---|
| New pandas planning rule | a `stage_*` function in `stages.py`, then add it to the tuple in `PandasBackend.suggest_plan` |
| Execute a new action | both `PandasBackend.apply_plan` and `PolarsBackend.apply_plan`, plus a case in `test_every_pandas_action_runs_identically_on_polars` |
| New config option | `models.py` (including `__post_init__` validation), then every backend that must honor it |
| New frame type | a backend class with `profile` / `suggest_plan` / `apply_plan`, plus a branch in `resolve_backend` |

## Known backend differences

Both backends run the same actions. These differences remain by design or
because closing them costs streaming-scale performance.

- **Order of row operations.** pandas runs actions in plan order, so it
  deduplicates before it imputes, and a row dropped for one column changes the
  statistics of the next. Polars computes each column's statistics on the input
  frame, runs row filters after the fused `with_columns`, and deduplicates last.
- **Statistics in plans.** pandas plans carry imputation values. Polars plans
  compute them at apply time, inside the lazy query.
- **Empty strings.** pandas `strip_strings` turns `""` into a missing value.
  Polars keeps `""`.
- **Currency detection.** Only the pandas planner detects currency columns
  (`fix_dtypes`). Both executors run a `parse_currency` action. Polars always
  returns `Float64`, and pandas may return `int64`.
- **Counts.** Polars does not count `drop_duplicates` in `rows_affected`, because
  that needs a full-row hash at plan time. `CleanReport` row counts are 0 for a
  `LazyFrame`, because `apply_plan` never collects.
- **Not detected:** two raw column names that normalize to the same name.

## Performance notes

- The pandas path is eager. `clean()` transforms the frame twice (once to plan,
  once to apply), and every stage copies the frame. For large data, use a
  Polars `LazyFrame`.
- The pandas string stages run one Python call per cell. `_strip_str` is the
  hottest function in `clean()`. Keep it allocation-light.
- The Polars planner runs at most one 1-row aggregate query. `profile()` also
  runs one query, so Polars evaluates all columns in parallel.
