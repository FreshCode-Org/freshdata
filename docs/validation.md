# Validation

**Limitations first:**

- Validation executes on **pandas**. Non-pandas inputs (Polars, Arrow, DuckDB
  relations) are materialized, and that materialization is recorded on
  `ValidationResult.execution` — it is never silent, but there is no native
  Polars/DuckDB/Spark validation execution today.
- `mostly=` tolerance applies to value-level checks (allowed values, ranges,
  regex, lengths); structural checks (missing column, dtype, uniqueness) are
  binary.
- Datetime bounds coerce with `pd.to_datetime(errors="coerce")`; values that
  fail to parse are *skipped* by the bound check (they surface through dtype /
  regex rules instead).
- There is no suite store or checkpoint scheduler — suites are plain JSON
  files you version in git and run via Python or the CLI.

Validation **never mutates data**. Every surface below is read-only; repairs
are a separate, explicit step (`fd.clean`, `fd.apply_plan`).

## Which validation surface do I want?

FreshData has four, each for a different job:

| Surface | Use when | Entry point |
|---|---|---|
| **ValidationSuite** | You want declarative, versioned pass/fail rules (Great-Expectations-shaped) with CI exit codes | `fd.validate(df, suite=...)` |
| **DataContract + baseline** | You want *drift* monitoring against a historical baseline (PSI/KS, schema changes) plus contract rules | `fd.compare_to_baseline`, `fd.monitor_contract`, `fd.diff_schema` |
| **Context policies** | You want plain-English rules compiled into protections that also govern cleaning | `fd.validate(df, context="...")` |
| **Domain packs** | Your data is a known domain (finance ledgers, FHIR, GTFS…) with canonical field rules | `fd.clean(df, domain=...)` |

ValidationSuite and DataContract share **one check engine** — a suite compiles
to a contract (`suite.to_contract()`), and an existing contract migrates with
`fd.ValidationSuite.from_contract(contract)`. There are not two competing
rule systems.

## ValidationSuite quickstart

```python
import freshdata as fd

suite = fd.ValidationSuite(
    name="customers",
    rules=[
        fd.ColumnRule("customer_id", dtype="string", nullable=False, unique=True),
        fd.ColumnRule("age", min_value=0, max_value=120),
        # tolerate up to 2% bad emails: violations below that are warnings
        fd.ColumnRule("email", regex=r"[^@]+@[^@]+\.[^@]+", mostly=0.98),
        fd.ColumnRule("signup", min_datetime="2015-01-01", max_datetime="2030-01-01"),
        fd.ColumnRule("plan", allowed_values=("free", "pro", "enterprise")),
        fd.ColumnRule("country", min_length=2, max_length=2),
    ],
    cross_column=[fd.CrossColumnRule("signup", "<=", "last_seen")],
    compound_unique=(("region", "external_ref"),),
    min_rows=1,
)

result = fd.validate(df, suite=suite)
result.passed          # bool
result.findings        # QualityFinding list (same shape as everything else)
result.report          # full DriftReport with per-check detail
result.raise_if_failed()   # raises fd.ValidationError — CI-friendly
```

Rule vocabulary on `ColumnRule` (= `ColumnContract`): `dtype` (family match by
default, `dtype_exact=True` for raw dtype strings), `nullable`, `required`,
`unique`, `allowed_values`, `min_value`/`max_value` (numeric),
`min_datetime`/`max_datetime`, `regex`, `min_length`/`max_length`,
`max_missing_ratio`, `max_cardinality`, `semantic_type`, and `mostly` (the
failure-tolerance knob). Dataset-level: `min_rows`, `max_rows`,
`compound_unique`, `strict_columns` (exact schema match), `trust_score_min`.

### Severity and thresholds

Findings carry `status` (`passed` / `warned` / `failed`) and `level`. With the
default `mostly=1.0`, any violation fails. With `mostly=0.95`, up to 5% of
non-null values may violate a value-level rule — the finding is then a
*warning* and the suite still passes. `violation_ratio` and `n_violations`
are always in the finding details, with an offending-value sample.

### Suites are artifacts

```python
suite.save("customers_suite.json")       # versioned, schema-tagged JSON
suite = fd.ValidationSuite.load("customers_suite.json")
result.to_json()                          # serializable result, too
```

### CLI + CI

```console
$ freshdata validate data.csv --suite customers_suite.json --json result.json
freshdata validate: FAIL — 2 error(s), 1 warning(s) against suite 'customers'
  [failed] contract.min_value: 'age' minimum -3.0 below 0 (4/9821 = 0.04%)
$ echo $?
1
```

Exit codes: `0` pass, `1` failed findings, `2` usage/load error. `--contract`
accepts an existing DataContract JSON directly.

### Custom validators

Read-only plugin validators (`fd.register_validator` or the
`freshdata.validators` entry point) run inside `fd.validate(context=/policy=)`
— see [Writing plugins](plugins.md). For suite-shaped custom checks, add the
rule to the suite after computing it, or open an issue: we prefer growing the
shared vocabulary over per-user forks.

## Migrating from raw DataContract

```python
suite = fd.ValidationSuite.from_contract(existing_contract)   # 1:1
report = fd.enforce_contract(df, existing_contract)           # or stay on contracts
```

`fd.enforce_contract` is the contract-only check (no baseline, no drift) —
what `compare_to_baseline` runs *minus* the statistical drift layer.

## Comparison with Great Expectations

See [Comparison](comparison.md) for the honest table. Summary: GX has a much
larger expectation vocabulary and a suite/checkpoint/data-docs ecosystem;
FreshData covers the common core (nullability, uniqueness incl. compound,
enums, ranges incl. datetime, regex, lengths, row counts, cross-column
comparisons, `mostly` tolerance, severity levels, serializable results, CLI
exit codes) with one import and no context/store setup, plus an exporter
(`fd.export_gx_suite`) when you need to hand findings to a GX deployment.
