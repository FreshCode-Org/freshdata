# freshdata

**Dry-run data repair plans for Polars and pandas—compiled into native expressions without silent mutations.**

[![PyPI version](https://img.shields.io/pypi/v/freshdata.svg)](https://pypi.org/project/freshdata/)
[![Python versions](https://img.shields.io/pypi/pyversions/freshdata.svg)](https://pypi.org/project/freshdata/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/FreshCode-Org/freshdata/actions/workflows/ci.yml/badge.svg)](https://github.com/FreshCode-Org/freshdata/actions)

Most automated data-cleaning libraries fail in production for three reasons:
1. **Silent corruption:** One-call black boxes impute missing target labels or clip primary keys without warning.
2. **Broken lazy execution:** Wrappers claim Polars support but silently convert `LazyFrame` inputs to pandas or call `.unique(maintain_order=True)`, blowing up memory on multi-gigabyte scans.
3. **Zero diffability:** You cannot review or version-control what a heuristic cleaner is about to do before it mutates your data.

`freshdata` separates **planning** from **execution**:
* **`fd.suggest_plan(lf)`** runs a single streaming aggregation pass over a Polars `LazyFrame` (or pandas `DataFrame`), protects ID and target columns automatically, and outputs an inspectable, diffable `CleanPlan` containing precomputed scalar thresholds.
* **`fd.apply_plan(lf, plan)`** compiles those approved actions into pure, fused Polars expressions (`pl.Expr`)—requiring **zero `.collect()` calls** and **zero Python UDFs**. A `LazyFrame` in stays a `LazyFrame` out.

---

## Install

```bash
pip install freshdata
```

To install the AST codemod tool for upgrading legacy `2.x` scripts:

```bash
pip install "freshdata[migrate]"
```

---

## 30-Second Quickstart

```python
import polars as pl
import freshdata as fd

# 1. Point to a lazy Parquet scan (no rows loaded into memory)
lf = pl.scan_parquet("s3://bucket/raw_customer_events.parquet")

# 2. Define explicit column protections and rules
config = fd.CleanConfig(
    id_columns=("customer_id",),
    target_column="churned",
    default_impute="auto",      # median for numeric, mode for string
    default_outliers="clip",    # precomputed IQR clipping
    drop_duplicates=True,       # streaming-safe unique(maintain_order=False)
    rules={
        "signup_age": fd.ColumnRule(impute="median", outliers="clip"),
        "invoice_total": fd.ColumnRule(outliers="flag"),
    },
)

# 3. Generate a dry-run repair plan (single 1-row streaming summary query)
plan = fd.suggest_plan(lf, config=config)

for action in plan.actions:
    print(f"[{action.stage}] {action.column}: {action.operation} ({action.rationale})")
# [strings] segment: strip_whitespace (Strip leading and trailing whitespace)
# [missing] signup_age: fill_null (Impute 412 nulls via median=34.0)
# [outliers] signup_age: clip_iqr (IQR bounds [18, 72] (factor=1.5))
# [missing] churned: preserve (Protected role='target' or conservative strategy)

# 4. Compile the approved plan onto the LazyFrame (zero materialization)
cleaned_lf, report = fd.apply_plan(lf, plan, return_report=True)

assert isinstance(cleaned_lf, pl.LazyFrame)
assert "PYTHON_UDF" not in cleaned_lf.explain()

# 5. Materialize only when your downstream pipeline is ready
df = cleaned_lf.collect(engine="streaming")
```

---

## Diffing Repair Plans Before Execution

Never guess how a configuration change alters your dataset. Compare two `CleanPlan` artifacts directly in code or CI:

```python
conservative = fd.suggest_plan(
    lf, fd.CleanConfig(strategy="conservative", default_outliers="preserve")
)
aggressive = fd.suggest_plan(
    lf, fd.CleanConfig(strategy="aggressive", default_outliers="clip")
)

for (col, stage), (before, after) in conservative.diff(aggressive).items():
    before_op = before.operation if before else "none"
    after_op = after.operation if after else "none"
    print(f"{col} ({stage}): {before_op} -> {after_op}")
```

---

## The Complete API (12 Exports)

`freshdata` v3.0 exposes exactly 12 public symbols. There are no hidden submodules, natural-language rule compilers, or domain-specific bloat.

| Export | Type | Purpose |
| :--- | :--- | :--- |
| `fd.profile` | Function | Read-only column null, cardinality, and quantile summary |
| `fd.suggest_plan` | Function | Single-pass streaming planner returning a dry-run `CleanPlan` |
| `fd.apply_plan` | Function | Compiles a `CleanPlan` into native expressions without materializing |
| `fd.clean` | Function | Shorthand for `apply_plan(df, suggest_plan(df, config))` |
| `fd.Cleaner` | Class | Reusable stateful wrapper around `CleanConfig` for batch pipelines |
| `fd.CleanConfig` | Dataclass | Frozen pipeline specification with strict backend enforcement |
| `fd.ColumnRule` | Dataclass | Explicit per-column role, imputation, and outlier override |
| `fd.CleanPlan` | Dataclass | Serializable, diffable tuple of planned `ActionRecord` items |
| `fd.ActionRecord` | Dataclass | Single column operation with baked scalar parameters and rationale |
| `fd.CleanReport` | Dataclass | Post-application audit record of renamed/dropped columns and actions |
| `fd.ProfileReport` | Dataclass | Dataset-level read-only diagnostic container |
| `fd.ColumnProfile` | Dataclass | Per-column statistical summary emitted by `fd.profile()` |

---

## Where `freshdata` Fits

`freshdata` is a **planning and repair engine**, not a schema validation gate or orchestration framework. Use it alongside focused tools in your stack:

| Tool | Primary Job | Mutates Data? | Lazy Polars Native? |
| :--- | :--- | :--- | :--- |
| **Pandera / Great Expectations** | Assert schema contracts and fail bad batches | No | Partial |
| **pyjanitor** | Imperative method-chaining helpers | Yes (manual) | No (pandas-first) |
| **freshdata** | Propose, diff, and compile automated repair plans | Yes (via `CleanPlan`) | Yes (`pl.Expr`) |

---

## Guaranteeing Zero-Fallback Execution

By default, `CleanConfig(strict_backend=True)` refuses to run any operation that cannot be expressed natively on your input frame:
* **No silent `.to_pandas()` conversions:** Passing a `pl.LazyFrame` or `pl.DataFrame` executes 100% inside Polars.
* **No Python row loops:** String normalization, null imputation, and IQR clipping compile down to `.str.strip_chars()`, `.fill_null()`, and `.clip()`.
* **Streaming-compatible deduplication:** When `drop_duplicates=True`, `freshdata` emits `.unique(maintain_order=False)` so Polars streaming pipelines never stall holding row order in RAM.

---

## Upgrading from v2.x

If you are upgrading from `freshdata` 2.x (`freshdata-cleaner`), run the included `libcst` codemod to rewrite legacy wrappers (`clean_csv`, `fill_missing`, `compare_plans`, etc.) into the v3.0 core API:

```bash
# Preview unified diff without modifying files
freshdata-migrate src/ tests/ --check --diff

# Apply deterministic AST rewrites in-place
freshdata-migrate src/ tests/
```

---

## Contributing & Issues

Bug reports, benchmark reproductions, and pull requests are welcome.
1. Open an issue with a minimal reproducible Polars or pandas snippet.
2. Run the test suite locally via `pytest -q`.
3. Ensure any changes to `polars_engine.py` pass the lazy materialization poison test (`tests/test_polars_engine.py`).

## License

MIT License. See [LICENSE](LICENSE) for details.
