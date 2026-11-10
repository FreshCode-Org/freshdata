# Freshdata Core Architectural Pivot (v3.0.0) Specification

- **Target Package:** `freshdata` (PyPI: `freshdata-cleaner` -> `freshdata`)
- **Status:** Draft / Approved
- **Branch:** `refactor/core-pivot-v3`
- **Date:** 2026-11-10

---

## 1. Context & Motivation

`FreshCode-Org/freshdata` currently suffers from acute scope bloat: 159 exports, 16 pipeline stages, 61 configuration options, and dozens of unfinished enterprise compliance and format-parsing modules (21 CFR Part 11, SOX 404, HIPAA, HL7, EDIFACT, PII vaults). Despite advertising out-of-core streaming and DuckDB execution, Polars frames are silently converted back into pandas in memory (`pd_in = df.to_pandas()`), causing memory ballooning and blocking true out-of-core streaming.

This specification details the complete architectural cut:
1. Amputating ~80% of secondary modules to isolate a clean 12-export public API.
2. Replacing the 16-stage pipeline and 61-option config with a frozen, deterministic 6-stage pipeline.
3. Implementing native, zero-fallback execution across pandas and Polars (`DataFrame` and `LazyFrame`).
4. Establishing test coverage >= 90% on the core engine running under 3 seconds.

---

## 2. Target Core API (12 Public Exports)

The public surface area of `freshdata` will consist strictly of the following 12 symbols:

```python
# src/freshdata/__init__.py
from freshdata._core import (
    ActionRecord,
    CleanConfig,
    Cleaner,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
    apply_plan,
    clean,
    profile,
    suggest_plan,
)

__all__ = [
    "ActionRecord",
    "CleanConfig",
    "Cleaner",
    "CleanPlan",
    "CleanReport",
    "ColumnProfile",
    "ColumnRule",
    "ProfileReport",
    "apply_plan",
    "clean",
    "profile",
    "suggest_plan",
]
```

### Export Classification

| Category | Surviving Exports | Primary Role |
| :--- | :--- | :--- |
| **Workflow Functions** | `profile`, `suggest_plan`, `apply_plan`, `clean` | Inspect, plan, and execute data repairs |
| **Configuration Classes** | `Cleaner`, `CleanConfig`, `ColumnRule` | Define deterministic column-level constraints |
| **Plan Artifacts** | `CleanPlan`, `ActionRecord` | Diffable, serializable dry-run action sequences |
| **Audit Artifacts** | `CleanReport`, `ProfileReport`, `ColumnProfile` | Structured telemetry and data inspection |

---

## 3. Subsystem Amputation

All non-core subsystems are purged from git:

* `src/freshdata/compliance/`: 21 CFR Part 11, SOX 404, GDPR 30, HIPAA Safe Harbor stubs.
* `src/freshdata/enterprise/`: PII vaults, entity resolution, quality ledgers.
* `src/freshdata/parsers/`: EDIFACT, HL7, FHIR, domain-specific text parsers.
* `src/freshdata/domains/`: Vertical-specific dictionaries (finance, healthcare, marketing).
* `src/freshdata/engine/backends/_spark.py`, `_freshcore.py`: Spark connectors and Rust C-FFI layers.
* `src/freshdata/simple.py`: Redundant wrapper functions (`clean_csv`, `clean_excel`, `fill_missing`).
* `src/freshdata/context/`, `src/freshdata/learning/`: Natural-language rule compilers and semantic memory.

---

## 4. Pipeline & Configuration Specification

### 4.1 Configuration Models

```python
from dataclasses import dataclass, field
from typing import Literal, Mapping

@dataclass(frozen=True)
class ColumnRule:
    role: Literal["id", "target", "numeric", "categorical", "datetime", "text"] | None = None
    impute: Literal["preserve", "drop", "mean", "median", "mode", "ffill"] | None = None
    outliers: Literal["preserve", "clip", "drop", "flag"] | None = None
    protected: bool = False

@dataclass(frozen=True)
class CleanConfig:
    strategy: Literal["conservative", "balanced", "aggressive"] = "balanced"
    rules: Mapping[str, ColumnRule] = field(default_factory=dict)
    id_columns: tuple[str, ...] = ()
    target_column: str | None = None
    normalize_names: bool = True
    strip_strings: bool = True
    fix_dtypes: bool = True
    drop_duplicates: bool = False
    default_impute: Literal["auto", "preserve", "median", "mode"] = "auto"
    default_outliers: Literal["auto", "preserve", "clip", "flag"] = "preserve"
    iqr_factor: float = 1.5
    strict_backend: bool = True
```

### 4.2 6-Stage Execution Sequence

Execution proceeds strictly in the following order:

1. **`column_names`**:
   * Normalize headers to lowercase snake_case if `normalize_names=True`.
   * Update internal rule references to match normalized names.
2. **`strings`**:
   * Strip leading and trailing whitespace.
   * Collapse internal whitespace runs to a single space.
   * Convert empty strings `""` or whitespace-only values to null.
3. **`dtypes`**:
   * Parse formatted numeric strings (currency symbols `$`, commas `,`, percent signs `%`).
   * Parse common datetime representations (ISO-8601, standard date strings).
   * Downcast or cast low-cardinality string columns to categoricals.
4. **`duplicates`**:
   * Drop duplicate rows only when `drop_duplicates=True`. Defaults to `False` to maintain Polars streaming compatibility without order-sorting bottlenecks.
5. **`missing`**:
   * Impute or drop null values according to explicit `ColumnRule.impute` or `default_impute`.
   * Protected and target columns are never imputed.
6. **`outliers`**:
   * Compute IQR limits ($Q_1 - 1.5 \times \text{IQR}$, $Q_3 + 1.5 \times \text{IQR}$) on numeric columns.
   * Apply `clip`, `flag`, or `drop` based on `ColumnRule.outliers` or `default_outliers`.

---

## 5. Native Backend Dispatch (Zero-Fallback)

### 5.1 Architecture

Execution uses dedicated backend handlers implementing zero cross-engine data conversion:

```python
class BackendHandler(Protocol[DataFrameT]):
    def profile(self, df: DataFrameT) -> ProfileReport: ...
    def apply_stage(self, df: DataFrameT, stage: str, actions: list[ActionRecord]) -> DataFrameT: ...
```

### 5.2 Polars Streaming & Lazy Execution Rules

* **Zero Conversion:** `PolarsBackend` works entirely with native Polars expressions (`pl.col()`). Calls to `to_pandas()` or `pl.from_pandas()` inside the execution path are strictly prohibited.
* **LazyFrame Preservation:** When input is a `pl.LazyFrame`, `apply_plan` returns a `pl.LazyFrame` with the plan appended as lazy expressions.
* **Strict Backend:** When `strict_backend=True`, operations that cannot run natively on the backend raise an immediate `TypeError`.

---

## 6. Verification & Test Plan

1. **`tests/test_core_api.py`**:
   * Verify all 12 exports import cleanly and correctly.
   * Verify `clean(df)` is functionally equivalent to `apply_plan(df, suggest_plan(df))`.
2. **`tests/test_stages.py`**:
   * Unit tests for each of the 6 stages across pandas and Polars.
   * Edge cases: numeric strings with currency/commas, null-only columns, single-row frames.
3. **`tests/test_backend_honesty.py`**:
   * Assert `isinstance(apply_plan(lazy_df, plan), pl.LazyFrame)`.
   * Monkeypatch `to_pandas()` on Polars types to ensure it is never called.
   * Verify `strict_backend=True` raises `TypeError` on unsupported inputs.
4. **`tests/test_plan.py`**:
   * Serialization tests: JSON export/import of `CleanPlan`, `CleanReport`, and `ProfileReport`.
   * Plan diffing: `plan_a.diff(plan_b)` verification.
5. **Coverage & Performance:**
   * Branch coverage >= 90% on `src/freshdata/_core`.
   * Test suite completes in < 3 seconds.
