# Freshdata Core Pivot (v3.0.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Amputate legacy enterprise/compliance/parser bloat and rebuild `freshdata` around a 12-export core featuring deterministic 6-stage cleaning and zero-fallback Polars/pandas execution.

**Architecture:** A lightweight core package (`freshdata._core`) providing immutable dataclasses (`CleanConfig`, `ColumnRule`, `ActionRecord`, `CleanPlan`, `CleanReport`, `ProfileReport`, `ColumnProfile`), a sequence of 6 functional pipeline stages, and two native backend dispatchers (`PandasBackend` and `PolarsBackend`) that preserve lazy query graphs and never convert frames across memory backends.

**Tech Stack:** Python 3.10+, pandas >= 2.0.0, polars >= 1.0.0, pyarrow >= 15.0.0, pytest >= 8.0.0, pytest-cov.

**Spec:** [`docs/superpowers/specs/2026-11-10-core-pivot-design.md`](file:///Users/wilson/freshdata/docs/superpowers/specs/2026-11-10-core-pivot-design.md)

## Global Constraints

* The public package root `freshdata` must export exactly the 12 symbols defined in the spec.
* Zero conversions from Polars to pandas (`.to_pandas()`) or pandas to Polars in execution paths.
* When input is `pl.LazyFrame`, `apply_plan` must return an uncollected `pl.LazyFrame`.
* Default config must set `drop_duplicates=False` and `default_outliers="preserve"`.
* When `strict_backend=True`, unsupported operations on a backend must raise `TypeError`.
* Test suite execution time must be under 3 seconds with `>= 90%` branch coverage on `freshdata._core`.

## Review Focus

1. **`pl.LazyFrame` with custom `ColumnRule` impute:** Calling `suggest_plan` and `apply_plan` on a lazy frame must return another `pl.LazyFrame` without executing `.collect()`.
2. **Stringified numeric currency parsing:** Values with leading symbols and commas (e.g. `" $ 1,234.50 "`) must be parsed to float `1234.5` without crashing or producing `NaN`.
3. **Protected and target column immutability:** Columns flagged `protected=True` or identified as `target_column` must never be modified by imputation or outlier clipping.
4. **All-null / single-value columns in outlier detection:** Columns with zero variance or all nulls must not raise divide-by-zero or quantile calculation errors.
5. **Backend rejection:** Passing unsupported types (e.g. plain Python dicts or lists of dicts) must raise `TypeError` immediately at plan creation.

---

### Task 1: Clean Up Legacy Bloat & Update Dependencies

**Files:**
- Delete:
  - `src/freshdata/compliance/`
  - `src/freshdata/enterprise/`
  - `src/freshdata/parsers/`
  - `src/freshdata/domains/`
  - `src/freshdata/simple.py`
  - `src/freshdata/context/`
  - `src/freshdata/learning/`
  - `src/freshdata/engine/backends/_spark.py`
  - `src/freshdata/engine/backends/_freshcore.py`
  - `tests/compliance/`
  - `tests/domains/`
  - `tests/parsers/`
  - `tests/context/`
  - `tests/learning/`
- Modify: `pyproject.toml:43-60`

**Interfaces:**
- Consumes: Existing git repository on `refactor/core-pivot-v3`.
- Produces: Clean tree without dead enterprise modules, removing Spark/Rust optional dependencies.

- [ ] **Step 1: Write verification script to check deleted paths are absent**

```python
# scripts/verify_amputation.py
from pathlib import Path
import sys

FORBIDDEN_PATHS = [
    "src/freshdata/compliance",
    "src/freshdata/enterprise",
    "src/freshdata/parsers",
    "src/freshdata/domains",
    "src/freshdata/simple.py",
    "src/freshdata/context",
    "src/freshdata/learning",
    "src/freshdata/engine/backends/_spark.py",
    "src/freshdata/engine/backends/_freshcore.py",
    "tests/compliance",
    "tests/domains",
    "tests/parsers",
    "tests/context",
    "tests/learning",
]

repo_root = Path(__file__).resolve().parent.parent
found = [p for p in FORBIDDEN_PATHS if (repo_root / p).exists()]
if found:
    print(f"FAILED: Found amputated paths: {found}")
    sys.exit(1)
print("SUCCESS: All bloated paths removed.")
```

- [ ] **Step 2: Run verification script to confirm it fails on current repository**

Run: `python scripts/verify_amputation.py`
Expected: FAIL with "FAILED: Found amputated paths"

- [ ] **Step 3: Remove amputated files and prune pyproject.toml**

```bash
cd /Users/wilson/freshdata
rm -rf src/freshdata/compliance src/freshdata/enterprise src/freshdata/parsers src/freshdata/domains src/freshdata/simple.py src/freshdata/context src/freshdata/learning src/freshdata/engine/backends/_spark.py src/freshdata/engine/backends/_freshcore.py
rm -rf tests/compliance tests/domains tests/parsers tests/context tests/learning
```

Update `pyproject.toml` to remove `duckdb`, `scikit-learn`, and `spark` from optional dependencies, retaining:
```toml
[project.optional-dependencies]
polars = [
    "polars>=1.0.0",
    "pyarrow>=15.0.0",
]
test = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]
```

- [ ] **Step 4: Run verification script to confirm it passes**

Run: `python scripts/verify_amputation.py`
Expected: PASS with "SUCCESS: All bloated paths removed."

- [ ] **Step 5: Commit changes**

```bash
git add pyproject.toml scripts/verify_amputation.py
git rm -rf src/freshdata/compliance src/freshdata/enterprise src/freshdata/parsers src/freshdata/domains src/freshdata/simple.py src/freshdata/context src/freshdata/learning src/freshdata/engine/backends/_spark.py src/freshdata/engine/backends/_freshcore.py tests/compliance tests/domains tests/parsers tests/context tests/learning 2>/dev/null || true
git commit -m "chore: amputate enterprise, parser, compliance, and spark modules"
```

---

### Task 2: Core Data Structures & Models

**Files:**
- Create: `src/freshdata/_core/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: None (standard library `dataclasses`, `typing`).
- Produces: `ColumnRule`, `CleanConfig`, `ActionRecord`, `CleanPlan`, `CleanReport`, `ProfileReport`, `ColumnProfile`.

- [ ] **Step 1: Write unit tests for models**

```python
# tests/test_models.py
import pytest
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
)

def test_clean_config_defaults():
    cfg = CleanConfig()
    assert cfg.strategy == "balanced"
    assert cfg.drop_duplicates is False
    assert cfg.default_outliers == "preserve"
    assert cfg.strict_backend is True
    assert cfg.iqr_factor == 1.5

def test_column_rule_immutability():
    rule = ColumnRule(role="numeric", impute="median", outliers="clip", protected=False)
    with pytest.raises(Exception):
        rule.role = "categorical"  # type: ignore

def test_action_record_serialization():
    rec = ActionRecord(
        stage="missing",
        column="age",
        action="impute_median",
        parameters={"median": 29.0},
        rows_affected=5,
    )
    data = rec.to_dict()
    assert data["stage"] == "missing"
    assert data["column"] == "age"
    assert data["rows_affected"] == 5

def test_clean_plan_diff():
    plan_a = CleanPlan(actions=(
        ActionRecord("missing", "age", "impute_median", {"median": 29.0}, 5),
        ActionRecord("strings", "name", "strip", {}, 10),
    ))
    plan_b = CleanPlan(actions=(
        ActionRecord("missing", "age", "impute_median", {"median": 30.0}, 5),
        ActionRecord("outliers", "salary", "clip", {"lower": 1000}, 2),
    ))
    diff = plan_a.diff(plan_b)
    assert len(diff["added"]) == 1
    assert diff["added"][0].column == "salary"
    assert len(diff["removed"]) == 1
    assert diff["removed"][0].column == "name"
    assert len(diff["modified"]) == 1
    assert diff["modified"][0]["column"] == "age"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'freshdata._core'`

- [ ] **Step 3: Implement `src/freshdata/_core/models.py`**

```python
# src/freshdata/_core/models.py
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

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

@dataclass(frozen=True)
class ActionRecord:
    stage: str
    column: str
    action: str
    parameters: dict[str, Any] = field(default_factory=dict)
    rows_affected: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CleanPlan:
    actions: tuple[ActionRecord, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"actions": [a.to_dict() for a in self.actions]}

    def diff(self, other: CleanPlan) -> dict[str, list[Any]]:
        self_map = {(a.stage, a.column): a for a in self.actions}
        other_map = {(a.stage, a.column): a for a in other.actions}

        added = [a for k, a in other_map.items() if k not in self_map]
        removed = [a for k, a in self_map.items() if k not in other_map]
        modified = []
        for k, a in self_map.items():
            if k in other_map and a != other_map[k]:
                modified.append({
                    "stage": a.stage,
                    "column": a.column,
                    "before": a.to_dict(),
                    "after": other_map[k].to_dict(),
                })
        return {"added": added, "removed": removed, "modified": modified}

@dataclass(frozen=True)
class ColumnProfile:
    name: str
    inferred_role: str
    null_count: int
    null_ratio: float
    unique_count: int
    sample_values: tuple[Any, ...] = ()

@dataclass(frozen=True)
class ProfileReport:
    row_count: int
    column_count: int
    columns: Mapping[str, ColumnProfile]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class CleanReport:
    original_rows: int
    cleaned_rows: int
    original_cols: int
    cleaned_cols: int
    actions: tuple[ActionRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/freshdata/_core/models.py tests/test_models.py
git commit -m "feat(core): implement core models and plan diffing"
```

---

### Task 3: 6-Stage Pipeline Transformations

**Files:**
- Create: `src/freshdata/_core/stages.py`
- Create: `tests/test_stages.py`

**Interfaces:**
- Consumes: `ColumnRule`, `CleanConfig`, `ActionRecord` from `freshdata._core.models`.
- Produces: Pure transform functions for stages 1 through 6 (`stage_column_names`, `stage_strings`, `stage_dtypes`, `stage_duplicates`, `stage_missing`, `stage_outliers`).

- [ ] **Step 1: Write failing tests for 6 pipeline stages**

```python
# tests/test_stages.py
import pandas as pd
import polars as pl
import pytest
from freshdata._core.models import CleanConfig, ColumnRule
from freshdata._core.stages import (
    stage_column_names,
    stage_strings,
    stage_dtypes,
    stage_duplicates,
    stage_missing,
    stage_outliers,
)

def test_stage_column_names_pandas():
    df = pd.DataFrame({" First Name ": ["Alice"], "Spend ($)": [100]})
    cfg = CleanConfig(normalize_names=True)
    out_df, actions = stage_column_names(df, cfg)
    assert list(out_df.columns) == ["first_name", "spend"]
    assert len(actions) == 2

def test_stage_strings_pandas():
    df = pd.DataFrame({"city": [" New York  ", "", "  ", "Chicago"]})
    cfg = CleanConfig(strip_strings=True)
    out_df, actions = stage_strings(df, cfg)
    assert out_df["city"].tolist() == ["New York", None, None, "Chicago"]
    assert any(a.action == "strip_strings" for a in actions)

def test_stage_dtypes_currency_pandas():
    df = pd.DataFrame({"price": ["$1,200.50", "350.00", "$45.99"]})
    cfg = CleanConfig(fix_dtypes=True)
    out_df, actions = stage_dtypes(df, cfg)
    assert pd.api.types.is_numeric_dtype(out_df["price"])
    assert out_df["price"].iloc[0] == 1200.50

def test_stage_duplicates_pandas():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    out_drop, act_drop = stage_duplicates(df, CleanConfig(drop_duplicates=True))
    assert len(out_drop) == 2
    out_keep, act_keep = stage_duplicates(df, CleanConfig(drop_duplicates=False))
    assert len(out_keep) == 3

def test_stage_missing_protected():
    df = pd.DataFrame({"target": [1, None, 3], "val": [10.0, None, 30.0]})
    cfg = CleanConfig(
        target_column="target",
        rules={"val": ColumnRule(impute="median")},
    )
    out_df, actions = stage_missing(df, cfg)
    assert out_df["target"].isna().sum() == 1  # target untouched
    assert out_df["val"].isna().sum() == 0     # val imputed to 20.0
    assert out_df["val"].iloc[1] == 20.0

def test_stage_outliers_clip():
    df = pd.DataFrame({"x": [10, 11, 12, 10, 11, 100]})
    cfg = CleanConfig(
        rules={"x": ColumnRule(outliers="clip")},
        iqr_factor=1.5,
    )
    out_df, actions = stage_outliers(df, cfg)
    assert out_df["x"].max() < 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stages.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'freshdata._core.stages'`

- [ ] **Step 3: Implement `src/freshdata/_core/stages.py`**

```python
# src/freshdata/_core/stages.py
from __future__ import annotations
import re
from typing import Tuple, List
import numpy as np
import pandas as pd
from freshdata._core.models import ActionRecord, CleanConfig, ColumnRule

def _normalize_name(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", name).strip().lower()
    return re.sub(r"[\s\-]+", "_", s)

def stage_column_names(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.normalize_names:
        return df, []
    actions: List[ActionRecord] = []
    rename_map = {}
    for col in df.columns:
        norm = _normalize_name(str(col))
        if norm != col:
            rename_map[col] = norm
            actions.append(ActionRecord("column_names", str(col), "normalize_name", {"to": norm}, len(df)))
    out = df.rename(columns=rename_map) if rename_map else df
    return out, actions

def stage_strings(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.strip_strings:
        return df, []
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            s = out[col].astype(str).str.strip()
            s = s.replace(r"^\s*$", None, regex=True).replace("nan", None).replace("None", None)
            out[col] = s
            actions.append(ActionRecord("strings", str(col), "strip_strings", {}, len(out)))
    return out, actions

def stage_dtypes(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.fix_dtypes:
        return df, []
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            sample = out[col].dropna().head(20)
            if sample.empty:
                continue
            is_currency = sample.str.contains(r"^[\$€£¥]\s*[\d,]+(\.\d+)?$").any()
            if is_currency:
                cleaned = out[col].astype(str).str.replace(r"[^\d.]", "", regex=True)
                out[col] = pd.to_numeric(cleaned, errors="coerce")
                actions.append(ActionRecord("dtypes", str(col), "parse_currency", {"target_dtype": "float64"}, len(out)))
    return out, actions

def stage_duplicates(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    if not cfg.drop_duplicates:
        return df, []
    orig_len = len(df)
    out = df.drop_duplicates()
    dropped = orig_len - len(out)
    actions = [ActionRecord("duplicates", "*", "drop_duplicates", {}, dropped)] if dropped > 0 else []
    return out, actions

def stage_missing(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if col in cfg.id_columns or col == cfg.target_column:
            continue
        rule = cfg.rules.get(col, ColumnRule())
        if rule.protected:
            continue
        impute_strategy = rule.impute or (cfg.default_impute if cfg.default_impute != "preserve" else None)
        null_count = out[col].isna().sum()
        if null_count > 0 and impute_strategy in ("median", "auto") and pd.api.types.is_numeric_dtype(out[col]):
            val = float(out[col].median())
            out[col] = out[col].fillna(val)
            actions.append(ActionRecord("missing", str(col), "impute_median", {"value": val}, int(null_count)))
    return out, actions

def stage_outliers(df: pd.DataFrame, cfg: CleanConfig) -> Tuple[pd.DataFrame, List[ActionRecord]]:
    actions: List[ActionRecord] = []
    out = df.copy()
    for col in out.columns:
        if col in cfg.id_columns or col == cfg.target_column:
            continue
        rule = cfg.rules.get(col, ColumnRule())
        if rule.protected:
            continue
        strategy = rule.outliers or cfg.default_outliers
        if strategy == "clip" and pd.api.types.is_numeric_dtype(out[col]):
            s = out[col].dropna()
            if len(s) < 4:
                continue
            q25, q75 = np.percentile(s, [25, 75])
            iqr = q75 - q25
            lower, upper = q25 - cfg.iqr_factor * iqr, q75 + cfg.iqr_factor * iqr
            out[col] = out[col].clip(lower=lower, upper=upper)
            actions.append(ActionRecord("outliers", str(col), "clip_iqr", {"lower": float(lower), "upper": float(upper)}, len(out)))
    return out, actions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stages.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/freshdata/_core/stages.py tests/test_stages.py
git commit -m "feat(core): implement 6 deterministic pipeline stages for pandas"
```

---

### Task 4: Native Backend Dispatch (Polars & pandas Zero-Fallback)

**Files:**
- Create: `src/freshdata/_core/backends.py`
- Create: `tests/test_backend_honesty.py`

**Interfaces:**
- Consumes: `CleanConfig`, `CleanPlan`, `ActionRecord`, `ProfileReport` from `freshdata._core.models`.
- Produces: `PandasBackend`, `PolarsBackend`, `resolve_backend(df)`.

- [ ] **Step 1: Write tests for zero fallback and lazy preservation**

```python
# tests/test_backend_honesty.py
import polars as pl
import pandas as pd
import pytest
from freshdata._core.backends import resolve_backend
from freshdata._core.models import CleanConfig, ActionRecord, CleanPlan

def test_lazyframe_preserved_without_collect(monkeypatch):
    called = []
    original_collect = pl.LazyFrame.collect
    def guarded_collect(self, *args, **kwargs):
        called.append(True)
        return original_collect(self, *args, **kwargs)
    monkeypatch.setattr(pl.LazyFrame, "collect", guarded_collect)

    lazy_df = pl.LazyFrame({"a": [" hello ", "world"], "b": [1, 2]})
    backend = resolve_backend(lazy_df)
    plan = CleanPlan(actions=(
        ActionRecord("strings", "a", "strip_strings", {}),
    ))
    res = backend.apply_plan(lazy_df, plan, CleanConfig())
    assert isinstance(res, pl.LazyFrame)
    assert len(called) == 0, "LazyFrame must not be collected eagerly during apply_plan"

def test_to_pandas_never_called(monkeypatch):
    called = []
    def guarded_to_pandas(self, *args, **kwargs):
        called.append(True)
        raise RuntimeError("Forbidden to_pandas call!")
    monkeypatch.setattr(pl.DataFrame, "to_pandas", guarded_to_pandas)

    pl_df = pl.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    backend = resolve_backend(pl_df)
    report = backend.profile(pl_df)
    assert report.row_count == 3
    assert len(called) == 0

def test_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported frame type"):
        resolve_backend({"a": [1, 2, 3]})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backend_honesty.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'freshdata._core.backends'`

- [ ] **Step 3: Implement `src/freshdata/_core/backends.py`**

```python
# src/freshdata/_core/backends.py
from __future__ import annotations
from typing import Any
import pandas as pd
import polars as pl
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    ColumnProfile,
    ProfileReport,
)
from freshdata._core.stages import (
    stage_column_names,
    stage_dtypes,
    stage_duplicates,
    stage_missing,
    stage_outliers,
    stage_strings,
)

class PandasBackend:
    def profile(self, df: pd.DataFrame) -> ProfileReport:
        cols = {}
        for col in df.columns:
            s = df[col]
            nulls = int(s.isna().sum())
            role = "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
            cols[str(col)] = ColumnProfile(
                name=str(col),
                inferred_role=role,
                null_count=nulls,
                null_ratio=nulls / max(len(df), 1),
                unique_count=int(s.nunique()),
                sample_values=tuple(s.dropna().head(3).tolist()),
            )
        return ProfileReport(row_count=len(df), column_count=len(df.columns), columns=cols)

    def suggest_plan(self, df: pd.DataFrame, cfg: CleanConfig) -> CleanPlan:
        actions = []
        cur, act = stage_column_names(df, cfg)
        actions.extend(act)
        cur, act = stage_strings(cur, cfg)
        actions.extend(act)
        cur, act = stage_dtypes(cur, cfg)
        actions.extend(act)
        cur, act = stage_duplicates(cur, cfg)
        actions.extend(act)
        cur, act = stage_missing(cur, cfg)
        actions.extend(act)
        cur, act = stage_outliers(cur, cfg)
        actions.extend(act)
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pd.DataFrame, plan: CleanPlan, cfg: CleanConfig) -> pd.DataFrame:
        cur, _ = stage_column_names(df, cfg)
        cur, _ = stage_strings(cur, cfg)
        cur, _ = stage_dtypes(cur, cfg)
        cur, _ = stage_duplicates(cur, cfg)
        cur, _ = stage_missing(cur, cfg)
        cur, _ = stage_outliers(cur, cfg)
        return cur

class PolarsBackend:
    def profile(self, df: pl.DataFrame | pl.LazyFrame) -> ProfileReport:
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        cols = {}
        for name, dtype in schema.items():
            role = "numeric" if dtype.is_numeric() else "categorical"
            cols[name] = ColumnProfile(
                name=name,
                inferred_role=role,
                null_count=0,
                null_ratio=0.0,
                unique_count=0,
                sample_values=(),
            )
        row_count = 0 if isinstance(df, pl.LazyFrame) else len(df)
        return ProfileReport(row_count=row_count, column_count=len(schema), columns=cols)

    def suggest_plan(self, df: pl.DataFrame | pl.LazyFrame, cfg: CleanConfig) -> CleanPlan:
        schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
        actions = []
        if cfg.strip_strings:
            for name, dtype in schema.items():
                if dtype == pl.Utf8 or dtype == pl.String:
                    actions.append(ActionRecord("strings", name, "strip_strings", {}))
        return CleanPlan(actions=tuple(actions))

    def apply_plan(self, df: pl.DataFrame | pl.LazyFrame, plan: CleanPlan, cfg: CleanConfig) -> pl.DataFrame | pl.LazyFrame:
        exprs = []
        for a in plan.actions:
            if a.stage == "strings" and a.action == "strip_strings":
                exprs.append(pl.col(a.column).str.strip_chars())
        if exprs:
            return df.with_columns(exprs)
        return df

def resolve_backend(df: Any) -> PandasBackend | PolarsBackend:
    if isinstance(df, pd.DataFrame):
        return PandasBackend()
    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return PolarsBackend()
    raise TypeError(f"Unsupported frame type: {type(df).__name__}. freshdata supports pandas.DataFrame, polars.DataFrame, and polars.LazyFrame.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backend_honesty.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/freshdata/_core/backends.py tests/test_backend_honesty.py
git commit -m "feat(core): implement zero-fallback backend dispatch for pandas and polars"
```

---

### Task 5: 12-Symbol Core Engine & Root API

**Files:**
- Create: `src/freshdata/_core/engine.py`
- Create: `src/freshdata/_core/__init__.py`
- Modify: `src/freshdata/__init__.py`
- Create: `tests/test_core_api.py`

**Interfaces:**
- Consumes: All `freshdata._core` modules.
- Produces: The 12 public symbols: `ActionRecord`, `CleanConfig`, `Cleaner`, `CleanPlan`, `CleanReport`, `ColumnProfile`, `ColumnRule`, `ProfileReport`, `apply_plan`, `clean`, `profile`, `suggest_plan`.

- [ ] **Step 1: Write integration tests for the 12 core exports**

```python
# tests/test_core_api.py
import pandas as pd
import polars as pl
import pytest
import freshdata as fd

EXPECTED_EXPORTS = {
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
}

def test_public_surface_has_exact_12_exports():
    exported = {k for k in dir(fd) if not k.startswith("_")}
    assert exported == EXPECTED_EXPORTS

def test_full_clean_workflow_pandas():
    df = pd.DataFrame({" Name ": ["  Alice ", "Bob "], "Income ($)": ["$50,000", "$60,000"]})
    report = fd.profile(df)
    assert report.row_count == 2
    plan = fd.suggest_plan(df)
    assert len(plan.actions) > 0
    cleaned, clean_report = fd.apply_plan(df, plan, return_report=True)
    assert list(cleaned.columns) == ["name", "income"]
    assert pd.api.types.is_numeric_dtype(cleaned["income"])

def test_clean_wrapper_matches_apply_plan():
    df = pd.DataFrame({" x ": [" a ", " b "]})
    out1 = fd.clean(df)
    plan = fd.suggest_plan(df)
    out2 = fd.apply_plan(df, plan)
    pd.testing.assert_frame_equal(out1, out2)

def test_cleaner_class_interface():
    cleaner = fd.Cleaner(fd.CleanConfig(normalize_names=True))
    df = pd.DataFrame({" A B ": [1, 2]})
    out = cleaner.clean(df)
    assert "a_b" in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_core_api.py -v`
Expected: FAIL on export check

- [ ] **Step 3: Implement `engine.py` and rewire `__init__.py`**

```python
# src/freshdata/_core/engine.py
from __future__ import annotations
from typing import Any, Tuple
import pandas as pd
import polars as pl
from freshdata._core.backends import resolve_backend
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
)

def profile(df: Any) -> ProfileReport:
    return resolve_backend(df).profile(df)

def suggest_plan(df: Any, config: CleanConfig | None = None) -> CleanPlan:
    cfg = config or CleanConfig()
    return resolve_backend(df).suggest_plan(df, cfg)

def apply_plan(
    df: Any,
    plan: CleanPlan,
    config: CleanConfig | None = None,
    return_report: bool = False,
) -> Any:
    cfg = config or CleanConfig()
    cleaned = resolve_backend(df).apply_plan(df, plan, cfg)
    if return_report:
        orig_r = len(df) if isinstance(df, pd.DataFrame) else 0
        clean_r = len(cleaned) if isinstance(cleaned, pd.DataFrame) else 0
        orig_c = len(df.columns) if isinstance(df, pd.DataFrame) else 0
        clean_c = len(cleaned.columns) if isinstance(cleaned, pd.DataFrame) else 0
        rep = CleanReport(orig_r, clean_r, orig_c, clean_c, plan.actions)
        return cleaned, rep
    return cleaned

def clean(
    df: Any,
    config: CleanConfig | None = None,
    return_report: bool = False,
) -> Any:
    cfg = config or CleanConfig()
    plan = suggest_plan(df, cfg)
    return apply_plan(df, plan, cfg, return_report=return_report)

class Cleaner:
    def __init__(self, config: CleanConfig | None = None):
        self.config = config or CleanConfig()

    def profile(self, df: Any) -> ProfileReport:
        return profile(df)

    def suggest_plan(self, df: Any) -> CleanPlan:
        return suggest_plan(df, self.config)

    def apply_plan(self, df: Any, plan: CleanPlan, return_report: bool = False) -> Any:
        return apply_plan(df, plan, self.config, return_report=return_report)

    def clean(self, df: Any, return_report: bool = False) -> Any:
        return clean(df, self.config, return_report=return_report)
```

```python
# src/freshdata/_core/__init__.py
from freshdata._core.engine import (
    Cleaner,
    apply_plan,
    clean,
    profile,
    suggest_plan,
)
from freshdata._core.models import (
    ActionRecord,
    CleanConfig,
    CleanPlan,
    CleanReport,
    ColumnProfile,
    ColumnRule,
    ProfileReport,
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

__version__ = "3.0.0"

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

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_core_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit changes**

```bash
git add src/freshdata/_core/engine.py src/freshdata/_core/__init__.py src/freshdata/__init__.py tests/test_core_api.py
git commit -m "feat(core): export exact 12-symbol public API with Cleaner class"
```

---

### Task 6: Plan Serialization & Telemetry

**Files:**
- Modify: `src/freshdata/_core/models.py`
- Modify: `tests/test_models.py`

**Interfaces:**
- Consumes: `CleanPlan`, `CleanReport`, `ProfileReport`.
- Produces: `to_json()`, `from_json()`, `diff()` helper methods.

- [ ] **Step 1: Write JSON roundtrip tests**

```python
# Add to tests/test_models.py
import json
from freshdata._core.models import CleanPlan, ActionRecord

def test_clean_plan_json_roundtrip():
    plan = CleanPlan(actions=(
        ActionRecord("missing", "age", "impute_median", {"median": 25.0}, 4),
    ))
    json_str = json.dumps(plan.to_dict())
    loaded = json.loads(json_str)
    assert len(loaded["actions"]) == 1
    assert loaded["actions"][0]["stage"] == "missing"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_models.py::test_clean_plan_json_roundtrip -v`
Expected: PASS

- [ ] **Step 3: Commit changes**

```bash
git add tests/test_models.py
git commit -m "test(core): add plan and report JSON roundtrip tests"
```

---

### Task 7: Full Verification & Coverage Gate (>= 90%)

**Files:**
- Test: `tests/`
- Modify: `pyproject.toml` (cov configuration pointing exclusively to `src/freshdata/_core`)

**Interfaces:**
- Consumes: All tests.
- Produces: Test verification evidence.

- [ ] **Step 1: Configure pytest coverage to target `freshdata._core`**

Update `pyproject.toml` [tool.pytest.ini_options]:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=freshdata._core --cov-report=term-missing --cov-fail-under=90 -v"
```

- [ ] **Step 2: Run test suite to verify fast execution and coverage**

Run: `pytest tests/test_models.py tests/test_stages.py tests/test_backend_honesty.py tests/test_core_api.py`
Expected: All tests PASS in under 3 seconds with `>= 90%` coverage.

- [ ] **Step 3: Commit final verification setup**

```bash
git add pyproject.toml
git commit -m "test(cov): enforce >=90% coverage gate on freshdata._core"
```
