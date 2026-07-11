# FreshData Performance Baseline and Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic benchmark and profiling system, capture the immutable `6f6c2fe` baseline across the required scale matrix, and produce authoritative evidence that decides which production optimizations deserve separate implementation plans.

**Architecture:** A development-only `benchmarks/performance/` package generates deterministic mixed frames, executes one benchmark case per subprocess, records schema-validated JSON, profiles functions/allocations/observed pandas operations, and renders Markdown comparisons. This phase does not change the cleaning implementation: it establishes the architecture summary, baseline, confirmed bottlenecks, and rejected hypotheses that gate later TDD optimization plans.

**Tech Stack:** Python 3.9+, pandas 1.5-2.x, NumPy, psutil from the existing `bench` extra, standard-library `argparse`, `cProfile`, `pstats`, `subprocess`, `statistics`, `threading`, `tracemalloc`, pytest, jsonschema, GitHub Actions, MkDocs.

## Global Constraints

- Python support remains `>=3.9`, including the project's tested Python 3.9-3.13 matrix.
- pandas support remains `>=1.5,<3`; NumPy remains `>=1.21`.
- The default cleaning strategy remains `balanced`, with the current representation-repair defaults unchanged.
- Public function and class signatures, configuration fields, return types, warnings, exceptions, index semantics, and audit/report contracts remain compatible.
- No unrelated runtime dependency is introduced. Profiling additions use the standard library and already-declared benchmark/development dependencies.
- Baseline source is commit `6f6c2fe`; the design-only commit `21bec88` is not a production-code change.
- One warm-up and five measured repetitions are the default for reportable results.
- A claimed win must be at least 10%, exceed twice observed run-to-run variability, and introduce no meaningful primary-workload regression.
- Preserve the user's untracked `.venv-qa/`; never stage, edit, or remove it.
- Run this plan in an isolated workspace created by `superpowers:using-git-worktrees` before Task 1.

## Scope Decomposition and Evidence Gate

This is Phase 1 of the approved design. It intentionally ends before production optimization because exact optimization code cannot be specified responsibly until profiles identify the functions and lines responsible. The remaining work is split as follows:

1. **Phase 1 — this plan:** benchmark infrastructure, immutable baseline, profiling, architecture summary, root-cause decisions.
2. **Phase 2 — evidence-derived plans:** one TDD plan per confirmed pandas bottleneck; rejected hypotheses receive no production change.
3. **Phase 3 — scale and backend plan:** before/after matrix, native/materialization evaluation, scheduled workflow, documentation corrections.
4. **Phase 4 — verification plan:** complete compatibility matrix, all repository gates, and the 17-section final report audit.

Phase 2 plans may be written only after Task 8 produces `confirmed_root_causes` entries with exact files, lines, stage percentages, copy/scan evidence, and affected benchmark cases.

## File Structure

- `benchmarks/performance/__init__.py` — supported row/width constants and public benchmark-tool exports.
- `benchmarks/performance/models.py` — immutable case/environment/result data models and JSON conversion.
- `benchmarks/performance/datasets.py` — deterministic mixed-schema DataFrame generator.
- `benchmarks/performance/schema.py` — JSON Schema and semantic result validation.
- `benchmarks/performance/environment.py` — package, Git, hardware, and platform capture.
- `benchmarks/performance/memory.py` — peak-RSS sampler used only by benchmark workers.
- `benchmarks/performance/worker.py` — execute one case in a fresh process and write one result.
- `benchmarks/performance/runner.py` — expand matrices and supervise timeout-safe subprocess workers.
- `benchmarks/performance/instrumentation.py` — cProfile, tracemalloc, copy-call, conversion, and scan observations.
- `benchmarks/performance/baselines.py` — comparable component-level pandas operations.
- `benchmarks/performance/analysis.py` — variability, ratios, stage aggregation, and hypothesis classification.
- `benchmarks/performance/render.py` — deterministic Markdown report generation.
- `benchmarks/performance/cli.py` / `__main__.py` — `run`, `profile`, `analyze`, and `render` commands.
- `tests/performance/` — deterministic unit/contract tests for every module and small end-to-end cases.
- `benchmarks/results/performance/` — ignored raw run directories plus explicitly committed compact evidence snapshots.
- `docs/performance-investigation.md` — architecture, commands, baseline tables, profiles, confirmed causes, and rejected hypotheses; later phases extend it to all 17 deliverables.
- `.github/workflows/performance-large.yml` — manual/scheduled large matrix.
- `Makefile` — local commands for CI-sized and large benchmark profiles.
- `.gitignore` — ignore raw results while allowing named compact evidence snapshots.

---

### Task 1: Deterministic Mixed-Schema Dataset Generator

**Files:**
- Create: `benchmarks/performance/__init__.py`
- Create: `benchmarks/performance/datasets.py`
- Create: `tests/performance/conftest.py`
- Create: `tests/performance/test_datasets.py`

**Interfaces:**
- Consumes: NumPy and pandas only.
- Produces: `WIDTHS: dict[str, int]`, `DATASET_TYPES: tuple[str, ...]`,
  `DatasetSpec`, and `make_mixed_frame(spec: DatasetSpec) -> pd.DataFrame`.

- [ ] **Step 1: Write failing generator tests**

```python
# tests/performance/test_datasets.py
from __future__ import annotations

import pandas as pd
import pytest

from benchmarks.performance.datasets import DatasetSpec, WIDTHS, make_mixed_frame


@pytest.mark.parametrize("width, n_cols", WIDTHS.items())
def test_mixed_frame_has_exact_shape_and_required_roles(width: str, n_cols: int) -> None:
    df = make_mixed_frame(DatasetSpec(rows=1_000, width=width, seed=42))
    assert df.shape == (1_000, n_cols)
    assert {"record_id", "target", "numeric_0", "category_0", "text_0", "event_time_0"} <= set(df.columns)
    assert df["target"].isna().sum() > 0
    assert df.duplicated().sum() > 0
    assert pd.api.types.is_categorical_dtype(df["category_0"].dtype)
    assert isinstance(df["event_time_0"].dtype, pd.DatetimeTZDtype)


def test_mixed_frame_is_deterministic_and_seed_sensitive() -> None:
    spec = DatasetSpec(rows=2_000, width="medium", seed=7)
    assert make_mixed_frame(spec).equals(make_mixed_frame(spec))
    assert not make_mixed_frame(spec).equals(
        make_mixed_frame(DatasetSpec(rows=2_000, width="medium", seed=8))
    )


def test_mixed_frame_covers_nullable_outlier_and_high_cardinality_cases() -> None:
    df = make_mixed_frame(DatasetSpec(rows=5_000, width="medium", seed=42))
    assert str(df["nullable_int_0"].dtype) == "Int64"
    assert df["nullable_int_0"].isna().any()
    assert df["numeric_0"].isna().any()
    assert df["numeric_0"].max() > 1_000
    assert df["high_cardinality_0"].nunique() > 2_000


@pytest.mark.parametrize(
    "dataset_type, expected_prefix",
    [
        ("numeric", "numeric_"), ("categorical", "category_"),
        ("string", "text_"), ("nullable", "nullable_int_"),
        ("datetime", "datetime_"),
        ("high_cardinality", "high_cardinality_"),
    ],
)
def test_family_profile_dominates_non_role_columns(dataset_type: str, expected_prefix: str) -> None:
    df = make_mixed_frame(
        DatasetSpec(rows=1_000, width="medium", seed=42, dataset_type=dataset_type)
    )
    family_columns = [name for name in df.columns if name.startswith(expected_prefix)]
    assert len(family_columns) >= 24
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run: `python -m pytest tests/performance/test_datasets.py -q --no-cov`

Expected: FAIL during collection because `benchmarks.performance.datasets` does not exist.

- [ ] **Step 3: Implement the generator**

```python
# benchmarks/performance/datasets.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WIDTHS = {"narrow": 8, "medium": 32, "wide": 128}
DATASET_TYPES = (
    "mixed", "numeric", "categorical", "string", "nullable", "datetime",
    "high_cardinality",
)


@dataclass(frozen=True)
class DatasetSpec:
    rows: int
    width: str
    seed: int = 42
    dataset_type: str = "mixed"

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("rows must be >= 1")
        if self.width not in WIDTHS:
            raise ValueError(f"width must be one of {sorted(WIDTHS)}")
        if self.dataset_type not in DATASET_TYPES:
            raise ValueError(f"dataset_type must be one of {DATASET_TYPES}")


def _nullable_int(rng: np.random.Generator, rows: int) -> pd.Series:
    values = pd.array(rng.integers(0, 10_000, rows), dtype="Int64")
    values[rng.random(rows) < 0.10] = pd.NA
    return pd.Series(values)


def make_mixed_frame(spec: DatasetSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    rows = spec.rows
    numeric = rng.normal(100.0, 20.0, rows)
    numeric[rng.random(rows) < 0.10] = np.nan
    numeric[rng.choice(rows, max(1, rows // 100), replace=False)] *= 20.0
    categories = pd.Categorical(
        rng.choice(["alpha", "beta", "gamma", None], rows, p=[0.35, 0.3, 0.25, 0.1])
    )
    frame = pd.DataFrame(
        {
            "record_id": np.arange(rows, dtype=np.int64),
            "target": pd.array(rng.choice([0, 1, None], rows, p=[0.47, 0.48, 0.05]), dtype="Int8"),
            "numeric_0": numeric,
            "nullable_int_0": _nullable_int(rng, rows),
            "category_0": categories,
            "text_0": pd.array([f"free form note {i % 97}" if i % 11 else None for i in range(rows)], dtype="string"),
            "event_time_0": pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC"),
            "high_cardinality_0": pd.array([f"key-{spec.seed}-{i}" for i in range(rows)], dtype="string"),
        }
    )
    factories = (
        lambda i: pd.Series(rng.normal(i, 1.0, rows), name=f"numeric_{i}"),
        lambda i: _nullable_int(rng, rows).rename(f"nullable_int_{i}"),
        lambda i: pd.Series(pd.Categorical(rng.choice(["a", "b", "c", None], rows)), name=f"category_{i}"),
        lambda i: pd.Series(pd.array([f"value {i}-{j % 211}" for j in range(rows)], dtype="string"), name=f"text_{i}"),
        lambda i: pd.Series(pd.date_range("2020-01-01", periods=rows, freq="h"), name=f"datetime_{i}"),
        lambda i: pd.Series(pd.array([f"hc-{i}-{j}" for j in range(rows)], dtype="string"), name=f"high_cardinality_{i}"),
    )
    family_index = {
        "numeric": 0, "nullable": 1, "categorical": 2, "string": 3,
        "datetime": 4, "high_cardinality": 5,
    }.get(spec.dataset_type)
    if family_index is not None:
        frame = frame[["record_id", "target"]].copy()
    i = 1
    while frame.shape[1] < WIDTHS[spec.width]:
        index = family_index if family_index is not None else (i - 1) % len(factories)
        series = factories[index](i)
        frame[series.name] = series
        i += 1
    if rows >= 100:
        duplicate_count = max(1, rows // 100)
        frame = pd.concat(
            [frame.iloc[:-duplicate_count], frame.iloc[:duplicate_count].copy()],
            ignore_index=True,
        )
    return frame
```

```python
# benchmarks/performance/__init__.py
from .datasets import DATASET_TYPES, DatasetSpec, WIDTHS, make_mixed_frame

__all__ = ["DATASET_TYPES", "DatasetSpec", "WIDTHS", "make_mixed_frame"]
```

```python
# tests/performance/conftest.py
from __future__ import annotations

# Performance-tool modules live in the repository and are imported as the
# namespace package ``benchmarks.performance``. No sys.path mutation is needed
# when tests run with ``python -m pytest`` from the repository root.
```

- [ ] **Step 4: Run generator tests**

Run: `python -m pytest tests/performance/test_datasets.py -q --no-cov`

Expected: PASS for all width, determinism, dtype, defect, and role assertions.

- [ ] **Step 5: Commit the generator**

```bash
git add benchmarks/performance/__init__.py benchmarks/performance/datasets.py tests/performance/conftest.py tests/performance/test_datasets.py
git commit -m "bench: add deterministic scalability datasets"
```

### Task 2: Typed Cases, Environment Capture, and Result Schema

**Files:**
- Create: `benchmarks/performance/models.py`
- Create: `benchmarks/performance/environment.py`
- Create: `benchmarks/performance/schema.py`
- Create: `tests/performance/test_models_schema.py`

**Interfaces:**
- Consumes: `DatasetSpec` from Task 1.
- Produces: `BenchmarkCase`, `EnvironmentInfo`, `BenchmarkResult`, `capture_environment()`, `RESULT_SCHEMA`, and `validate_result(payload)`.

- [ ] **Step 1: Write failing model/schema tests**

```python
# tests/performance/test_models_schema.py
from __future__ import annotations

from dataclasses import replace

import pytest

from benchmarks.performance.environment import capture_environment
from benchmarks.performance.models import BenchmarkCase, BenchmarkResult
from benchmarks.performance.schema import validate_result


def test_case_id_is_stable_and_configuration_sensitive() -> None:
    case = BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={})
    assert case.case_id == BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}).case_id
    assert case.case_id != replace(case, return_report=True).case_id


def test_environment_contains_required_reproduction_fields() -> None:
    env = capture_environment()
    assert env.python_version
    assert env.pandas_version
    assert env.numpy_version
    assert env.git_commit
    assert env.platform
    assert env.cpu_count_logical >= 1


def test_result_round_trip_validates() -> None:
    case = BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={})
    result = BenchmarkResult.completed(
        case=case,
        environment=capture_environment(),
        samples_seconds=[1.0, 1.1, 0.9],
        peak_rss_bytes=1_000_000,
        peak_python_bytes=500_000,
        input_bytes=250_000,
        command="python -m benchmarks.performance worker",
    )
    payload = result.to_dict()
    validate_result(payload)
    assert BenchmarkResult.from_dict(payload).case == case


def test_completed_result_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="samples_seconds"):
        BenchmarkResult.completed(
            case=BenchmarkCase(rows=10_000, width="narrow", config_name="default", options={}),
            environment=capture_environment(),
            samples_seconds=[],
            peak_rss_bytes=0,
            peak_python_bytes=0,
            input_bytes=1,
            command="x",
        )
```

- [ ] **Step 2: Verify the tests fail because the modules are absent**

Run: `python -m pytest tests/performance/test_models_schema.py -q --no-cov`

Expected: FAIL during collection for missing `models`, `environment`, or `schema`.

- [ ] **Step 3: Implement exact case and result models**

```python
# benchmarks/performance/models.py
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkCase:
    rows: int
    width: str
    config_name: str
    options: dict[str, Any]
    dataset_type: str = "mixed"
    return_report: bool = False
    backend: str = "pandas"
    output_format: str = "pandas"
    seed: int = 42
    warmups: int = 1
    repetitions: int = 5

    @property
    def case_id(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class EnvironmentInfo:
    git_commit: str
    git_dirty: bool
    python_version: str
    pandas_version: str
    numpy_version: str
    freshdata_version: str
    optional_versions: dict[str, str | None]
    platform: str
    processor: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    total_ram_bytes: int | None


@dataclass(frozen=True)
class BenchmarkResult:
    schema_version: int
    status: str
    case: BenchmarkCase
    environment: EnvironmentInfo
    samples_seconds: list[float] = field(default_factory=list)
    median_seconds: float | None = None
    min_seconds: float | None = None
    max_seconds: float | None = None
    stdev_seconds: float | None = None
    coefficient_of_variation: float | None = None
    throughput_rows_per_second: float | None = None
    peak_rss_bytes: int | None = None
    peak_python_bytes: int | None = None
    input_bytes: int | None = None
    input_to_peak_ratio: float | None = None
    command: str = ""
    error_type: str | None = None
    error_message: str | None = None
    output_fingerprint: str | None = None
    report_fingerprint: str | None = None
    result_type: str | None = None

    @classmethod
    def completed(cls, *, case: BenchmarkCase, environment: EnvironmentInfo,
                  samples_seconds: list[float], peak_rss_bytes: int,
                  peak_python_bytes: int, input_bytes: int, command: str,
                  output_fingerprint: str | None = None,
                  report_fingerprint: str | None = None,
                  result_type: str | None = None) -> "BenchmarkResult":
        if not samples_seconds:
            raise ValueError("samples_seconds must not be empty")
        median = statistics.median(samples_seconds)
        stdev = statistics.stdev(samples_seconds) if len(samples_seconds) > 1 else 0.0
        return cls(
            schema_version=1, status="completed", case=case, environment=environment,
            samples_seconds=samples_seconds, median_seconds=median,
            min_seconds=min(samples_seconds), max_seconds=max(samples_seconds),
            stdev_seconds=stdev,
            coefficient_of_variation=(stdev / median) if median else 0.0,
            throughput_rows_per_second=(case.rows / median) if median else None,
            peak_rss_bytes=peak_rss_bytes, peak_python_bytes=peak_python_bytes,
            input_bytes=input_bytes,
            input_to_peak_ratio=(peak_rss_bytes / input_bytes) if input_bytes else None,
            command=command, output_fingerprint=output_fingerprint,
            report_fingerprint=report_fingerprint, result_type=result_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkResult":
        data = dict(payload)
        data["case"] = BenchmarkCase(**data["case"])
        data["environment"] = EnvironmentInfo(**data["environment"])
        return cls(**data)
```

- [ ] **Step 4: Implement environment capture and schema validation**

```python
# benchmarks/performance/environment.py
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess

import numpy as np
import pandas as pd

import freshdata

from .models import EnvironmentInfo


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment() -> EnvironmentInfo:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout)
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        total_ram = int(psutil.virtual_memory().total)
    except ImportError:
        physical = None
        total_ram = None
    return EnvironmentInfo(
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        freshdata_version=freshdata.__version__,
        optional_versions={name: _version(name) for name in ("polars", "duckdb", "pyspark", "pyarrow")},
        platform=platform.platform(), processor=platform.processor(),
        cpu_count_logical=os.cpu_count() or 1,
        cpu_count_physical=physical, total_ram_bytes=total_ram,
    )
```

```python
# benchmarks/performance/schema.py
from __future__ import annotations

from typing import Any

RESULT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "status", "case", "environment", "samples_seconds", "command"],
    "properties": {
        "schema_version": {"const": 1},
        "status": {"enum": ["completed", "failed", "timeout", "oom", "skipped"]},
        "case": {"type": "object", "required": ["rows", "width", "config_name", "options", "dataset_type", "return_report", "backend", "output_format", "seed", "warmups", "repetitions"]},
        "environment": {"type": "object", "required": ["git_commit", "python_version", "pandas_version", "numpy_version", "freshdata_version", "platform"]},
        "samples_seconds": {"type": "array", "items": {"type": "number", "minimum": 0}},
        "command": {"type": "string"},
    },
}


def validate_result(payload: dict[str, Any]) -> None:
    import jsonschema
    jsonschema.validate(payload, RESULT_SCHEMA)
    if payload["status"] == "completed":
        if not payload["samples_seconds"]:
            raise ValueError("completed result requires samples_seconds")
        for field in ("median_seconds", "min_seconds", "max_seconds", "peak_rss_bytes", "peak_python_bytes", "input_bytes"):
            if payload.get(field) is None:
                raise ValueError(f"completed result requires {field}")
```

- [ ] **Step 5: Run and pass model/schema tests**

Run: `python -m pytest tests/performance/test_models_schema.py -q --no-cov`

Expected: PASS.

- [ ] **Step 6: Commit the data contracts**

```bash
git add benchmarks/performance/models.py benchmarks/performance/environment.py benchmarks/performance/schema.py tests/performance/test_models_schema.py
git commit -m "bench: define performance result contract"
```

### Task 3: Isolated Worker Timing and Peak-Memory Measurement

**Files:**
- Create: `benchmarks/performance/memory.py`
- Create: `benchmarks/performance/worker.py`
- Create: `tests/performance/test_worker.py`

**Interfaces:**
- Consumes: `BenchmarkCase`, `BenchmarkResult`, `DatasetSpec`, `capture_environment()`.
- Produces: `PeakRSS`, `execute_case(case: BenchmarkCase, command: str) -> BenchmarkResult`, and `worker_main(case_path, result_path)`.

- [ ] **Step 1: Write failing worker tests**

```python
# tests/performance/test_worker.py
from __future__ import annotations

import pandas as pd

from benchmarks.performance.models import BenchmarkCase
from benchmarks.performance.worker import execute_case


def test_worker_records_warm_samples_memory_and_embedded_report() -> None:
    case = BenchmarkCase(
        rows=500, width="narrow", config_name="default", options={"verbose": False},
        return_report=False, warmups=1, repetitions=2,
    )
    result = execute_case(case, command="unit-test")
    assert result.status == "completed"
    assert len(result.samples_seconds) == 2
    assert result.peak_rss_bytes is not None and result.peak_rss_bytes >= 0
    assert result.peak_python_bytes is not None and result.peak_python_bytes > 0
    assert result.input_bytes is not None and result.input_bytes > 0


def test_worker_report_flag_does_not_change_cleaned_values() -> None:
    base = BenchmarkCase(rows=500, width="narrow", config_name="default", options={"verbose": False}, warmups=0, repetitions=1)
    without_report = execute_case(base, command="unit-test")
    with_report = execute_case(
        BenchmarkCase(**{**base.__dict__, "return_report": True}), command="unit-test"
    )
    assert without_report.status == with_report.status == "completed"
    assert without_report.output_fingerprint == with_report.output_fingerprint
    assert without_report.report_fingerprint == with_report.report_fingerprint
    assert without_report.result_type == "CleanResult"
    assert with_report.result_type == "tuple[CleanResult,CleanReport]"
```

- [ ] **Step 2: Verify the worker tests fail**

Run: `python -m pytest tests/performance/test_worker.py -q --no-cov`

Expected: FAIL because `worker.py` and `memory.py` do not exist.

- [ ] **Step 3: Implement the RSS sampler**

```python
# benchmarks/performance/memory.py
from __future__ import annotations

import gc
import threading


class PeakRSS:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        import psutil
        self._process = psutil.Process()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline = 0
        self._peak = 0

    def __enter__(self) -> "PeakRSS":
        gc.collect()
        self._baseline = self._process.memory_info().rss
        self._peak = self._baseline
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self._peak = max(self._peak, self._process.memory_info().rss)

    def __exit__(self, *_args: object) -> None:
        self._peak = max(self._peak, self._process.memory_info().rss)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def increase_bytes(self) -> int:
        return max(0, self._peak - self._baseline)
```

- [ ] **Step 4: Implement one-case execution**

```python
# benchmarks/performance/worker.py
from __future__ import annotations

import gc
import json
import time
import tracemalloc
from pathlib import Path

import freshdata as fd

from .datasets import DatasetSpec, make_mixed_frame
from .environment import capture_environment
from .memory import PeakRSS
from .models import BenchmarkCase, BenchmarkResult
from .schema import validate_result


def _run_clean(frame, case: BenchmarkCase):
    return fd.clean(
        frame, config=case.options, return_report=case.return_report,
        engine=case.backend, output_format=case.output_format,
    )


def _fingerprints(result, return_report: bool) -> tuple[str, str, str]:
    import hashlib
    import json
    import pandas as pd

    if return_report:
        cleaned, report = result
        result_type = "tuple[CleanResult,CleanReport]"
    else:
        cleaned, report = result, result.report()
        result_type = "CleanResult"
    frame_hash = hashlib.sha256()
    frame_hash.update(pd.util.hash_pandas_object(cleaned, index=True).values.tobytes())
    frame_hash.update(repr(list(cleaned.columns)).encode())
    frame_hash.update(repr([str(dtype) for dtype in cleaned.dtypes]).encode())
    report_payload = report.to_dict()
    report_payload.pop("duration_seconds", None)
    report_payload.pop("stage_timings", None)
    report_hash = hashlib.sha256(
        json.dumps(report_payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return frame_hash.hexdigest(), report_hash, result_type


def execute_case(case: BenchmarkCase, *, command: str) -> BenchmarkResult:
    frame = make_mixed_frame(
        DatasetSpec(case.rows, case.width, case.seed, case.dataset_type)
    )
    input_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    for _ in range(case.warmups):
        _run_clean(frame, case)
    samples: list[float] = []
    for _ in range(case.repetitions):
        gc.collect()
        started = time.perf_counter()
        _run_clean(frame, case)
        samples.append(time.perf_counter() - started)
    gc.collect()
    tracemalloc.start()
    with PeakRSS() as rss:
        measured_result = _run_clean(frame, case)
    _current, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    output_fingerprint, report_fingerprint, result_type = _fingerprints(
        measured_result, case.return_report
    )
    return BenchmarkResult.completed(
        case=case, environment=capture_environment(), samples_seconds=samples,
        peak_rss_bytes=rss.increase_bytes, peak_python_bytes=python_peak,
        input_bytes=input_bytes, command=command,
        output_fingerprint=output_fingerprint,
        report_fingerprint=report_fingerprint, result_type=result_type,
    )


def worker_main(case_path: str, result_path: str, command: str) -> None:
    case = BenchmarkCase(**json.loads(Path(case_path).read_text()))
    result = execute_case(case, command=command)
    payload = result.to_dict()
    validate_result(payload)
    Path(result_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
```

- [ ] **Step 5: Run worker tests**

Run: `python -m pytest tests/performance/test_worker.py -q --no-cov`

Expected: PASS.

- [ ] **Step 6: Commit worker measurement**

```bash
git add benchmarks/performance/memory.py benchmarks/performance/worker.py tests/performance/test_worker.py
git commit -m "bench: measure isolated runtime and peak memory"
```

### Task 4: Matrix Expansion, Timeout-Safe Subprocess Runner, and CLI

**Files:**
- Create: `benchmarks/performance/runner.py`
- Create: `benchmarks/performance/cli.py`
- Create: `benchmarks/performance/__main__.py`
- Create: `tests/performance/test_runner_cli.py`

**Interfaces:**
- Consumes: `BenchmarkCase`, `worker_main`, and JSON result validation.
- Produces: `expand_cases(...)`, `run_case_subprocess(...)`, `run_matrix(...)`, and `python -m benchmarks.performance run`.

- [ ] **Step 1: Write failing runner/CLI tests**

```python
# tests/performance/test_runner_cli.py
from __future__ import annotations

import json

from benchmarks.performance.cli import main
from benchmarks.performance.runner import expand_cases


def test_expand_cases_crosses_required_dimensions() -> None:
    cases = expand_cases(
        rows=[10_000, 100_000], widths=["narrow", "wide"],
        dataset_types=["mixed"],
        configs=["default", "conservative"], report_modes=[False, True],
        backends=["pandas"], output_formats=["pandas"], seed=42,
        warmups=1, repetitions=5,
    )
    assert len(cases) == 16
    assert len({case.case_id for case in cases}) == 16


def test_cli_small_matrix_writes_valid_result(tmp_path) -> None:
    exit_code = main([
        "run", "--rows", "250", "--widths", "narrow", "--configs", "default",
        "--report-modes", "false", "--repetitions", "1", "--warmups", "0",
        "--timeout", "60", "--output", str(tmp_path),
    ])
    assert exit_code == 0
    payloads = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(payloads) == 1
    assert payloads[0]["status"] == "completed"
```

- [ ] **Step 2: Verify runner/CLI tests fail**

Run: `python -m pytest tests/performance/test_runner_cli.py -q --no-cov`

Expected: FAIL because `runner.py` and `cli.py` do not exist.

- [ ] **Step 3: Implement matrix expansion and failure records**

Implement these exact configuration profiles in `benchmarks/performance/runner.py`:

```python
CONFIGS = {
    "default": {"verbose": False},
    "conservative": {"strategy": "conservative", "verbose": False},
    "representation_off": {
        "strategy": "conservative", "column_names": False,
        "strip_whitespace": False, "normalize_sentinels": False,
        "fix_dtypes": False, "drop_duplicates": False, "verbose": False,
    },
    "statistical_off": {
        "strategy": "conservative", "impute": None, "outliers": None,
        "verbose": False,
    },
    "explicit": {
        "strategy": "conservative", "impute": "median", "outliers": "flag",
        "verbose": False,
    },
    "aggressive": {"strategy": "aggressive", "verbose": False},
    "semantic": {"semantic_mode": "assist", "verbose": False},
    "missforest": {
        "strategy": "conservative", "impute": "missforest", "verbose": False,
    },
}
```

Use `itertools.product` across rows, widths, dataset types, configs, report
modes, backends, and output formats to construct `BenchmarkCase` objects.
`run_case_subprocess` must write the case JSON into a temporary directory,
invoke:

```python
[
    sys.executable, "-c",
    "from benchmarks.performance.worker import worker_main; "
    "worker_main(__import__('sys').argv[1], __import__('sys').argv[2], __import__('sys').argv[3])",
    str(case_path), str(result_path), command,
]
```

with `subprocess.run(..., timeout=timeout_seconds, capture_output=True, text=True)`. On timeout, non-zero exit, missing result, or a result containing `MemoryError`, write a schema-valid `BenchmarkResult` with status `timeout`, `failed`, or `oom`, plus the exception type/message and exact command. Do not raise and lose the remaining matrix.

- [ ] **Step 4: Implement the CLI parser**

`benchmarks/performance/cli.py` must parse comma-separated values with these defaults:

```python
run.add_argument("--rows", default="10000,100000,500000,1000000")
run.add_argument("--widths", default="narrow,medium,wide")
run.add_argument("--dataset-types", default="mixed")
run.add_argument("--configs", default="default,conservative,representation_off,statistical_off,explicit")
run.add_argument("--report-modes", default="false,true")
run.add_argument("--backends", default="pandas")
run.add_argument("--output-formats", default="pandas")
run.add_argument("--seed", type=int, default=42)
run.add_argument("--warmups", type=int, default=1)
run.add_argument("--repetitions", type=int, default=5)
run.add_argument("--timeout", type=int, default=1800)
run.add_argument("--output", required=True)
```

The command prints one line per case and ends with counts by status. It returns `0` only when every requested case is completed or explicitly skipped; failures, timeouts, and OOM records return `1` after all cases finish.

```python
# benchmarks/performance/__main__.py
from .cli import main

raise SystemExit(main())
```

- [ ] **Step 5: Run runner/CLI tests**

Run: `python -m pytest tests/performance/test_runner_cli.py -q --no-cov`

Expected: PASS, with one completed small-case JSON file.

- [ ] **Step 6: Commit the matrix runner**

```bash
git add benchmarks/performance/runner.py benchmarks/performance/cli.py benchmarks/performance/__main__.py tests/performance/test_runner_cli.py
git commit -m "bench: add configurable subprocess matrix runner"
```

### Task 5: Function, Allocation, Copy, Conversion, and Scan Profiling

**Files:**
- Create: `benchmarks/performance/instrumentation.py`
- Create: `tests/performance/test_instrumentation.py`
- Modify: `benchmarks/performance/worker.py`
- Modify: `benchmarks/performance/models.py`
- Modify: `benchmarks/performance/schema.py`
- Modify: `benchmarks/performance/cli.py`

**Interfaces:**
- Consumes: one `BenchmarkCase` and the Task 3 worker execution path.
- Produces: `OperationCounter`, `profile_case(case) -> ProfileResult`, exact function/file/line records, allocation records, stage aggregates, and `python -m benchmarks.performance profile`.

- [ ] **Step 1: Write failing instrumentation tests**

```python
# tests/performance/test_instrumentation.py
from __future__ import annotations

from benchmarks.performance.instrumentation import OperationCounter, profile_case
from benchmarks.performance.models import BenchmarkCase


def test_operation_counter_observes_copy_scan_and_conversion_calls() -> None:
    import pandas as pd
    frame = pd.DataFrame({"a": [1, None, 3], "b": ["1", "2", "3"]})
    with OperationCounter() as counter:
        frame.copy(deep=False)
        frame["a"].isna()
        frame["a"].nunique(dropna=True)
        frame["b"].astype("string")
    assert counter.counts["dataframe.copy"] == 1
    assert counter.counts["series.isna"] >= 1
    assert counter.counts["series.nunique"] == 1
    assert counter.counts["series.astype"] == 1


def test_profile_case_reports_exact_hot_functions_and_allocations() -> None:
    result = profile_case(BenchmarkCase(
        rows=500, width="narrow", config_name="default",
        options={"verbose": False}, warmups=0, repetitions=1,
    ))
    assert result.functions
    assert all({"file", "line", "function", "self_seconds", "cumulative_seconds", "calls"} <= set(item) for item in result.functions)
    assert result.allocations
    assert all({"file", "line", "bytes", "count"} <= set(item) for item in result.allocations)
    assert result.stages["total"] > 0
    assert result.operations["dataframe.copy"] >= 1
```

- [ ] **Step 2: Verify instrumentation tests fail**

Run: `python -m pytest tests/performance/test_instrumentation.py -q --no-cov`

Expected: FAIL because `instrumentation.py` does not exist.

- [ ] **Step 3: Implement controlled operation counting**

Create `OperationCounter` as a context manager using `unittest.mock.patch.object` around the original methods. Store originals before patching and route each wrapper directly to its original to avoid recursion. Count these keys:

```python
METHODS = {
    "dataframe.copy": (pd.DataFrame, "copy"),
    "series.copy": (pd.Series, "copy"),
    "series.isna": (pd.Series, "isna"),
    "series.notna": (pd.Series, "notna"),
    "series.nunique": (pd.Series, "nunique"),
    "series.value_counts": (pd.Series, "value_counts"),
    "series.astype": (pd.Series, "astype"),
    "dataframe.astype": (pd.DataFrame, "astype"),
    "dataframe.corr": (pd.DataFrame, "corr"),
    "dataframe.corrwith": (pd.DataFrame, "corrwith"),
}
```

Mark these as observed Python method calls, not physical-buffer-copy counts.

- [ ] **Step 4: Implement cProfile and tracemalloc extraction**

Add this immutable result contract to `benchmarks/performance/instrumentation.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileResult:
    functions: list[dict[str, object]]
    allocations: list[dict[str, object]]
    stages: dict[str, float]
    operations: dict[str, int]
```

`profile_case(case: BenchmarkCase) -> ProfileResult` must:

1. Generate the deterministic frame outside the measured profile.
2. Start `tracemalloc`, `cProfile.Profile`, and `OperationCounter`.
3. Execute one `fd.clean` call.
4. Disable profiling and collect the top 100 `pstats.Stats.stats` entries sorted by cumulative time.
5. Collect the top 100 `tracemalloc` line statistics limited to paths containing `/freshdata/` or `/benchmarks/performance/`.
6. Aggregate self time into these stages using normalized path/function rules:

```python
STAGE_RULES = {
    "context": ("engine/context.py",),
    "engine_cache": ("engine/cache.py",),
    "correlation": ("numeric_corr_matrix", "corr", "corrwith"),
    "missing": ("engine/missing.py", "steps/missing.py"),
    "outliers": ("engine/outliers.py", "steps/outliers.py"),
    "role_inference": ("infer_role", "build_context"),
    "dtype_repair": ("steps/dtypes.py",),
    "duplicates": ("steps/duplicates.py",),
    "audit_events": ("report.py", "CleanReport.add"),
    "report_finalization": ("cleaner.py", "memory_bytes"),
    "semantic_ml": ("semantic/", "imputation/missforest.py", "sklearn/"),
    "backend_conversion": ("adapters/", "execution/backends/"),
}
```

Use exclusive self time for percentage totals so nested functions are not double-counted. Preserve cumulative time separately for locating call chains.

- [ ] **Step 5: Extend result schema and CLI**

Add optional `profile` to `BenchmarkResult` containing `functions`,
`allocations`, `stages`, and `operations`. Add a `profile` subcommand accepting
the same case selectors as `run` but requiring exactly one row count, width,
config, report mode, backend, and output format. It writes the case's stable
16-character ID followed by `.profile.json` and validates it with the same
schema.

- [ ] **Step 6: Run instrumentation and focused worker tests**

Run: `python -m pytest tests/performance/test_instrumentation.py tests/performance/test_worker.py -q --no-cov`

Expected: PASS.

- [ ] **Step 7: Commit profiling instrumentation**

```bash
git add benchmarks/performance/instrumentation.py benchmarks/performance/worker.py benchmarks/performance/models.py benchmarks/performance/schema.py benchmarks/performance/cli.py tests/performance/test_instrumentation.py
git commit -m "bench: profile stages allocations and pandas operations"
```

### Task 6: Comparable Pandas Baselines, Analysis, and Markdown Rendering

**Files:**
- Create: `benchmarks/performance/baselines.py`
- Create: `benchmarks/performance/analysis.py`
- Create: `benchmarks/performance/render.py`
- Create: `tests/performance/test_analysis_render.py`
- Modify: `benchmarks/performance/cli.py`

**Interfaces:**
- Consumes: completed benchmark/profile JSON files.
- Produces: `measure_pandas_baseline(case, baseline_name)`, a `baseline` CLI
  subcommand, component-level pandas baseline samples, slowdown/improvement
  ratios, variability-aware comparisons, hypothesis classifications, and
  deterministic Markdown.

- [ ] **Step 1: Write failing comparison/render tests**

```python
# tests/performance/test_analysis_render.py
from __future__ import annotations

from benchmarks.performance.analysis import classify_change, classify_hypotheses
from benchmarks.performance.render import render_report


def test_change_requires_ten_percent_and_twice_variability() -> None:
    assert classify_change(1.0, 0.85, baseline_cv=0.02, candidate_cv=0.02) == "improved"
    assert classify_change(1.0, 0.94, baseline_cv=0.01, candidate_cv=0.01) == "noise"
    assert classify_change(1.0, 1.12, baseline_cv=0.02, candidate_cv=0.02) == "regressed"


def test_hypothesis_classifier_requires_exact_profile_evidence() -> None:
    profile = {
        "stages": {"correlation": 0.40, "total": 1.0},
        "operations": {"dataframe.corr": 1, "dataframe.copy": 3},
        "functions": [{"file": "src/freshdata/engine/context.py", "line": 207, "function": "numeric_corr_matrix", "self_seconds": 0.4, "cumulative_seconds": 0.4, "calls": 1}],
    }
    decisions = classify_hypotheses(profile)
    assert decisions["unnecessary_correlation"]["status"] == "candidate"
    assert decisions["unnecessary_correlation"]["evidence"][0]["line"] == 207


def test_renderer_is_deterministic_and_contains_required_baseline_sections() -> None:
    payload = {"environment": {"python_version": "3.12", "pandas_version": "2.3"}, "results": [], "hypotheses": {}}
    first = render_report(payload)
    assert first == render_report(payload)
    for heading in ("Architecture and execution flow", "Reproduction commands", "Baseline benchmark table", "Profiling findings", "Confirmed root causes", "Rejected hypotheses"):
        assert f"## {heading}" in first
```

- [ ] **Step 2: Verify comparison/render tests fail**

Run: `python -m pytest tests/performance/test_analysis_render.py -q --no-cov`

Expected: FAIL because the analysis and render modules do not exist.

- [ ] **Step 3: Implement semantically comparable pandas baselines**

Provide named component baselines only:

```python
BASELINES = {
    "shallow_copy": lambda df: df.copy(deep=False),
    "numeric_median_fill": lambda df: df.assign(**{
        col: df[col].fillna(df[col].median())
        for col in df.select_dtypes(include="number").columns
        if df[col].notna().any()
    }),
    "duplicates": lambda df: df.drop_duplicates(),
    "null_counts": lambda df: df.isna().sum(),
}
```

Do not define a pandas baseline for FreshData's full balanced decision/audit pipeline. Label every baseline with its exact operation and include it only when the selected case has matching semantics.

`measure_pandas_baseline` uses the same one-warm-up/five-sample and isolated
PeakRSS/tracemalloc procedure as `execute_case`, but invokes the selected
function from `BASELINES`. Store `baseline_name` in the result payload and set
`backend="pandas-component-baseline"`; never route this sentinel backend
through `fd.clean`.

- [ ] **Step 4: Implement variability and hypothesis analysis**

`classify_change` calculates `(baseline - candidate) / baseline` and compares its absolute magnitude with both `0.10` and `2 * max(baseline_cv, candidate_cv)`. Return only `improved`, `regressed`, or `noise`.

`classify_hypotheses` produces entries for:

- `unnecessary_correlation`
- `repeated_null_scans`
- `repeated_uniqueness_scans`
- `copy_pressure`
- `dtype_conversion_pressure`
- `report_finalization_overhead`
- `optional_ml_overhead`
- `backend_conversion_overhead`

Each entry contains `status` (`candidate`, `rejected`, or `insufficient_evidence`), `stage_fraction`, `observed_calls`, and an `evidence` list of exact function/file/line records. A hypothesis is only a candidate when its stage consumes at least 10% of total self time or its peak allocations consume at least 10% of the traced peak, and the relevant operation count is non-zero.

- [ ] **Step 5: Implement deterministic Markdown rendering**

`render_report` sorts cases by rows, width, config, report flag, backend, and output format. It renders environment data, exact commands, median/min/max/CV/throughput, peak RSS/Python/input ratios, pandas comparison ratios, stage percentages, top exact functions/lines, confirmed candidates, rejected hypotheses, failures, timeouts, OOMs, and limitations. It must not print an improvement claim for a comparison classified as `noise`.

- [ ] **Step 6: Add `analyze` and `render` CLI commands**

Commands:

```bash
python -m benchmarks.performance analyze --input benchmarks/results/performance/baseline --output benchmarks/results/performance/baseline-summary.json
python -m benchmarks.performance render --input benchmarks/results/performance/baseline-summary.json --output benchmarks/results/performance/baseline-report.md
```

Also add:

```bash
python -m benchmarks.performance baseline --rows 10000,100000 --widths narrow,medium,wide --dataset-types mixed --baselines shallow_copy,numeric_median_fill,duplicates,null_counts --warmups 1 --repetitions 5 --output benchmarks/results/performance/baseline
```

The `baseline` command accepts only names in `BASELINES` and writes the same
schema-versioned result format as `run`.

- [ ] **Step 7: Run comparison/render tests**

Run: `python -m pytest tests/performance/test_analysis_render.py -q --no-cov`

Expected: PASS.

- [ ] **Step 8: Commit analysis and rendering**

```bash
git add benchmarks/performance/baselines.py benchmarks/performance/analysis.py benchmarks/performance/render.py benchmarks/performance/cli.py tests/performance/test_analysis_render.py
git commit -m "bench: analyze and render scalability evidence"
```

### Task 7: CI-Safe Contracts, Large Workflow, Make Targets, and Documentation Shell

**Files:**
- Modify: `.gitignore`
- Modify: `Makefile`
- Create: `.github/workflows/performance-large.yml`
- Create: `docs/performance-investigation.md`
- Create: `tests/performance/test_contracts.py`
- Modify: `docs/benchmarks.md`

**Interfaces:**
- Consumes: Tasks 1-6 CLI and schemas.
- Produces: fast PR contracts, manual/scheduled large runs, reproducible local commands, and committed compact evidence paths.

- [ ] **Step 1: Write failing fast-contract tests**

```python
# tests/performance/test_contracts.py
from __future__ import annotations

import json

from benchmarks.performance.cli import main
from benchmarks.performance.render import render_report


def test_ci_sized_matrix_completes_and_renders(tmp_path) -> None:
    raw = tmp_path / "raw"
    assert main([
        "run", "--rows", "500", "--widths", "narrow,medium",
        "--configs", "default,conservative", "--report-modes", "false,true",
        "--warmups", "0", "--repetitions", "1", "--timeout", "120",
        "--output", str(raw),
    ]) == 0
    results = [json.loads(path.read_text()) for path in raw.glob("*.json")]
    assert len(results) == 8
    assert all(item["status"] == "completed" for item in results)
    rendered = render_report({"environment": results[0]["environment"], "results": results, "hypotheses": {}})
    assert "500" in rendered
    assert "narrow" in rendered and "medium" in rendered


def test_report_modes_preserve_behavioral_contract(tmp_path) -> None:
    assert main([
        "run", "--rows", "250", "--widths", "narrow", "--configs", "default",
        "--report-modes", "false,true", "--warmups", "0", "--repetitions", "1",
        "--timeout", "120", "--output", str(tmp_path),
    ]) == 0
    results = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert {item["case"]["return_report"] for item in results} == {False, True}
```

- [ ] **Step 2: Verify the fast-contract tests pass against completed tooling**

Run: `python -m pytest tests/performance -q --no-cov`

Expected: PASS. If this fails, fix the responsible Task 1-6 module before changing workflow files.

- [ ] **Step 3: Make compact evidence committable and raw results ignored**

Replace the broad benchmark result rule with:

```gitignore
# Benchmark runtime results: raw case files stay local; compact evidence is committed.
benchmarks/results/*
!benchmarks/results/performance/
benchmarks/results/performance/*
!benchmarks/results/performance/*-summary.json
!benchmarks/results/performance/*-report.md
```

- [ ] **Step 4: Add Make targets**

```make
.PHONY: performance-ci performance-baseline performance-profile performance-report

performance-ci:
	$(PY) -m pytest tests/performance -q --no-cov

performance-baseline:
	$(PY) -m benchmarks.performance run --output benchmarks/results/performance/baseline

performance-profile:
	$(PY) -m benchmarks.performance profile --rows 100000 --width medium --config default --report-mode true --output benchmarks/results/performance/baseline

performance-report:
	$(PY) -m benchmarks.performance analyze --input benchmarks/results/performance/baseline --output benchmarks/results/performance/baseline-summary.json
	$(PY) -m benchmarks.performance render --input benchmarks/results/performance/baseline-summary.json --output benchmarks/results/performance/baseline-report.md
```

- [ ] **Step 5: Add the manual/scheduled large workflow**

`.github/workflows/performance-large.yml` must run on `workflow_dispatch` and weekly schedule, install `.[dev,bench,ml]`, execute row counts `100000,500000,1000000`, widths `narrow,medium,wide`, configs `default,conservative,representation_off,statistical_off,explicit`, both report modes, one warm-up, five repetitions, then analyze/render and upload the whole performance results directory. Set `timeout-minutes: 180`; do not add it to required PR checks.

- [ ] **Step 6: Add the documentation shell with resolved architecture content**

Create `docs/performance-investigation.md` with these populated sections from the approved design, not empty headings:

- Executive summary stating that baseline measurement is in progress and no slowdown/improvement is claimed yet.
- Architecture and execution flow copied from the approved design.
- Supported Python/pandas versions and default configuration.
- Reproduction commands for `performance-ci`, `performance-baseline`, `performance-profile`, and `performance-report`.
- Measurement methodology and the 10%/2x-variability rule.
- A statement that baseline, profile, root-cause, rejected-hypothesis, before/after, memory, backend, documentation, risk, and verification sections are generated from authoritative JSON as their phases complete.

Update `docs/benchmarks.md` with links and commands for the new harness while retaining the existing CleanBench and strategic-report documentation.

- [ ] **Step 7: Run fast performance contracts and docs build**

Run: `python -m pytest tests/performance -q --no-cov`

Expected: PASS.

Run: `mkdocs build --strict`

Expected: PASS with no missing navigation or link warnings.

- [ ] **Step 8: Commit integration and documentation**

```bash
git add .gitignore Makefile .github/workflows/performance-large.yml docs/performance-investigation.md docs/benchmarks.md tests/performance/test_contracts.py
git commit -m "bench: integrate large performance investigation workflow"
```

### Task 8: Capture the Immutable Baseline and Decide the Optimization Plans

**Files:**
- Create: `benchmarks/results/performance/baseline-summary.json`
- Create: `benchmarks/results/performance/baseline-report.md`
- Modify: `docs/performance-investigation.md`
- Create: `docs/superpowers/specs/2026-07-11-freshdata-confirmed-performance-bottlenecks-design.md`
- Create only the applicable evidence-named plans from:
  `docs/superpowers/plans/2026-07-11-freshdata-optimize-correlation.md`,
  `2026-07-11-freshdata-optimize-engine-statistics.md`,
  `2026-07-11-freshdata-optimize-copy-pressure.md`,
  `2026-07-11-freshdata-optimize-dtype-conversions.md`,
  `2026-07-11-freshdata-optimize-reporting.md`,
  `2026-07-11-freshdata-optimize-optional-ml.md`, or
  `2026-07-11-freshdata-optimize-backend-conversion.md`.

**Interfaces:**
- Consumes: the clean `6f6c2fe` baseline worktree and Tasks 1-7 tooling.
- Produces: recorded commands/results, exact root causes and rejected hypotheses, and evidence-derived Phase 2 plans.

- [ ] **Step 1: Verify the baseline identity before running**

Run:

```bash
git rev-parse HEAD
git status --short
python -VV
python -c "import pandas, numpy, freshdata; print(pandas.__version__, numpy.__version__, freshdata.__version__)"
```

Expected: the benchmarked production sources match `6f6c2fe`; documentation/tooling commits may be present, but `git diff 6f6c2fe -- src/freshdata` must be empty. The only pre-existing untracked item may be `.venv-qa/` outside the isolated execution workspace.

- [ ] **Step 2: Run the required pandas baseline in bounded batches**

Run these commands separately so a large-case failure does not erase smaller evidence:

```bash
python -m benchmarks.performance run --rows 10000 --widths narrow,medium,wide --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 900 --output benchmarks/results/performance/baseline
python -m benchmarks.performance run --rows 100000 --widths narrow,medium,wide --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 1800 --output benchmarks/results/performance/baseline
python -m benchmarks.performance run --rows 500000 --widths narrow,medium,wide --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 3600 --output benchmarks/results/performance/baseline
python -m benchmarks.performance run --rows 1000000 --widths narrow,medium,wide --configs default,conservative,representation_off,statistical_off,explicit --report-modes false,true --warmups 1 --repetitions 5 --timeout 7200 --output benchmarks/results/performance/baseline
python -m benchmarks.performance run --rows 100000 --widths medium --dataset-types numeric,categorical,string,nullable,datetime,high_cardinality --configs default --report-modes false,true --warmups 1 --repetitions 5 --timeout 1800 --output benchmarks/results/performance/baseline
```

Expected: every case writes `completed`, `failed`, `timeout`, or `oom` JSON; no case disappears. Report all poor and incomplete outcomes.

- [ ] **Step 3: Profile the representative bottleneck slices**

Run:

```bash
python -m benchmarks.performance profile --rows 10000 --width wide --config default --report-mode true --output benchmarks/results/performance/baseline
python -m benchmarks.performance profile --rows 100000 --width medium --config default --report-mode true --output benchmarks/results/performance/baseline
python -m benchmarks.performance profile --rows 500000 --width medium --config default --report-mode false --output benchmarks/results/performance/baseline
python -m benchmarks.performance profile --rows 1000000 --width narrow --config default --report-mode true --output benchmarks/results/performance/baseline
python -m benchmarks.performance profile --rows 100000 --width medium --config aggressive --report-mode true --output benchmarks/results/performance/baseline
python -m benchmarks.performance profile --rows 100000 --width medium --config semantic --report-mode true --output benchmarks/results/performance/baseline
python -m benchmarks.performance profile --rows 10000 --width medium --config missforest --report-mode true --output benchmarks/results/performance/baseline
```

Expected: profile JSON includes exact function/file/line records, allocation records, stage totals, and operation counts for every completed profile.

- [ ] **Step 4: Measure comparable pandas component baselines**

Run:

```bash
python -m benchmarks.performance baseline --rows 10000,100000,500000,1000000 --widths narrow,medium,wide --dataset-types mixed --baselines shallow_copy,numeric_median_fill,duplicates,null_counts --warmups 1 --repetitions 5 --timeout 3600 --output benchmarks/results/performance/baseline
```

Expected: component comparisons are labelled by exact operation; no result claims to replicate the full balanced decision/audit pipeline.

- [ ] **Step 5: Analyze and render compact evidence**

Run:

```bash
python -m benchmarks.performance analyze --input benchmarks/results/performance/baseline --output benchmarks/results/performance/baseline-summary.json
python -m benchmarks.performance render --input benchmarks/results/performance/baseline-summary.json --output benchmarks/results/performance/baseline-report.md
```

Expected: the summary includes environment, commands, all statuses, median/min/max/CV, peak RSS/Python memory, ratios, stage shares, exact hot functions/lines, and eight hypothesis decisions.

- [ ] **Step 6: Update the investigation report from authoritative evidence**

Copy generated tables and findings without hand-editing numeric values. Populate these sections in `docs/performance-investigation.md`:

1. Executive summary limited to baseline findings.
2. Reproduction commands.
3. Baseline benchmark table.
4. Profiling findings with exact functions/files/lines.
5. Confirmed root causes.
6. Rejected hypotheses.
12. Baseline peak-memory table.
13. Remaining baseline bottlenecks.
16. Baseline risks and limitations.

Leave before/after optimization sections explicitly marked `not yet applicable: no production optimization has been implemented`, which is a resolved status rather than an unknown placeholder.

- [ ] **Step 7: Write the evidence-derived Phase 2 design and plans**

Create `docs/superpowers/specs/2026-07-11-freshdata-confirmed-performance-bottlenecks-design.md` containing only hypotheses classified `candidate` by `baseline-summary.json`. For each candidate, include its exact functions/lines, affected cases, measured time/memory fraction, behavioral invariants, proposed smallest change, alternative rejected approaches, and benchmark acceptance case.

Then run the brainstorming approval gate for that design. After approval, create one detailed TDD plan per independently rejectable root cause. Do not create or execute a production optimization plan for any hypothesis classified `rejected` or `insufficient_evidence`.

- [ ] **Step 8: Verify and commit baseline evidence**

Run:

```bash
python -m pytest tests/performance tests/benchmark -q --no-cov
ruff check benchmarks/performance tests/performance
mypy benchmarks/performance
mkdocs build --strict
git diff --check
```

Expected: all commands pass. Raw case files remain ignored; compact summary/report, investigation document, and evidence-derived design/plans are staged.

```bash
git add benchmarks/results/performance/baseline-summary.json benchmarks/results/performance/baseline-report.md docs/performance-investigation.md docs/superpowers/specs/2026-07-11-freshdata-confirmed-performance-bottlenecks-design.md docs/superpowers/plans
git commit -m "bench: record FreshData scalability baseline"
```

## Phase 1 Completion Gate

Phase 1 is complete only when:

- The required 10k, 100k, 500k, and 1M row matrix has a recorded status for every requested width/config/report case.
- Median, minimum, maximum, variability, peak RSS, Python allocations, input ratio, and comparable pandas ratios are recorded.
- Exact functions, files, and lines are recorded for the largest time and allocation costs.
- Context/cache, correlation, missing, outlier, role, dtype, duplicate, audit, report, ML/semantic, conversion, and backend costs are separated where exercised.
- Copy/conversion/scan counts are explicitly labelled observed calls.
- Confirmed root causes and rejected hypotheses are evidence-backed.
- No production cleaning code differs from `6f6c2fe`.
- The Phase 2 design includes only confirmed causes and has been presented for user approval.

Do not claim the overall FreshData performance goal complete at this gate. The active goal remains open through production optimization, before/after benchmarks, backend recommendations, documentation correction, full compatibility verification, and the final 17-deliverable audit.
