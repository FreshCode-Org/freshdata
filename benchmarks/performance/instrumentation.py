from __future__ import annotations

import cProfile
import pstats
import tracemalloc
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from types import TracebackType
from typing import Callable
from unittest.mock import patch

import pandas as pd

import freshdata as fd

from .datasets import DatasetSpec, make_mixed_frame
from .models import BenchmarkCase


@dataclass(frozen=True)
class ProfileResult:
    functions: list[dict[str, object]]
    allocations: list[dict[str, object]]
    stages: dict[str, float]
    operations: dict[str, int]


class OperationCounter:
    """Observe Python-level pandas method calls made inside the context."""

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

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter(dict.fromkeys(self.METHODS, 0))
        self._stack: ExitStack | None = None

    def __enter__(self) -> OperationCounter:
        if self._stack is not None:
            raise RuntimeError("OperationCounter is already active")
        stack = ExitStack()
        try:
            for key, (owner, method_name) in self.METHODS.items():
                original = getattr(owner, method_name)

                @wraps(original)
                def observed(
                    instance: object,
                    *args: object,
                    _key: str = key,
                    _original: Callable[..., object] = original,
                    **kwargs: object,
                ) -> object:
                    self.counts[_key] += 1
                    return _original(instance, *args, **kwargs)

                stack.enter_context(patch.object(owner, method_name, observed))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._stack is not None
        try:
            self._stack.__exit__(exc_type, exc_value, traceback)
        finally:
            self._stack = None


STAGE_RULES = {
    "context": ("engine/context.py",),
    "engine_cache": ("engine/cache.py",),
    "correlation": ("numeric_corr_matrix", "corrwith"),
    "missing": ("engine/missing.py", "steps/missing.py"),
    "outliers": ("engine/outliers.py", "steps/outliers.py"),
    "role_inference": ("infer_role", "build_context"),
    "dtype_repair": ("steps/dtypes.py",),
    "duplicates": ("steps/duplicates.py",),
    "audit_events": ("report.py", "cleanreport.add"),
    "report_finalization": ("cleaner.py", "memory_bytes"),
    "semantic_ml": ("semantic/", "imputation/missforest.py", "sklearn/"),
    "backend_conversion": ("adapters/", "execution/backends/"),
}

_EXACT_FUNCTION_STAGES = {
    "numeric_corr_matrix": "correlation",
    "corr": "correlation",
    "corrwith": "correlation",
    "infer_role": "role_inference",
    "build_context": "role_inference",
}


def _stage_for(filename: str, function: str) -> str | None:
    normalized_function = function.lower()
    exact_stage = _EXACT_FUNCTION_STAGES.get(normalized_function)
    if exact_stage is not None:
        return exact_stage

    normalized_path = filename.replace("\\", "/").lower()
    normalized = f"{normalized_path}:{normalized_function}"
    exact_rules = _EXACT_FUNCTION_STAGES.keys()
    for stage, rules in STAGE_RULES.items():
        if any(rule not in exact_rules and rule in normalized for rule in rules):
            return stage
    return None


def _function_records(
    stats: pstats.Stats,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    records: list[dict[str, object]] = []
    stages = dict.fromkeys(STAGE_RULES, 0.0)
    total = 0.0
    for (filename, line, function), (
        _primitive,
        calls,
        self_time,
        cumulative,
        _callers,
    ) in stats.stats.items():
        record = {
            "file": filename,
            "line": line,
            "function": function,
            "self_seconds": self_time,
            "cumulative_seconds": cumulative,
            "calls": calls,
        }
        records.append(record)
        total += self_time
        stage = _stage_for(filename, function)
        if stage is not None:
            stages[stage] += self_time
    records.sort(key=lambda item: float(item["cumulative_seconds"]), reverse=True)
    stages["total"] = total
    return records[:100], stages


def _allocation_records(
    differences: list[tracemalloc.StatisticDiff],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for statistic in differences:
        if statistic.size_diff <= 0 or statistic.count_diff <= 0:
            continue
        frame = statistic.traceback[0]
        normalized = frame.filename.replace("\\", "/")
        if "/freshdata/" not in normalized and "/benchmarks/performance/" not in normalized:
            continue
        records.append(
            {
                "file": frame.filename,
                "line": frame.lineno,
                "bytes": statistic.size_diff,
                "count": statistic.count_diff,
            }
        )
        if len(records) == 100:
            break
    return records


def profile_case(case: BenchmarkCase) -> ProfileResult:
    frame = make_mixed_frame(
        DatasetSpec(
            rows=case.rows,
            width=case.width,
            seed=case.seed,
            dataset_type=case.dataset_type,
        )
    )
    profiler = cProfile.Profile()
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    try:
        before = tracemalloc.take_snapshot()
        with OperationCounter() as counter:
            profiler.enable()
            try:
                fd.clean(
                    frame,
                    config=case.options,
                    return_report=case.return_report,
                    engine=case.backend,
                    output_format=case.output_format,
                )
            finally:
                profiler.disable()
        after = tracemalloc.take_snapshot()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    functions, stages = _function_records(pstats.Stats(profiler))
    return ProfileResult(
        functions=functions,
        allocations=_allocation_records(after.compare_to(before, "lineno")),
        stages=stages,
        operations=dict(counter.counts),
    )
