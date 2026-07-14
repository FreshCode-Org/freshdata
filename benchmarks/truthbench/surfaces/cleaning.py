# ruff: noqa: E501, PLR0915
"""Observation adapters for FreshData's cleaning-oriented public APIs.

The adapters intentionally depend on :mod:`freshdata`'s public namespace only.
They collect evidence; grading is performed later by TruthBench normalisation.
"""

from __future__ import annotations

import contextlib
import io
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

import freshdata as fd

from ..privacy import SinkScanner
from .base import ExceptionDetails, SurfaceAdapter, SurfaceObservation, register_adapter


def _value(context: Any, key: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(key, default)
    return getattr(context, key, default)


def _frame(fixture: Any) -> pd.DataFrame:
    frame = getattr(fixture, "frame", fixture)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("TruthBench cleaning adapters require a pandas DataFrame")
    return frame


def _safe(value: Any, fixture: Any) -> Any:
    try:
        canaries = getattr(fixture, "pii_canaries", None)
        if canaries:
            return SinkScanner.from_canaries(canaries).redact(value)
    except Exception:
        pass
    return value


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, tuple, dict)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        with contextlib.suppress(Exception):
            return to_dict()
    return repr(value)


def _report_evidence(report: Any, fixture: Any) -> tuple[dict[str, Any], dict[str, Any], Any]:
    if report is None:
        return {}, {}, None
    payload = _plain(report)
    actions = getattr(report, "actions", ())
    action_payload = [_plain(a) for a in actions] if actions is not None else []
    raw = {"report": payload, "report_actions": _safe(action_payload, fixture)}
    for name in ("coerced_cells", "findings", "recommendations", "domain_repairs", "repairs"):
        if hasattr(report, name):
            raw[name] = _safe(_plain(getattr(report, name)), fixture)
    trust = {
        k: getattr(report, k)
        for k in ("trust_score", "domain_trust_score", "trust_before", "trust_after")
        if hasattr(report, k)
    }
    return raw, trust, report


class CleaningAdapter(SurfaceAdapter):
    """Dispatch a fixture through one of the public cleaning entry points."""

    name = "cleaning"

    def __init__(self, operation: str = "clean") -> None:
        self.operation = operation

    def _run(
        self, frame: pd.DataFrame, fixture: Any, context: Any
    ) -> tuple[Any, Any, dict[str, Any]]:
        operation = str(_value(context, "operation", self.operation))
        options = dict(_value(context, "options", {}) or {})
        if isinstance(context, Mapping):
            options.update(
                {
                    k: v
                    for k, v in context.items()
                    if k not in {"operation", "options", "partitions", "steps", "domain"}
                }
            )
        if operation in {"clean", "fd.clean"}:
            result = fd.clean(frame, return_report=True, **options)
            return result[0], result[1], {}
        if operation in {"cleaner", "Cleaner.clean"}:
            result = fd.Cleaner(**options).clean(frame, report=True)
            return result[0], result[1], {}
        if operation == "clean_csv":
            path = _value(context, "path")
            if path is None:
                with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as tmp:
                    frame.to_csv(tmp.name, index=False)
                    path = tmp.name
            result = fd.clean_csv(Path(path), return_report=True, **options)
            return result[0], result[1], {"path": str(path)}
        if operation in {"pipeline", "Pipeline.run"}:
            pipe = fd.pipeline()
            steps = _value(context, "steps", ("normalize_columns",))
            for step in steps:
                method = getattr(pipe, str(step))
                pipe = method()
            result = pipe.run(frame, return_report=True)
            return result[0], result[1], {"pipeline": pipe.to_dict()}
        if operation in {"suggest_plan", "plan"}:
            plan = fd.suggest_plan(frame, **options)
            return (
                frame.copy(deep=True),
                None,
                {
                    "plan": _safe(_plain(plan), fixture),
                    "plan_hash": getattr(plan, "plan_hash", None),
                },
            )
        if operation in {"apply_plan", "apply"}:
            plan = _value(context, "plan")
            if plan is None:
                plan = fd.suggest_plan(frame, **options)
            out, report = fd.apply_plan(frame, plan)
            return (
                out,
                report,
                {
                    "plan": _safe(_plain(plan), fixture),
                    "plan_hash": getattr(plan, "plan_hash", None),
                },
            )
        if operation in {"domain", "clean_domain"}:
            domain = _value(context, "domain", getattr(fixture, "domain", None))
            result = fd.clean(frame, domain=domain, return_report=True, **options)
            return result[0], result[1], {"domain": domain}
        if operation == "clean_timeseries":
            result = fd.clean_timeseries(frame, return_report=True, **options)
            return result[0], result[1], {}
        if operation in {"streaming", "StreamingCleaner"}:
            partitions = tuple(_value(context, "partitions", (8, 8)))
            reports: list[Any] = []
            outputs: list[pd.DataFrame] = []
            for partition in (partitions, (5, 5, 6)):
                cleaner = fd.StreamingCleaner(**options)
                start = 0
                for size in partition:
                    batch, report = cleaner.clean_batch(frame.iloc[start : start + size])
                    outputs.append(batch)
                    reports.append(_plain(report))
                    start += size
                with contextlib.suppress(Exception):
                    cleaner.finalize()
            return (
                pd.concat(outputs[: len(partitions)], axis=0),
                reports[-1] if reports else None,
                {"streaming_partitions": reports},
            )
        raise ValueError(f"unknown cleaning operation: {operation!r}")

    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            frame = _frame(fixture)
        except Exception as exc:
            message: Any = _safe(str(exc), fixture)
            if not isinstance(message, str):
                message = "[REDACTED]"
            return SurfaceObservation(
                unexpected_exception=ExceptionDetails(type(exc).__name__, message),
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                output, report, extra = self._run(frame, fixture, context)
            raw, trust, _ = _report_evidence(report, fixture)
            raw.update(extra)
            sinks = {
                "input_snapshot": _safe(frame.copy(deep=True), fixture),
                "output": _safe(output, fixture),
            }
            sinks.update(
                {
                    k: v
                    for k, v in raw.items()
                    if k in {"report", "findings", "coerced_cells", "plan"}
                }
            )
            return SurfaceObservation(
                output_frame=output,
                raw_decisions=raw,
                audit_sinks=sinks,
                trust=trust,
                backend_disclosure={"requested": "pandas", "actual": "pandas"},
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )
        except Exception as exc:
            message = _safe(str(exc), fixture)
            if not isinstance(message, str):
                message = "[REDACTED]"
            return SurfaceObservation(
                unexpected_exception=ExceptionDetails(type(exc).__name__, message),
                audit_sinks={"input_snapshot": _safe(frame.copy(deep=True), fixture)},
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )


for _name in (
    "CleanAdapter",
    "CleanerAdapter",
    "CsvAdapter",
    "PipelineAdapter",
    "PlanAdapter",
    "StreamingAdapter",
    "DomainAdapter",
):
    globals()[_name] = CleaningAdapter

CleanAdapter = CleaningAdapter
CleanerAdapter = CleaningAdapter
CsvAdapter = CleaningAdapter
PipelineAdapter = CleaningAdapter
PlanAdapter = CleaningAdapter
StreamingAdapter = CleaningAdapter
DomainAdapter = CleaningAdapter

register_adapter(CleaningAdapter)

__all__ = [
    "CleaningAdapter",
    "CleanAdapter",
    "CleanerAdapter",
    "CsvAdapter",
    "PipelineAdapter",
    "PlanAdapter",
    "StreamingAdapter",
    "DomainAdapter",
]
