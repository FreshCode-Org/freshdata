# ruff: noqa: E501, PLR0915
"""Observation adapters for read-only FreshData validation and text surfaces."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping
from typing import Any

import pandas as pd

import freshdata as fd

from ..privacy import SinkScanner
from .base import ExceptionDetails, SurfaceAdapter, SurfaceObservation, register_adapter


def _get(ctx: Any, key: str, default: Any = None) -> Any:
    return ctx.get(key, default) if isinstance(ctx, Mapping) else getattr(ctx, key, default)


def _safe(value: Any, fixture: Any) -> Any:
    try:
        canaries = getattr(fixture, "pii_canaries", None)
        return SinkScanner.from_canaries(canaries).redact(value) if canaries else value
    except Exception:
        return value


def _plain(value: Any) -> Any:
    fn = getattr(value, "to_dict", None)
    if callable(fn):
        with contextlib.suppress(Exception):
            return fn()
    return value


class ValidationAdapter(SurfaceAdapter):
    name = "validation"

    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        frame = getattr(fixture, "frame", fixture)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("validation adapters require a pandas DataFrame")
        operation = str(_get(context, "operation", "validate"))
        options = dict(_get(context, "options", {}) or {})
        if isinstance(context, Mapping):
            options.update(
                {k: v for k, v in context.items() if k not in {"operation", "options", "schema"}}
            )
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if operation in {"validate_fields", "fields"}:
                    schema = _get(context, "schema", options.pop("schema", None))
                    report = fd.validate_fields(frame, schema=schema, **options)
                    decisions = {
                        "findings": _safe(_plain(report), fixture),
                        "issues": _safe([_plain(i) for i in report.issues], fixture),
                    }
                    output = frame.copy(deep=True)
                elif operation in {"apply_field_policy", "field_policy"}:
                    schema = _get(context, "schema", options.pop("schema", None))
                    report = fd.validate_fields(frame, schema=schema, **options)
                    result = fd.apply_field_policy(frame, report)
                    output = result.accepted
                    decisions = {
                        "findings": _safe(_plain(report), fixture),
                        "policy_result": _safe(_plain(result), fixture),
                        "audit": _safe(result.audit, fixture),
                    }
                elif (
                    operation in {"suite", "ValidationSuite", "validate"}
                    and _get(context, "schema") is not None
                ):
                    schema = _get(context, "schema")
                    report = fd.validate_fields(frame, schema=schema, **options)
                    decisions = {
                        "findings": _safe(_plain(report), fixture),
                        "issues": _safe([_plain(i) for i in report.issues], fixture),
                    }
                    output = frame.copy(deep=True)
                elif operation in {"suite", "ValidationSuite", "validate"}:
                    suite = _get(context, "suite")
                    if suite is None:
                        text = (
                            _get(context, "context")
                            or _get(context, "policy")
                            or getattr(fixture, "policy", None)
                        )
                        if isinstance(text, str):
                            result = fd.validate(frame, context=text)
                        else:
                            result = fd.validate(frame, **options)
                    else:
                        result = fd.validate(frame, suite=suite)
                    decisions = {
                        "findings": _safe(_plain(result), fixture),
                        "passed": getattr(result, "passed", None),
                    }
                    output = frame.copy(deep=True)
                elif operation in {"compile_context", "context_compile"}:
                    policy = fd.compile_context(str(_get(context, "text", "")), df=frame)
                    decisions = {
                        "policy": _safe(_plain(policy), fixture),
                        "policy_hash": getattr(policy, "policy_hash", None),
                    }
                    output = frame.copy(deep=True)
                elif operation in {"text", "clean_text"}:
                    output, report = fd.clean_text(frame, **options)
                    decisions = {"text_report": _safe(_plain(report), fixture)}
                elif operation in {"lint", "lint_text_encoding"}:
                    report = fd.lint_text_encoding(frame, **options)
                    decisions = {"lint_report": _safe(_plain(report), fixture)}
                    output = frame.copy(deep=True)
                elif operation in {
                    "semantic",
                    "semantic_assist",
                    "semantic_review",
                    "semantic_auto",
                }:
                    mode = (
                        operation.removeprefix("semantic_")
                        if operation != "semantic"
                        else _get(context, "mode", "assist")
                    )
                    output, report = fd.clean(
                        frame, semantic_mode=mode, return_report=True, **options
                    )
                    decisions = {
                        "report": _safe(_plain(report), fixture),
                        "report_actions": _safe(
                            [_plain(a) for a in getattr(report, "actions", ())], fixture
                        ),
                    }
                elif operation in {"domain_validator", "domain_validate"}:
                    domain = _get(context, "domain", getattr(fixture, "domain", None))
                    output, report = fd.clean(frame, domain=domain, return_report=True, **options)
                    decisions = {"domain": domain, "report": _safe(_plain(report), fixture)}
                elif operation == "run_semantic_validation":
                    configs = _get(context, "configs", options.pop("configs", ()))
                    report = fd.run_semantic_validation(frame, configs)
                    decisions = {"semantic_report": _safe(_plain(report), fixture)}
                    output = frame.copy(deep=True)
                else:
                    raise ValueError(f"unknown validation operation: {operation!r}")
            sinks = {
                "input_snapshot": _safe(frame.copy(deep=True), fixture),
                "output": _safe(output, fixture),
            }
            sinks.update(decisions)
            return SurfaceObservation(
                output_frame=output,
                raw_decisions=decisions,
                audit_sinks=sinks,
                backend_disclosure={"requested": "pandas", "actual": "pandas"},
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )
        except Exception as exc:
            message: Any = _safe(str(exc), fixture)
            if not isinstance(message, str):
                message = "[REDACTED]"
            return SurfaceObservation(
                unexpected_exception=ExceptionDetails(type(exc).__name__, message),
                audit_sinks={"input_snapshot": _safe(frame.copy(deep=True), fixture)},
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )


for _name in (
    "FieldValidationAdapter",
    "ValidationSuiteAdapter",
    "ContextAdapter",
    "DomainValidationAdapter",
    "TextAdapter",
    "StreamingValidationAdapter",
):
    globals()[_name] = ValidationAdapter
FieldValidationAdapter = ValidationAdapter
ValidationSuiteAdapter = ValidationAdapter
ContextAdapter = ValidationAdapter
DomainValidationAdapter = ValidationAdapter
TextAdapter = ValidationAdapter
StreamingValidationAdapter = ValidationAdapter
register_adapter(ValidationAdapter)
__all__ = [
    "ValidationAdapter",
    "FieldValidationAdapter",
    "ValidationSuiteAdapter",
    "ContextAdapter",
    "DomainValidationAdapter",
    "TextAdapter",
    "StreamingValidationAdapter",
]
