# ruff: noqa: E501
"""TruthBench observations for FreshData privacy APIs."""

from __future__ import annotations

import contextlib
import io
from collections.abc import Mapping
from typing import Any

import pandas as pd

import freshdata as fd

from ..privacy import SinkScanner
from .base import ExceptionDetails, SurfaceAdapter, SurfaceObservation, register_adapter


def _get(context: Any, key: str, default: Any = None) -> Any:
    return (
        context.get(key, default)
        if isinstance(context, Mapping)
        else getattr(context, key, default)
    )


def _plain(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value
    method = getattr(value, "to_dict", None)
    if callable(method):
        with contextlib.suppress(Exception):
            return method()
    return value


class PrivacyAdapter(SurfaceAdapter):
    """Execute privacy operations while recording their public evidence."""

    name = "privacy"

    def scanner_for(self, fixture: Any) -> SinkScanner:
        return SinkScanner.from_fixture(fixture, key=b"truthbench-fixed-privacy-key")

    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            frame = getattr(fixture, "frame", fixture)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("privacy adapters require a pandas DataFrame")
            scanner = self.scanner_for(fixture)
            operation = str(_get(context, "operation", "detect_pii"))
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                if operation == "detect_pii":
                    result, output = fd.detect_pii(frame), frame.copy(deep=True)
                elif operation == "anonymize":
                    columns = tuple(str(c) for c in _get(context, "columns", ("email",)))
                    rule = fd.MaskingRule(
                        name="truthbench-fixed",
                        columns=columns,
                        strategy="hash",
                        salt="truthbench-fixed",
                    )
                    output, result = fd.anonymize(frame, rules=(rule,), audit_include_pii=False)
                elif operation == "anonymize_default_random":
                    rule = fd.MaskingRule(
                        name="truthbench-random", columns=("email",), strategy="hash"
                    )
                    output, result = fd.anonymize(frame, rules=(rule,), audit_include_pii=False)
                elif operation == "privacy_policy":
                    rule = fd.PrivacyRule(
                        id="truthbench-redact", columns=("email",), action="redact"
                    )
                    policy = fd.PrivacyPolicy(rules=(rule,))
                    output, result = fd.apply_privacy_policy(
                        frame, policy, audit_include_pii=False
                    )
                elif operation == "k_anonymity":
                    identifiers = list(_get(context, "quasi_identifiers", (frame.columns[0],)))
                    result, output = (
                        fd.check_k_anonymity(frame, identifiers, k=2),
                        frame.copy(deep=True),
                    )
                else:
                    raise ValueError(f"unknown privacy operation: {operation!r}")
            payload = scanner.redact(_plain(result))
            sinks = scanner.redact(
                {"input_snapshot": frame.copy(deep=True), "result": payload, "output": output}
            )
            if operation == "anonymize_default_random":
                # Do not retain the random salt; retain only the security-relevant fact.
                sinks["randomness"] = {"default_salt_generated": True}
            return SurfaceObservation(
                output_frame=output,
                raw_decisions={operation: payload},
                audit_sinks=sinks,
                backend_disclosure={"requested": "pandas", "actual": "pandas"},
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )
        except Exception as exc:
            scanner = self.scanner_for(fixture)
            return SurfaceObservation(
                unexpected_exception=ExceptionDetails(
                    type(exc).__name__, str(scanner.redact(str(exc)))
                ),
                captured_stdout=stdout.getvalue(),
                captured_stderr=stderr.getvalue(),
            )


register_adapter(PrivacyAdapter)

__all__ = ["PrivacyAdapter"]
