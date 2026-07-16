# ruff: noqa: E501
"""TruthBench observation adapter for the provider-free experimental Copilot."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import pandas as pd

from freshdata.experimental.ai_copilot import _build_prompt, analyze_dataset

from ..privacy import SinkScanner
from .base import ExceptionDetails, SurfaceAdapter, SurfaceObservation, register_adapter


class CopilotAdapter(SurfaceAdapter):
    """Run the deterministic Copilot path and retain all report sinks safely."""

    name = "copilot"

    def scanner_for(self, fixture: Any) -> SinkScanner:
        return SinkScanner.from_fixture(fixture, key=b"truthbench-fixed-copilot-key")

    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            frame = getattr(fixture, "frame", fixture)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Copilot adapter requires a pandas DataFrame")
            scanner = self.scanner_for(fixture)
            sensitive = tuple(
                sorted({c.column for c in getattr(fixture, "cells", ()) if c.sensitive})
            )
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                report = analyze_dataset(
                    frame, provider=None, sensitive_columns=sensitive
                )
            # The prompt is constructed exactly as a provider call would see it,
            # even though this adapter intentionally supplies no provider.
            prompt = _build_prompt(report.goal, report.model_context)
            sinks = scanner.redact(
                {
                    "prompt": prompt,
                    "prompt_digest": scanner.digest_for(prompt),
                    "prompt_source": "freshdata.experimental.ai_copilot._build_prompt",
                    "model_context": report.model_context,
                    "recommended_code": report.recommended_code,
                    "audit": report.audit,
                    "narrative": report.narrative,
                    "rendered": {
                        "dict": report.to_dict(),
                        "json": report.to_json(),
                        "plain": str(report),
                        "html": report._repr_html_(),
                    },
                }
            )
            return SurfaceObservation(
                output_frame=scanner.redact(frame.copy(deep=True)),
                raw_decisions=scanner.redact(report.to_dict()),
                audit_sinks=sinks,
                generated_code=scanner.redact(report.recommended_code),
                backend_disclosure={"requested": "pandas", "actual": "pandas"},
                captured_stdout=scanner.redact(stdout.getvalue()),
                captured_stderr=scanner.redact(stderr.getvalue()),
            )
        except Exception as exc:
            scanner = self.scanner_for(fixture)
            return SurfaceObservation(
                unexpected_exception=ExceptionDetails(
                    type(exc).__name__, str(scanner.redact(str(exc)))
                ),
                captured_stdout=scanner.redact(stdout.getvalue()),
                captured_stderr=scanner.redact(stderr.getvalue()),
            )


register_adapter(CopilotAdapter)
__all__ = ["CopilotAdapter"]
