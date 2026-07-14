# ruff: noqa: E501
"""TruthBench observation adapter for the provider-free experimental Copilot."""

from __future__ import annotations

import contextlib
import io
from typing import Any

import pandas as pd

from freshdata.experimental.ai_copilot import analyze_dataset

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
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                report = analyze_dataset(frame, provider=None)
            prompt = "provider=None: deterministic local analysis; model context is captured below"
            sinks = scanner.redact(
                {
                    "prompt": prompt,
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
                output_frame=frame.copy(deep=True),
                raw_decisions=scanner.redact(report.to_dict()),
                audit_sinks=sinks,
                generated_code=report.recommended_code,
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


register_adapter(CopilotAdapter)
__all__ = ["CopilotAdapter"]
