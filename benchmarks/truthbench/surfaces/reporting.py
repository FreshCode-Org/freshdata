# ruff: noqa: E501
"""TruthBench observations for reports, exports, CLI, and trust surfaces."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

import freshdata as fd

from ..privacy import SinkScanner
from .base import ExceptionDetails, SurfaceAdapter, SurfaceObservation, register_adapter


def _plain(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value
    method = getattr(value, "to_dict", None)
    if callable(method):
        with contextlib.suppress(Exception):
            return method()
    return value


class ReportingAdapter(SurfaceAdapter):
    """Capture all externally-visible reporting and quality-ops outputs."""

    name = "reporting"

    def scanner_for(self, fixture: Any) -> SinkScanner:
        return SinkScanner.from_fixture(fixture, key=b"truthbench-fixed-reporting-key")

    def _reports(
        self,
        frame: pd.DataFrame,
        pristine: pd.DataFrame,
        sensitive_columns: tuple[str, ...] = (),
    ) -> tuple[pd.DataFrame, Any, dict[str, Any], dict[str, Any]]:
        cleaned, clean_report = fd.clean(
            frame, return_report=True, sensitive_columns=sensitive_columns
        )
        quality = fd.build_quality_report(frame, cleaned, clean_report)
        debt_out, debt = fd.evaluate_quality_debt(frame, debt_policy="warn")
        insight = fd.insight_report(frame, clean_report=clean_report, cleaned_df=cleaned)
        compliance = fd.generate_compliance_report(clean_report, ["gdpr_30"], dataframe=frame)
        stakeholder = fd.stakeholder_summary(clean_report)
        reports = {
            "quality": quality,
            "debt": debt,
            "insight": insight,
            "compliance": compliance,
            "stakeholder": stakeholder,
        }
        rendered: dict[str, Any] = {
            "to_dict": {name: _plain(report) for name, report in reports.items()},
            "to_frame": clean_report.to_frame()
            if hasattr(clean_report, "to_frame")
            else pd.DataFrame(),
            "to_findings": [f.to_dict() for f in clean_report.to_findings()],
            "json": json.dumps(_plain(clean_report), default=str, sort_keys=True),
            "markdown": quality.to_markdown(),
            "html": stakeholder.to_html(),
            "plain": str(insight),
            "peel": repr(clean_report),
        }
        destructive = pd.DataFrame(
            {str(c): [None] * len(frame) for c in frame.columns}, index=frame.index
        )
        trust = {
            "pristine": fd.compute_trust_score(pristine).overall,
            "adversarial": fd.compute_trust_score(frame).overall,
            "cleaned": fd.compute_trust_score(cleaned).overall,
            "destructive": fd.compute_trust_score(destructive).overall,
        }
        return (
            cleaned,
            clean_report,
            {name: _plain(value) for name, value in reports.items()},
            {"rendered": rendered, "trust": trust, "debt_output": debt_out},
        )

    def observe(self, fixture: Any, context: Any) -> SurfaceObservation:
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            frame = getattr(fixture, "frame", fixture)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("reporting adapters require a pandas DataFrame")
            scanner = self.scanner_for(fixture)
            pristine = getattr(fixture, "pristine", frame)
            if not isinstance(pristine, pd.DataFrame):
                pristine = frame
            operation = str(
                context.get("operation", "reports") if isinstance(context, Mapping) else "reports"
            )
            sensitive = tuple(
                context.get("sensitive_columns", ())
                if isinstance(context, Mapping)
                else ()
            )
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                cleaned, clean_report, reports, sinks = self._reports(
                    frame, pristine, sensitive
                )
                if operation == "exports":
                    from freshdata.enterprise.cli import main as cli_main

                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        output = fd.export_quality_ops(
                            clean_report,
                            dbt_path=str(root / "tests.yml"),
                            gx_path=str(root / "suite.json"),
                            exception_table_path=str(root / "exceptions.csv"),
                            df=frame,
                        )
                        cli_stdout, cli_stderr = io.StringIO(), io.StringIO()
                        with (
                            contextlib.redirect_stdout(cli_stdout),
                            contextlib.redirect_stderr(cli_stderr),
                        ):
                            try:
                                cli_main(["--help"])
                            except SystemExit as exc:
                                if exc.code not in (0, None):
                                    raise
                        sinks["exports"] = {
                            "quality_ops": _plain(output),
                            "dbt": (root / "tests.yml").read_text(),
                            "great_expectations": (root / "suite.json").read_text(),
                            "exceptions": (root / "exceptions.csv").read_text(),
                            "cli": {
                                "stdout": cli_stdout.getvalue(),
                                "stderr": cli_stderr.getvalue(),
                            },
                        }
            safe_sinks = scanner.redact(sinks)
            return SurfaceObservation(
                output_frame=scanner.redact(cleaned),
                raw_decisions=scanner.redact(reports),
                audit_sinks=safe_sinks,
                trust=safe_sinks["trust"],
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


register_adapter(ReportingAdapter)
__all__ = ["ReportingAdapter"]
