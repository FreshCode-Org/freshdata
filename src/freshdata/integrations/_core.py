"""Shared trust-gate primitives for :mod:`freshdata.integrations`.

Every orchestrator adapter (Dagster, Airflow, dbt) runs the same operation: clean a
DataFrame, score the cleaned result, and decide whether its data quality clears a
threshold. :func:`evaluate_trust_gate` *is* that operation; the adapters merely wrap
the resulting :class:`TrustGateResult` in their framework's native objects.

Nothing here imports an orchestration framework, so importing
``freshdata.integrations`` stays cheap and free of optional dependencies. The
0-100 trust score comes from the enterprise layer's
:func:`~freshdata.enterprise.compute_trust_score`; a compliance artifact is folded
in *only* when :mod:`freshdata.compliance` is importable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

import freshdata as fd
from freshdata.enterprise import compute_trust_score

if TYPE_CHECKING:  # annotations only — keep import cost down at module load
    import pandas as pd

    from freshdata import CleanConfig, CleanReport

logger = logging.getLogger("freshdata.integrations")

#: How an adapter should react when the gate does not pass.
OnLowScore = Literal["warn", "fail", "skip"]


class TrustGateError(RuntimeError):
    """Raised when a failing trust gate is configured to hard-fail.

    The framework-agnostic core never raises this itself; adapters (or direct
    callers) raise it — or translate it — when ``on_low_score == "fail"``.
    """


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string ending in ``Z``."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class TrustGateResult:
    """Outcome of a trust gate over a cleaned DataFrame.

    ``passed`` is ``trust_score >= threshold``. ``on_low_score`` records the caller's
    desired reaction; :attr:`should_fail` / :attr:`should_skip` translate it into a
    single decision the adapters act on, so the policy lives in one place.
    """

    trust_score: float
    grade: str
    threshold: float
    passed: bool
    high_risk_count: int
    row_count_in: int
    row_count_out: int
    generated_utc: str
    on_low_score: OnLowScore = "warn"
    #: Full audit payload, populated only when ``publish_full_report=True``. Holds
    #: the ``clean_report`` dict and, when :mod:`freshdata.compliance` is present, a
    #: ``compliance`` bundle dict.
    report_dict: dict[str, Any] | None = None

    @property
    def should_fail(self) -> bool:
        """True when a failing gate is configured to hard-fail."""
        return not self.passed and self.on_low_score == "fail"

    @property
    def should_skip(self) -> bool:
        """True when a failing gate is configured to skip downstream work."""
        return not self.passed and self.on_low_score == "skip"

    @property
    def message(self) -> str:
        """A human-readable one-line summary of the gate decision."""
        verb = "passed" if self.passed else "failed"
        return (
            f"freshdata trust gate {verb}: score {self.trust_score:.1f} "
            f"(grade {self.grade}) vs threshold {self.threshold:.1f}; "
            f"{self.high_risk_count} high-risk action(s); "
            f"{self.row_count_in} -> {self.row_count_out} rows."
        )

    def as_metadata(self) -> dict[str, Any]:
        """Return a flat ``freshdata/*`` metadata dict for orchestrator UIs."""
        return {
            "freshdata/trust_score": self.trust_score,
            "freshdata/grade": self.grade,
            "freshdata/threshold": self.threshold,
            "freshdata/passed": self.passed,
            "freshdata/high_risk_count": self.high_risk_count,
            "freshdata/row_count_in": self.row_count_in,
            "freshdata/row_count_out": self.row_count_out,
            "freshdata/generated_utc": self.generated_utc,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full result (scalar fields plus any ``report_dict``) as a dict."""
        data = {
            "trust_score": self.trust_score,
            "grade": self.grade,
            "threshold": self.threshold,
            "passed": self.passed,
            "high_risk_count": self.high_risk_count,
            "row_count_in": self.row_count_in,
            "row_count_out": self.row_count_out,
            "generated_utc": self.generated_utc,
            "on_low_score": self.on_low_score,
        }
        if self.report_dict is not None:
            data["report"] = self.report_dict
        return data


def _maybe_compliance_report(report: CleanReport, system_actor: str) -> dict[str, Any] | None:
    """Return a compliance bundle dict, or ``None`` when compliance is unavailable.

    Optional metadata must never break a pipeline gate, so both the absence of
    :mod:`freshdata.compliance` and any generation error degrade to ``None``.
    """
    try:
        from freshdata.compliance import ComplianceConfig, generate_compliance_report
    except ImportError:
        return None
    try:
        bundle = generate_compliance_report(
            report,
            frameworks=["sox_404"],
            config=ComplianceConfig(system_actor=system_actor),
        )
        return bundle.to_dict()
    except Exception:  # noqa: BLE001 — never let optional metadata fail the gate
        logger.debug("freshdata: compliance report generation skipped", exc_info=True)
        return None


def evaluate_trust_gate(
    df: pd.DataFrame,
    *,
    clean_config: CleanConfig | None = None,
    trust_score_threshold: float = 80.0,
    on_low_score: OnLowScore = "warn",
    publish_full_report: bool = False,
    system_actor: str = "freshdata",
) -> tuple[pd.DataFrame, TrustGateResult]:
    """Clean ``df``, score the result, and evaluate it against ``trust_score_threshold``.

    This is the shared engine behind every orchestrator adapter. It cleans the frame
    via :func:`freshdata.clean`, derives the 0-100 Data Trust Score of the *cleaned*
    output via :func:`~freshdata.enterprise.compute_trust_score`, and reports whether
    the score clears the threshold.

    The function itself never raises on a low score — it logs a warning and returns a
    :class:`TrustGateResult` whose :attr:`~TrustGateResult.should_fail` /
    :attr:`~TrustGateResult.should_skip` flags let each adapter react in its own terms
    (raise, skip, or warn).

    Parameters
    ----------
    df:
        The input DataFrame to clean and gate.
    clean_config:
        Optional :class:`freshdata.CleanConfig` forwarded to :func:`freshdata.clean`.
    trust_score_threshold:
        Minimum acceptable 0-100 trust score; the gate passes at or above it.
    on_low_score:
        Desired reaction when the gate fails — ``"warn"``, ``"fail"``, or ``"skip"``.
    publish_full_report:
        When ``True``, attach the clean report (and, if available, a compliance
        bundle) to :attr:`TrustGateResult.report_dict`.
    system_actor:
        Actor name recorded on the optional compliance artifact.

    Returns
    -------
    tuple[pandas.DataFrame, TrustGateResult]
        The cleaned DataFrame and the gate result.
    """
    row_count_in = len(df)
    cleaned, report = fd.clean(df, config=clean_config, return_report=True)
    trust = compute_trust_score(cleaned)
    score = float(trust.overall)
    high_risk = sum(1 for action in report.actions if getattr(action, "risk", None) == "high")

    report_dict: dict[str, Any] | None = None
    if publish_full_report:
        report_dict = {"clean_report": report.to_dict()}
        compliance = _maybe_compliance_report(report, system_actor)
        if compliance is not None:
            report_dict["compliance"] = compliance

    result = TrustGateResult(
        trust_score=score,
        grade=str(getattr(trust, "grade", "")),
        threshold=float(trust_score_threshold),
        passed=score >= trust_score_threshold,
        high_risk_count=high_risk,
        row_count_in=row_count_in,
        row_count_out=len(cleaned),
        generated_utc=_utc_now(),
        on_low_score=on_low_score,
        report_dict=report_dict,
    )

    if not result.passed:
        logger.warning(result.message)

    return cleaned, result
