"""Shared scaffolding for :mod:`freshdata.compliance`.

This module holds the framework-agnostic pieces every generator depends on:
the :class:`ComplianceConfig` knobs, the :class:`FrameworkReport` /
:class:`ComplianceBundle` containers, the :class:`ComplianceGapError`, and a
handful of deterministic helpers (timestamps, identifiers, dataframe hashing).

Nothing here imports the rest of :mod:`freshdata` at module load, so importing
``freshdata.compliance`` stays cheap and free of import cycles.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

#: Advisory carried by every framework report that does not define its own
#: verbatim caveat. Non-negotiable: a compliance artifact supports — but never
#: constitutes — a legal determination of compliance.
GENERAL_CAVEAT = (
    "This artifact is generated automatically to support compliance workflows. "
    "It documents the data transformations freshdata applied and how they map to "
    "the named control framework, but it is not itself a certified compliance "
    "system and does not constitute a legal determination of compliance. Review "
    "by a qualified compliance professional is required."
)


# --------------------------------------------------------------------------- #
# Deterministic helpers                                                        #
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string ending in ``Z``.

    Uses a timezone-aware ``datetime`` (``datetime.utcnow`` is deprecated).
    """
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def new_session_id() -> str:
    """Return a fresh session identifier, unique per generation call."""
    return f"SESSION-{uuid.uuid4().hex[:12].upper()}"


def new_entry_id(prefix: str) -> str:
    """Return an ``f"{prefix}-{8 hex chars}"`` identifier (e.g. ``CFR11-1A2B3C4D``)."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def sha256_df(df: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 hex digest of a DataFrame's values."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class ComplianceConfig:
    """Caller-supplied context for compliance report generation.

    Every field has a sensible default, so ``ComplianceConfig()`` is valid.
    ``trust_score`` and ``masked_columns`` are freshdata-specific additions that
    let the report-only path (no ``EnterpriseResult``) still drive the trust
    gates and PII-coverage detection.
    """

    controller_name: str = ""
    controller_contact: str = ""
    processing_purpose: str = "Data quality improvement"
    legal_basis: str = "legitimate_interest"
    retention_days: int = 2555  # 7 years — a common regulatory floor
    data_subject_categories: list[str] = field(default_factory=list)
    system_actor: str = "freshdata"
    input_dataframe_hash: str | None = None
    operator_id: str | None = None
    fail_on_hipaa_gap: bool = False
    #: Trust score (0–100) used by the SOX / 21 CFR gates when no EnterpriseResult
    #: is supplied. Falls back to ``domain_trust_score * 100`` when available.
    trust_score: float | None = None
    #: Columns known to have been PII-masked, for the report-only path. Merged
    #: with any columns reported by an EnterpriseResult's mask report.
    masked_columns: list[str] = field(default_factory=list)
    #: When True, the 21 CFR audit treats normalizing rewrites (whitespace trim,
    #: sentinel canonicalisation) as *obscuring* — i.e. a value was overwritten
    #: without retaining its pre-image — so the gate fails. Default keeps these
    #: lossless normalisations non-obscuring (only genuine value rewrites such as
    #: outlier capping / fuzzy clustering fail the gate).
    strict_cfr_normalization: bool = False


class ComplianceGapError(Exception):
    """Raised when a framework gate fails and the caller opted into hard failure."""


# --------------------------------------------------------------------------- #
# Per-framework report                                                         #
# --------------------------------------------------------------------------- #
#: Keys whose value is a list (or dict) of homogeneous entries, used by
#: :meth:`FrameworkReport.to_frame` to pick the natural tabular view.
_FRAME_KEYS = (
    "audit_entries",
    "transformation_log",
    "action_evidence",
    "erasure_log",
    "principles",
    "identifier_coverage",
)


@dataclass
class FrameworkReport:
    """Result of one framework generator: status, messages, and payload."""

    framework_key: str
    framework_name: str
    passed: bool
    warnings: list[str]
    errors: list[str]
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework_key,
            "framework_name": self.framework_name,
            "passed": self.passed,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            **self.data,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_frame(self) -> pd.DataFrame:
        """Return the report's natural tabular view (one row per entry).

        Picks the first *non-empty* entry collection; if every candidate is
        empty (e.g. an erasure log with no erasures), falls back to a single
        summary row so the frame is never spuriously empty.
        """
        for key in _FRAME_KEYS:
            if key in self.data:
                value = self.data[key]
                rows = list(value.values()) if isinstance(value, dict) else value
                if len(rows):
                    return pd.DataFrame(rows)
        return pd.DataFrame([self.to_dict()])


# --------------------------------------------------------------------------- #
# Bundle of reports                                                            #
# --------------------------------------------------------------------------- #
class ComplianceBundle:
    """A keyed collection of :class:`FrameworkReport` objects."""

    def __init__(self, reports: dict[str, FrameworkReport]):
        self._reports = dict(reports)

    def __getitem__(self, key: str) -> FrameworkReport:
        return self._reports[key]

    def __contains__(self, key: object) -> bool:
        return key in self._reports

    def __iter__(self):
        return iter(self._reports)

    def __len__(self) -> int:
        return len(self._reports)

    @property
    def frameworks(self) -> list[str]:
        return list(self._reports)

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self._reports.items()}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_frame(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for key, report in self._reports.items():
            frame = report.to_frame()
            if "framework" in frame.columns:  # summary-row fallback already has it
                frame = frame.drop(columns=["framework"])
            frame.insert(0, "framework", key)
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            k: {"passed": v.passed, "warnings": list(v.warnings), "errors": list(v.errors)}
            for k, v in self._reports.items()
        }
