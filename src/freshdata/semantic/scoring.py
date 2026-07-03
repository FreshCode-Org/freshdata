"""Deterministic, explainable scoring and calibration for semantic proposals.

Confidence is derived from evidence weights so every score can be traced back
to concrete reasons; risk is a function of issue type and confidence. There is
no randomness here, so a given (value, column) always scores identically.

Phase 3 adds post-hoc calibration: expert scores are treated as *raw* scores
and mapped through per-(backend, issue-family) isotonic lookup tables to
calibrated probabilities-of-correct-repair. Tables load from the model
registry (``calib-v1``) when installed, else from the packaged default; when
neither exists the raw score passes through with
``calibration_version="uncalibrated"``. The packaged default is the identity
for deterministic and memory proposals — calibration never changes a default
install's decisions — and embedding proposals are always capped below 1.0.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .types import SemanticColumnInfo, SemanticEvidence, SemanticProposal

if TYPE_CHECKING:
    from ..config import CleanConfig
    from ..report import CleanReport
    from .types import SemanticContext

#: Issue types that are inherently riskier and should rarely auto-apply.
_HIGHER_RISK_ISSUES = frozenset({"category_synonym", "date_phrase", "unsafe_ambiguous"})


def confidence_from_evidence(base: float, evidence: tuple[SemanticEvidence, ...]) -> float:
    """Combine a base confidence with evidence weights, clamped to [0, 0.999].

    The 0.999 ceiling keeps semantic repairs honestly below the certainty of
    deterministic representation repairs (which report ``confidence=1.0``).
    """
    score = base + sum(e.weight for e in evidence)
    return max(0.0, min(0.999, score))


def risk_for(issue_type: str, confidence: float) -> str:
    """Map an issue type and confidence to ``"low" | "medium" | "high"``."""
    if issue_type == "unsafe_ambiguous":
        return "high"
    if confidence < 0.70:
        return "high"
    if issue_type in _HIGHER_RISK_ISSUES:
        return "medium" if confidence >= 0.85 else "high"
    return "low" if confidence >= 0.90 else "medium"


def make_proposal(
    *,
    column: str,
    raw_value: object,
    proposed_value: object,
    issue_type: str,
    expert: str,
    base_confidence: float,
    evidence: tuple[SemanticEvidence, ...],
    count: int,
    rationale: str,
    info: SemanticColumnInfo | None = None,
    risk_override: str | None = None,
    backend: str = "deterministic",
) -> SemanticProposal:
    """Build a fully scored :class:`SemanticProposal`.

    Experts describe *what* they found (raw/proposed/evidence); scoring decides
    *how confident* and *how risky* it is, so the policy gate sees a uniform,
    comparable score regardless of which expert produced the proposal.

    ``risk_override`` lets an expert assert a risk level ``risk_for`` cannot
    derive from confidence alone (e.g. an ambiguous date phrase is high-risk
    regardless of how confident the *fallback* interpretation is). Only
    :class:`~freshdata.semantic.experts.DatePhraseExpert` uses it; every other
    expert omits it and gets the same automatic scoring as before.
    """
    confidence = confidence_from_evidence(base_confidence, evidence)
    risk = risk_override if risk_override is not None else risk_for(issue_type, confidence)
    return SemanticProposal(
        column=column,
        raw_value=raw_value,
        proposed_value=proposed_value,
        issue_type=issue_type,
        expert=expert,
        confidence=round(confidence, 4),
        risk=risk,
        rationale=rationale,
        evidence=evidence,
        count=count,
        backend=backend,
    )


# ---------------------------------------------------------------------------
# Calibration (Phase 3)
# ---------------------------------------------------------------------------

_UNCALIBRATED = "uncalibrated"
_EMBEDDING_CEILING = 0.999  # embedding proposals are never pinned at 1.0
_PACKAGED_TABLE = Path(__file__).resolve().parent / "data" / "calib_default.json"


@dataclass(frozen=True)
class ActionConfidence:
    """A calibrated confidence with full provenance for the audit trail."""

    point: float
    raw: float
    calibration_version: str
    features_hash: str


def calibration_features(
    proposal: SemanticProposal,
    ctx: SemanticContext,
    info: SemanticColumnInfo | None,
) -> dict[str, Any]:
    """The deterministic feature record a calibrator may condition on.

    Hashed (not stored raw) into ``ActionConfidence.features_hash`` so a report
    consumer can verify two actions were scored from identical evidence.
    """
    evidence_kinds = sorted({e.kind for e in proposal.evidence})
    margin = None
    semantic_type_confidence = None
    for e in proposal.evidence:
        if e.kind == "embedding" and "margin=" in e.detail:
            try:
                margin = float(e.detail.split("margin=")[1].split()[0])
            except (IndexError, ValueError):  # pragma: no cover - defensive
                margin = None
        if e.kind == "semantic_type" and "confidence=" in e.detail:
            try:
                semantic_type_confidence = float(e.detail.split("confidence=")[1].split()[0])
            except (IndexError, ValueError):  # pragma: no cover - defensive
                semantic_type_confidence = None
    n_nonnull = info.n_nonnull if info is not None else None
    nunique = info.nunique if info is not None else None
    return {
        "raw_score": proposal.confidence,
        "backend": proposal.backend,
        "issue_type": proposal.issue_type,
        "risk": proposal.risk,
        "role_confidence": 1.0 if info is not None and info.role else 0.0,
        "semantic_type_confidence": semantic_type_confidence,
        "distinct_support": nunique,
        "coverage": (
            round(proposal.count / n_nonnull, 6) if n_nonnull else None
        ),
        "memory_support_count": sum(1 for e in proposal.evidence if e.kind == "memory_replay"),
        "policy_rule_present": bool(info is not None and info.mutable is not None),
        "allowed_values_present": bool(info is not None and info.allowed_values),
        "margin_to_second_candidate": margin,
        "evidence_kinds": evidence_kinds,
    }


def features_hash(features: dict[str, Any]) -> str:
    """Stable short hash of the canonical feature JSON."""
    canonical = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class _IsotonicTable:
    """Piecewise-linear monotone lookup loaded from a JSON calibration file.

    JSON shape::

        {"version": "calib-default-1",
         "tables": {"embedding": {"category_synonym": {"x": [...], "y": [...]}}}}

    Lookup is by ``(backend, issue_type)`` with a per-backend ``"*"`` fallback;
    a missing entry means identity (raw score passes through).
    """

    def __init__(self, version: str, tables: dict[str, dict[str, dict[str, list[float]]]]):
        self.version = version
        self._tables = tables

    @classmethod
    def from_json(cls, text: str) -> _IsotonicTable:
        data = json.loads(text)
        version = str(data.get("version", "unknown"))
        tables = data.get("tables", {})
        if not isinstance(tables, dict):
            raise ValueError("calibration file: 'tables' must be an object")
        for backend, families in tables.items():
            for family, curve in families.items():
                xs, ys = curve.get("x", []), curve.get("y", [])
                if len(xs) != len(ys) or len(xs) < 2:
                    raise ValueError(
                        f"calibration curve {backend}/{family}: x and y must be "
                        "equal-length with at least two points"
                    )
                if sorted(xs) != list(xs) or sorted(ys) != list(ys):
                    raise ValueError(
                        f"calibration curve {backend}/{family}: x and y must be non-decreasing"
                    )
        return cls(version, tables)

    def curve(self, backend: str, issue_type: str) -> dict[str, list[float]] | None:
        families = self._tables.get(backend)
        if not families:
            return None
        return families.get(issue_type) or families.get("*")

    def apply(self, backend: str, issue_type: str, raw: float) -> float:
        curve = self.curve(backend, issue_type)
        if curve is None:
            return raw
        xs, ys = curve["x"], curve["y"]
        if raw <= xs[0]:
            return ys[0]
        if raw >= xs[-1]:
            return ys[-1]
        hi = bisect.bisect_right(xs, raw)
        lo = hi - 1
        span = xs[hi] - xs[lo]
        frac = (raw - xs[lo]) / span if span else 0.0
        return ys[lo] + frac * (ys[hi] - ys[lo])


_table_cache: dict[str, _IsotonicTable | None] = {}


def _load_calibration_table() -> _IsotonicTable | None:
    """Registry ``calib-v1`` if installed, else the packaged default, else None."""
    if "table" in _table_cache:
        return _table_cache["table"]
    table: _IsotonicTable | None = None
    try:
        from ..models import registry as _registry  # noqa: PLC0415 - lazy, no cycle at import

        if _registry.is_installed(_registry.CALIBRATION_ID):
            table = _IsotonicTable.from_json(
                _registry.path(_registry.CALIBRATION_ID).read_text(encoding="utf-8")
            )
    except (OSError, ValueError):  # pragma: no cover - corrupt user table
        table = None
    if table is None and _PACKAGED_TABLE.is_file():
        try:
            table = _IsotonicTable.from_json(_PACKAGED_TABLE.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # pragma: no cover - corrupt packaged table
            table = None
    _table_cache["table"] = table
    return table


def reset_calibration_cache() -> None:
    """Forget the cached table (tests / after pulling a new calib model)."""
    _table_cache.clear()


def calibrate_proposals(
    proposals: list[SemanticProposal],
    config: CleanConfig,
    ctx: SemanticContext,
    report: CleanReport | None = None,
) -> list[SemanticProposal]:
    """Attach calibrated confidences; the gate then reads them transparently.

    Each proposal's ``confidence`` is replaced by the calibrated point (so
    :func:`~freshdata.semantic.policy.decide` needs no change) and the full
    :class:`ActionConfidence` provenance rides along in ``calibration``.
    Without any table the raw score passes through as ``"uncalibrated"``.
    """
    del config
    table = _load_calibration_table()
    missing_noted = False
    out: list[SemanticProposal] = []
    for p in proposals:
        raw = p.confidence
        curve = table.curve(p.backend, p.issue_type) if table is not None else None
        if curve is None and p.backend == "deterministic":
            # No curve for the deterministic experts (the packaged default is
            # deliberately identity there): leave the proposal byte-identical
            # so calibration can never change a default install's decisions.
            out.append(p)
            continue
        if table is None or curve is None:
            point, version = raw, _UNCALIBRATED
            if p.backend == "embedding" and report is not None and not missing_noted:
                report.add_warning(
                    "Semantic confidence calibration table missing; raw model scores "
                    "used (calibration_version=uncalibrated)"
                )
                missing_noted = True
        else:
            point, version = table.apply(p.backend, p.issue_type, raw), table.version
        if p.backend == "embedding":
            point = min(point, _EMBEDDING_CEILING)
        point = round(max(0.0, min(1.0, point)), 4)
        features = calibration_features(p, ctx, ctx.info(p.column))
        confidence = ActionConfidence(
            point=point,
            raw=raw,
            calibration_version=version,
            features_hash=features_hash(features),
        )
        out.append(dataclasses.replace(p, confidence=point, calibration=confidence))
    return out
