"""Deterministic, explainable scoring for semantic proposals.

Confidence is derived from evidence weights so every score can be traced back
to concrete reasons; risk is a function of issue type and confidence. There is
no randomness here, so a given (value, column) always scores identically.
"""

from __future__ import annotations

from .types import SemanticColumnInfo, SemanticEvidence, SemanticProposal

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
) -> SemanticProposal:
    """Build a fully scored :class:`SemanticProposal`.

    Experts describe *what* they found (raw/proposed/evidence); scoring decides
    *how confident* and *how risky* it is, so the policy gate sees a uniform,
    comparable score regardless of which expert produced the proposal.
    """
    confidence = confidence_from_evidence(base_confidence, evidence)
    risk = risk_for(issue_type, confidence)
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
    )
