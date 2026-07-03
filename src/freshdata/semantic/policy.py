"""The semantic policy gate: decide apply / suggest / skip for each proposal.

This is the single place where mode, thresholds, and column protection combine.
It never touches data; it only returns a :class:`SemanticPolicyDecision`. The
hard safety rules (never mutate target/id/preserve columns, never apply below the
review threshold, never auto-apply high risk) live here so they are easy to audit.
"""

from __future__ import annotations

from ..config import CleanConfig
from .types import SemanticContext, SemanticPolicyDecision, SemanticProposal


def _protection(proposal: SemanticProposal, ctx: SemanticContext) -> str | None:
    """Return a reason string if the column must never be mutated, else ``None``."""
    column = proposal.column
    info = ctx.info(column)
    if column == ctx.target_column or (info is not None and info.role == "target"):
        return "target column is never modified"
    if info is not None and info.mutable is False:
        return "column is context-protected (mutable=False)"
    if column in ctx.preserve_columns or (info is not None and info.preserve):
        return "column is in preserve_columns"
    is_id = column in ctx.id_columns or (info is not None and info.identifier_like)
    if is_id:
        if info is not None and info.mutable is True:
            return None  # context explicitly opted this identifier in
        return "identifier column is protected (set mutable=True in semantic_context to allow)"
    return None


def _decision(
    proposal: SemanticProposal, action: str, status: str, reason: str, *, human_review: bool
) -> SemanticPolicyDecision:
    risk = "high" if proposal.issue_type == "unsafe_ambiguous" else proposal.risk
    return SemanticPolicyDecision(
        proposal=proposal,
        action=action,
        status=status,
        risk=risk,
        reason=reason,
        human_review=human_review,
    )


def decide(
    proposal: SemanticProposal, config: CleanConfig, ctx: SemanticContext
) -> SemanticPolicyDecision:
    """Gate one proposal into ``apply`` / ``suggest`` / ``skip``."""
    # 1. Protective veto expert: never mutate, always auditable.
    if proposal.issue_type == "identifier_like":
        return _decision(
            proposal, "skip", "skipped", proposal.rationale, human_review=True
        )

    # 2. Column-level protection (target / id / preserve).
    protected = _protection(proposal, ctx)
    if protected is not None:
        return _decision(proposal, "skip", "skipped", protected, human_review=True)

    # 3. Confidence floor: below the review threshold we never even suggest a
    #    mutation — record a skip so the (low-confidence) signal is still audited.
    if proposal.confidence < ctx.review_threshold:
        return _decision(
            proposal,
            "skip",
            "skipped",
            f"confidence {proposal.confidence:.2f} below review threshold "
            f"{ctx.review_threshold:.2f}",
            human_review=False,
        )

    mode = ctx.mode
    high_conf = proposal.confidence >= ctx.auto_threshold

    # 4. assist: detect and report only, never mutate.
    if mode == "assist":
        return _decision(
            proposal, "suggest", "suggested", "assist mode records suggestions only",
            human_review=True,
        )

    # 5. review: apply only deterministic, low-risk, high-confidence repairs.
    if mode == "review":
        if high_conf and proposal.risk == "low":
            return _decision(
                proposal, "apply", "automatic", "deterministic low-risk repair",
                human_review=False,
            )
        return _decision(
            proposal, "suggest", "suggested", "held for review (mode=review)",
            human_review=True,
        )

    # 6. auto: apply high-confidence, non-high-risk, policy-clean repairs.
    if mode == "auto":
        if high_conf and proposal.risk != "high":
            return _decision(
                proposal, "apply", "automatic", "high-confidence low-risk repair",
                human_review=False,
            )
        return _decision(
            proposal, "suggest", "suggested", "held for review (mode=auto)",
            human_review=True,
        )

    # 7. Disabled / unknown mode: do nothing.
    return _decision(proposal, "skip", "skipped", "semantic layer disabled", human_review=False)
