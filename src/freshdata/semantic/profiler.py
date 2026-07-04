"""Profile semantic issues and generate scored proposals.

This is steps 1–3 of the semantic pipeline: profile issues, generate repair
proposals, score them. It never mutates data and never consults policy — it just
produces a :class:`SemanticProposalSet` for the policy gate and preview surfaces
to consume.
"""

from __future__ import annotations

import pandas as pd

from .context import build_semantic_context
from .router import route
from .types import (
    SemanticColumnInfo,
    SemanticContext,
    SemanticIssue,
    SemanticProposalSet,
)


def _column_eligible(info: SemanticColumnInfo, ctx: SemanticContext) -> bool:
    """Skip columns that are free text or too high-cardinality to reason about.

    Email/phone columns and columns with an explicit allowed-values list stay
    eligible even when the role inference read them as free text — the user's
    hint is authoritative about what the column contains.
    """
    if info.free_text and not (info.email_like or info.phone_like or info.allowed_values):
        return False
    if info.nunique is None:
        return False
    return info.nunique <= ctx.max_distinct_values


def profile_proposals(df: pd.DataFrame, ctx: SemanticContext) -> SemanticProposalSet:
    """Run eligible experts over every eligible column and collect proposals."""
    proposals = SemanticProposalSet()
    for col in df.columns:
        info = ctx.columns[str(col)]
        if not _column_eligible(info, ctx):
            continue
        for expert in route(info):
            proposals.extend(expert.propose(df[col], info))
    return proposals


def profile_proposals_native(
    series_by_col: dict[str, pd.Series], ctx: SemanticContext
) -> SemanticProposalSet:
    """Like :func:`profile_proposals`, but over pre-built *distinct* series.

    Each series in *series_by_col* holds a column's distinct values and carries
    its true ``value_counts`` in ``series.attrs['fd_value_counts']`` (attached by
    the native-engine extractor). This is the deterministic-backend seam for the
    native distinct path: routing, experts, and proposals are byte-for-byte what
    the reference path produces from the full column.
    """
    proposals = SemanticProposalSet()
    for col, series in series_by_col.items():
        info = ctx.columns[str(col)]
        if not _column_eligible(info, ctx):
            continue
        for expert in route(info):
            proposals.extend(expert.propose(series, info))
    return proposals


def profile_semantic_issues(df: pd.DataFrame, ctx: SemanticContext) -> list[SemanticIssue]:
    """Public, mutation-free view of the issues found (used by preview APIs)."""
    issues: list[SemanticIssue] = []
    for p in profile_proposals(df, ctx):
        issues.append(
            SemanticIssue(
                column=p.column,
                issue_type=p.issue_type,
                raw_value=p.raw_value,
                count=p.count,
                expert=p.expert,
            )
        )
    return issues


def plan_semantic(df: pd.DataFrame, config) -> SemanticProposalSet:
    """Convenience: build context and profile proposals in one call (no mutation)."""
    ctx = build_semantic_context(df, config)
    return profile_proposals(df, ctx)
