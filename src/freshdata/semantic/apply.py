"""Orchestrate the semantic stage: profile -> decide -> apply -> audit.

This is the only module the pipeline imports. It builds the context, profiles
proposals, runs the policy gate, applies the accepted repairs with vectorized
pandas operations (mapping distinct values, never row-by-row), and records every
decision — applied, suggested, or skipped — in the :class:`CleanReport`.
"""

from __future__ import annotations

import pandas as pd

from ..config import CleanConfig
from ..report import CleanReport
from .context import build_semantic_context
from .policy import decide
from .profiler import profile_proposals
from .types import SemanticPolicyDecision


def _maybe_downcast(series: pd.Series) -> pd.Series:
    """Tighten dtype after replacement (object -> numeric/bool) when unambiguous."""
    non_null = series.dropna()
    if non_null.empty:
        return series
    if all(isinstance(v, bool) for v in non_null):
        return series.astype("bool") if series.notna().all() else series
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


def _apply_column(series: pd.Series, mapping: dict) -> pd.Series:
    """Replace mapped distinct values, leaving everything else (and NaN) intact."""
    replaced = series.map(lambda v: mapping.get(v, v))
    return _maybe_downcast(replaced)


def _describe(decision: SemanticPolicyDecision) -> str:
    p = decision.proposal
    if p.issue_type == "identifier_like":
        return f"Protected identifier column {p.column!r} from semantic repair"
    pair = f"{p.raw_value!r} -> {p.proposed_value!r}"
    if decision.action == "apply":
        return f"Normalized {p.count} value(s) {pair}"
    if decision.action == "suggest":
        return f"Suggested semantic repair {pair}"
    return f"Skipped semantic repair {pair}"


def _record(report: CleanReport, decision: SemanticPolicyDecision) -> None:
    p = decision.proposal
    report.add(
        step="semantic",
        description=_describe(decision),
        column=p.column,
        count=p.count if decision.action != "skip" or p.issue_type == "identifier_like" else 0,
        rationale=p.rationale,
        risk=decision.risk,
        confidence=p.confidence,
        model_id=f"semantic:{p.issue_type}:v1",
        status=decision.status,
        reversible=True,
        memory_influenced=False,
        human_review=decision.human_review,
    )


def run_semantic(df: pd.DataFrame, config: CleanConfig, report: CleanReport) -> pd.DataFrame:
    """Run the semantic cleaning stage; return the (possibly) updated frame.

    A no-op when ``config.semantic_enabled`` is False, so the caller can invoke it
    unconditionally. Applied changes rebind whole columns, honoring
    ``preserve_original`` exactly like the rest of the pipeline.
    """
    if not config.semantic_enabled:
        return df

    ctx = build_semantic_context(df, config)
    proposals = profile_proposals(df, ctx)
    if not proposals:
        return df

    decisions = [decide(p, config, ctx) for p in proposals]

    # Collect accepted replacements per column, then apply vectorized.
    replacements: dict[str, dict] = {}
    for d in decisions:
        if d.action == "apply":
            replacements.setdefault(d.proposal.column, {})[d.proposal.raw_value] = (
                d.proposal.proposed_value
            )

    out = df
    if replacements:
        # Shallow copy so we never mutate the caller's frame when applying.
        out = df.copy(deep=False)
        for col, mapping in replacements.items():
            out[col] = _apply_column(out[col], mapping)

    for d in decisions:
        _record(report, d)
    return out
