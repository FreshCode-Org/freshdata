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
from .memory import build_semantic_metadata, is_memory_replay, semantic_memory_proposals
from .policy import decide
from .profiler import profile_proposals
from .scoring import make_proposal
from .types import SemanticContext, SemanticEvidence, SemanticPolicyDecision, SemanticProposal


def _maybe_downcast(series: pd.Series) -> pd.Series:
    """Tighten dtype after replacement (object -> numeric/bool/datetime) when
    unambiguous. A column only becomes datetime64 when *every* non-null value
    is already a ``pd.Timestamp`` — any leftover unconverted string leaves the
    column as object, so the conversion never silently drops information.
    """
    non_null = series.dropna()
    if non_null.empty:
        return series
    if all(isinstance(v, bool) for v in non_null):
        return series.astype("bool") if series.notna().all() else series
    if all(isinstance(v, pd.Timestamp) for v in non_null):
        try:
            return pd.to_datetime(series)
        except (ValueError, TypeError):
            return series
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


def _conflict_proposal(
    det: SemanticProposal, mem: SemanticProposal, memory: object
) -> SemanticProposal:
    """A high-risk, never-auto-applied record when memory and a deterministic
    expert disagree on how to repair the same raw value."""
    dataset_id = getattr(memory, "dataset_id", "?")
    evidence = (
        SemanticEvidence("conflict", f"deterministic expert proposed {det.proposed_value!r}", 0.0),
        SemanticEvidence(
            "conflict", f"cleaning memory {dataset_id!r} proposed {mem.proposed_value!r}", 0.0
        ),
    )
    return make_proposal(
        column=det.column,
        raw_value=det.raw_value,
        proposed_value=None,
        issue_type="unsafe_ambiguous",
        expert=f"{det.expert}+memory",
        base_confidence=0.75,
        evidence=evidence,
        count=det.count,
        rationale=(
            f"deterministic expert and cleaning memory {dataset_id!r} disagree on "
            f"{det.raw_value!r} ({det.proposed_value!r} vs {mem.proposed_value!r}); "
            "held for review"
        ),
        info=None,
    )


def _merge_proposals(
    deterministic: list[SemanticProposal],
    memory_proposals: list[SemanticProposal],
    memory: object,
) -> list[SemanticProposal]:
    """Merge deterministic + memory-retrieved proposals, deduping same-key repairs.

    Non-colliding proposals from either source pass through unchanged. When both
    a deterministic expert and memory propose the same ``(column, raw_value)``:
    if they agree on the proposed value, keep whichever has the higher
    confidence (memory wins ties); if they disagree, replace both with one
    high-risk, human-review-required ``unsafe_ambiguous`` record instead of
    auto-applying either.
    """
    if not memory_proposals:
        return list(deterministic)

    det_by_key: dict[tuple, list[SemanticProposal]] = {}
    for p in deterministic:
        det_by_key.setdefault((p.column, p.raw_value), []).append(p)

    merged: list[SemanticProposal] = []
    touched: set[tuple] = set()
    for mem_p in memory_proposals:
        key = (mem_p.column, mem_p.raw_value)
        det_list = det_by_key.get(key)
        if not det_list:
            merged.append(mem_p)
            continue
        touched.add(key)
        for det_p in det_list:
            if det_p.proposed_value == mem_p.proposed_value:
                merged.append(mem_p if mem_p.confidence >= det_p.confidence else det_p)
            else:
                merged.append(_conflict_proposal(det_p, mem_p, memory))

    for key, det_list in det_by_key.items():
        if key not in touched:
            merged.extend(det_list)
    return merged


def _record(report: CleanReport, decision: SemanticPolicyDecision, ctx: SemanticContext) -> None:
    p = decision.proposal
    from_memory = is_memory_replay(p)
    model_id_suffix = "memory" if from_memory else "v1"
    report.add(
        step="semantic",
        description=_describe(decision),
        column=p.column,
        count=p.count if decision.action != "skip" or p.issue_type == "identifier_like" else 0,
        rationale=p.rationale,
        risk=decision.risk,
        confidence=p.confidence,
        model_id=f"semantic:{p.issue_type}:{model_id_suffix}",
        status=decision.status,
        reversible=True,
        memory_influenced=from_memory,
        human_review=decision.human_review,
        metadata=build_semantic_metadata(p, ctx.info(p.column)),
    )


def run_semantic(
    df: pd.DataFrame, config: CleanConfig, report: CleanReport, memory: object | None = None
) -> pd.DataFrame:
    """Run the semantic cleaning stage; return the (possibly) updated frame.

    A no-op when ``config.semantic_enabled`` is False, so the caller can invoke it
    unconditionally. Applied changes rebind whole columns, honoring
    ``preserve_original`` exactly like the rest of the pipeline.

    With a ``memory`` (a :class:`~freshdata.CleaningMemory`), compatible learned
    semantic repairs are retrieved and merged in as additional candidate
    proposals before the policy gate runs — see :mod:`freshdata.semantic.memory`.
    Memory is evidence, not authority: every retrieved proposal still passes
    through :func:`~freshdata.semantic.policy.decide` exactly like a
    deterministic one.
    """
    if not config.semantic_enabled:
        return df

    ctx = build_semantic_context(df, config)
    proposals = list(profile_proposals(df, ctx))

    if memory is not None:
        memory_proposals = list(semantic_memory_proposals(df, ctx, memory))
        proposals = _merge_proposals(proposals, memory_proposals, memory)

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
        _record(report, d, ctx)
    return out
