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
from .memory import build_semantic_metadata, is_memory_replay
from .policy import decide
from .scoring import calibrate_proposals, make_proposal
from .types import SemanticContext, SemanticEvidence, SemanticPolicyDecision, SemanticProposal


def _maybe_downcast(series: pd.Series, *, allow_numeric: bool = True) -> pd.Series:
    """Tighten dtype after replacement (object -> numeric/bool/datetime) when
    unambiguous. A column only becomes datetime64 when *every* non-null value
    is already a ``pd.Timestamp`` — any leftover unconverted string leaves the
    column as object, so the conversion never silently drops information.

    ``allow_numeric=False`` skips the numeric attempt entirely: repairs whose
    *targets* are strings (canonical phones like ``"+9198..."``, normalized
    emails) must stay text, since ``to_numeric`` would happily parse
    ``"+919876543210"`` into an integer and destroy the canonical form.
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
    if not allow_numeric:
        return series
    try:
        return pd.to_numeric(series)
    except (ValueError, TypeError):
        return series


def _apply_column(series: pd.Series, mapping: dict) -> pd.Series:
    """Replace mapped distinct values, leaving everything else (and NaN) intact."""
    replaced = series.map(lambda v: mapping.get(v, v))
    string_targets = any(isinstance(v, str) for v in mapping.values())
    return _maybe_downcast(replaced, allow_numeric=not string_targets)


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
    auto-applying either. A flag-only proposal (``proposed_value=None``) is
    an abstention, not a disagreement: a concrete repair from the other
    source supersedes it.
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
        # A flag (proposed_value=None) is an abstention — "this value looks
        # wrong but no repair is within reach" — not a competing repair, so
        # only concrete deterministic repairs can agree or conflict with a
        # concrete replayed one (mirrors the embedding backend's treatment
        # of deterministic flags).
        concrete_det = [d for d in det_list if d.proposed_value is not None]
        if mem_p.proposed_value is not None and not concrete_det:
            merged.append(mem_p)
            continue
        if mem_p.proposed_value is None and concrete_det:
            merged.extend(concrete_det)
            continue
        for det_p in concrete_det or det_list:
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
    if from_memory:
        model_id_suffix = "memory"
    elif p.backend in ("embedding", "profile"):
        model_id_suffix = p.backend
    else:
        model_id_suffix = "v1"
    metadata = build_semantic_metadata(p, ctx.info(p.column))
    if p.provenance is not None:
        # Learned-profile provenance (profile_influenced, profile_id, support,
        # learned_precision, transform_family) rides into the action metadata.
        metadata = {**metadata, **dict(p.provenance)}
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
        metadata=metadata,
    )


def run_semantic(
    df: pd.DataFrame,
    config: CleanConfig,
    report: CleanReport,
    memory: object | None = None,
    profile: object | None = None,
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

    With a ``profile`` (a :class:`~freshdata.learning.LearningProfile`), the
    profile's learned value maps and (optional) example retrieval join the
    candidate pool through the same gate — learned evidence is never
    authority either.
    """
    if not config.semantic_enabled:
        return df

    from .backends import gather_proposals  # noqa: PLC0415 - avoid import cycle

    ctx = build_semantic_context(df, config)
    proposals = gather_proposals(df, ctx, config, memory=memory, profile=profile, report=report)

    if not proposals:
        return df

    replacements = resolve_replacements(proposals, config, ctx, report)

    out = df
    if replacements:
        # Shallow copy so we never mutate the caller's frame when applying.
        out = df.copy(deep=False)
        for col, mapping in replacements.items():
            out[col] = _apply_column(out[col], mapping)
    return out


def resolve_replacements(
    proposals: list[SemanticProposal],
    config: CleanConfig,
    ctx: SemanticContext,
    report: CleanReport,
) -> dict[str, dict]:
    """Calibrate + gate proposals, record every decision, and return the accepted
    per-column ``{raw_value: proposed_value}`` maps.

    Shared by the pandas reference path (:func:`run_semantic`) and the native
    distinct path (:mod:`freshdata.semantic.native`) so both surface identical
    actions and apply identical repairs — only the *application* differs
    (vectorized pandas vs. native ``replace``/SQL).
    """
    proposals = calibrate_proposals(proposals, config, ctx, report=report)
    decisions = [decide(p, config, ctx) for p in proposals]

    replacements: dict[str, dict] = {}
    for d in decisions:
        if d.action == "apply":
            replacements.setdefault(d.proposal.column, {})[d.proposal.raw_value] = (
                d.proposal.proposed_value
            )
    for d in decisions:
        _record(report, d, ctx)
    return replacements
