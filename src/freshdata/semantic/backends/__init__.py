"""Named semantic backends behind the ``semantic_backends`` config tuple.

Order is trust order: ``deterministic`` (Phase-2 experts) always first, then
``memory`` replay merged in with the existing conflict semantics, then the
optional ``embedding`` backend appended for the residue the earlier backends
left unhandled. The default tuple ``("deterministic",)`` reproduces the
pre-backend behavior byte-for-byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...context.types import PolicyError
from .base import BackendUnavailable, Budget, SemanticBackend, record_backend_skip
from .deterministic import DeterministicBackend
from .memory import MemoryBackend

if TYPE_CHECKING:
    import pandas as pd

    from ...config import CleanConfig
    from ...report import CleanReport
    from ..types import SemanticContext, SemanticProposal

__all__ = [
    "BackendUnavailable",
    "Budget",
    "DeterministicBackend",
    "MemoryBackend",
    "SemanticBackend",
    "gather_proposals",
    "record_backend_skip",
]

_KNOWN_BACKENDS = ("deterministic", "profile", "memory", "embedding")


def gather_proposals(
    df: pd.DataFrame,
    ctx: SemanticContext,
    config: CleanConfig,
    *,
    memory: object | None = None,
    profile: object | None = None,
    report: CleanReport | None = None,
) -> list[SemanticProposal]:
    """Run the configured backends in trust order and combine their proposals.

    Backends only generate candidates — the policy gate in
    :func:`freshdata.semantic.policy.decide` still judges every proposal, and
    the executor's byte-identity guard still protects hard-protected columns
    no matter what any backend emits.
    """
    from ...plugins import known_backend_names  # noqa: PLC0415 - avoid import cycle
    from ..apply import _merge_proposals  # noqa: PLC0415 - avoid import cycle

    requested = tuple(config.semantic_backends)
    plugin_names = known_backend_names()
    for name in requested:
        if name not in _KNOWN_BACKENDS and name not in plugin_names:
            known = ", ".join((*_KNOWN_BACKENDS, *plugin_names))
            message = f"unknown semantic backend {name!r}; known backends: {known}"
            if config.strict:
                raise PolicyError(message)
            record_backend_skip(report, name, message)

    proposals: list[SemanticProposal] = []
    if "deterministic" in requested:
        proposals = DeterministicBackend().propose(df, ctx, Budget())

    # Learned-profile replay sits between the deterministic experts and
    # memory in the trust order: it runs whenever a profile is supplied and
    # merges through the same conflict-aware path as memory — agreement keeps
    # the higher-confidence proposal, disagreement becomes a held-for-review
    # ``unsafe_ambiguous`` record instead of auto-applying either side.
    if profile is not None:
        from .profile import ProfileBackend  # noqa: PLC0415 - lazy import

        profile_proposals = ProfileBackend(profile).propose(df, ctx, Budget())
        proposals = _merge_proposals(proposals, profile_proposals, profile)

    # Memory replay keeps its Phase-2 contract: it runs whenever a memory is
    # supplied (with or without "memory" in the tuple) and merges through the
    # same conflict-aware path as before.
    if memory is not None:
        memory_proposals = MemoryBackend(memory).propose(df, ctx, Budget())
        proposals = _merge_proposals(proposals, memory_proposals, memory)
    elif "memory" in requested:
        record_backend_skip(
            report,
            "memory",
            "no cleaning memory supplied; pass memory= to enable replay proposals",
        )

    if "embedding" in requested:
        proposals = proposals + _embedding_proposals(df, ctx, config, proposals, report)

    # Plugin backends run last (lowest trust), appended like embedding: their
    # proposals still pass the gate and the byte-identity guard unchanged.
    plugin_requested = [n for n in requested if n in plugin_names]
    if plugin_requested:
        proposals = proposals + _plugin_backend_proposals(
            df, ctx, config, plugin_requested, report)

    return proposals


def _plugin_backend_proposals(
    df: pd.DataFrame,
    ctx: SemanticContext,
    config: CleanConfig,
    names: list[str],
    report: CleanReport | None,
) -> list[SemanticProposal]:
    """Run each requested, active plugin backend under the model budget."""
    from ...plugins import get_active_backend  # noqa: PLC0415 - avoid import cycle

    out: list[SemanticProposal] = []
    for name in names:
        backend = get_active_backend(name)
        if backend is None:
            record_backend_skip(
                report, name,
                "plugin backend is registered but inactive (missing dependency, "
                "or network-using and not enabled)")
            continue
        try:
            backend.warm_up()
        except BackendUnavailable as exc:
            record_backend_skip(report, name, str(exc))
            continue
        budget = Budget.from_config(config.semantic_budget)
        out.extend(backend.propose(df, ctx, budget))
        if budget.exhausted and report is not None:
            reason = f"stopped early, budget exhausted ({budget.exhausted_reason})"
            report.record_fallback(backend=name, step="semantic", reason=reason)
            report.add_warning(f"Semantic backend {name!r} {reason}")
    return out


def _embedding_proposals(
    df: pd.DataFrame,
    ctx: SemanticContext,
    config: CleanConfig,
    prior: list[SemanticProposal],
    report: CleanReport | None,
) -> list[SemanticProposal]:
    """Run the embedding backend, degrading to a report note when unavailable."""
    try:
        from .embedding import EmbeddingBackend  # noqa: PLC0415 - lazy optional path
    except ImportError as exc:  # pragma: no cover - transitional safety net
        record_backend_skip(report, "embedding", f"backend import failed: {exc}")
        return []

    backend = EmbeddingBackend(
        config,
        # Only actual repairs suppress embedding candidates; a deterministic
        # *flag* (proposed_value=None) is an abstention the model may rescue.
        already_proposed=frozenset(
            (p.column, p.raw_value) for p in prior if p.proposed_value is not None
        ),
    )
    try:
        backend.warm_up()
    except BackendUnavailable as exc:
        record_backend_skip(report, "embedding", str(exc))
        return []

    budget = Budget.from_config(config.semantic_budget)
    result = backend.propose(df, ctx, budget)
    if budget.exhausted and report is not None:
        reason = f"stopped early, budget exhausted ({budget.exhausted_reason})"
        report.record_fallback(backend="embedding", step="semantic", reason=reason)
        report.add_warning(f"Semantic backend 'embedding' {reason}")
    return result
