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

_KNOWN_BACKENDS = ("deterministic", "memory", "embedding")


def _high_confidence_columns(
    proposals: list[SemanticProposal], ctx: SemanticContext
) -> frozenset[str]:
    """Columns already holding an auto-eligible proposal (embedding skips them)."""
    return frozenset(p.column for p in proposals if p.confidence >= ctx.auto_threshold)


def gather_proposals(
    df: pd.DataFrame,
    ctx: SemanticContext,
    config: CleanConfig,
    *,
    memory: object | None = None,
    report: CleanReport | None = None,
) -> list[SemanticProposal]:
    """Run the configured backends in trust order and combine their proposals.

    Backends only generate candidates — the policy gate in
    :func:`freshdata.semantic.policy.decide` still judges every proposal, and
    the executor's byte-identity guard still protects hard-protected columns
    no matter what any backend emits.
    """
    from ..apply import _merge_proposals  # noqa: PLC0415 - avoid import cycle

    requested = tuple(config.semantic_backends)
    for name in requested:
        if name not in _KNOWN_BACKENDS:
            message = (
                f"unknown semantic backend {name!r}; known backends: "
                f"{', '.join(_KNOWN_BACKENDS)}"
            )
            if config.strict:
                raise PolicyError(message)
            record_backend_skip(report, name, message)

    proposals: list[SemanticProposal] = []
    if "deterministic" in requested:
        proposals = DeterministicBackend().propose(df, ctx, Budget())

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

    return proposals


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
        settled_columns=_high_confidence_columns(prior, ctx),
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
