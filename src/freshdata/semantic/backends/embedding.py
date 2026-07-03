"""The optional embedding backend — local model evidence for the residue.

Runs only when explicitly listed in ``semantic_backends``, the ``[semantic]``
extra is installed, and the ``fd-col-encoder-v1`` model is present (or a test
encoder is injected); otherwise :meth:`EmbeddingBackend.warm_up` raises
:class:`BackendUnavailable` and cleaning continues without it. It never
downloads anything.

Scope is deliberately narrow: distinct values of low-cardinality category /
reference columns that the deterministic experts left unhandled. Protected,
id, target, free-text, and high-cardinality columns are excluded before any
value reaches the model. Proposals with allowed-values evidence use
``issue_type="reference_value"`` (auto-eligible through the normal gate at
very high calibrated confidence); pure similarity clustering uses
``issue_type="category_synonym"`` at ``risk="medium"`` minimum and a
calibration ceiling below the default auto threshold, so it is suggest-only
unless the user loosens their thresholds. Ambiguous matches (a close second
candidate) produce **no** proposal at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ...models import runtime as model_runtime
from ..cache import EmbeddingCache
from ..scoring import make_proposal
from ..types import SemanticEvidence
from .base import BackendUnavailable, Budget

if TYPE_CHECKING:
    from ...config import CleanConfig
    from ..types import SemanticColumnInfo, SemanticContext, SemanticProposal

#: Minimum cosine for a reference (allowed-values) match to become a proposal.
_REFERENCE_MIN_COSINE = 0.80
#: Minimum cosine for a clustering (no allowed values) suggestion.
_CLUSTER_MIN_COSINE = 0.85
#: Required cosine margin over the runner-up; closer than this is ambiguous.
_MARGIN = 0.05
#: Clustering only considers columns at or below this many distinct values.
_CLUSTER_MAX_DISTINCT = 30


class EmbeddingBackend:
    name = "embedding"

    def __init__(
        self,
        config: CleanConfig,
        *,
        already_proposed: frozenset[tuple[str, object]] = frozenset(),
        settled_columns: frozenset[str] = frozenset(),
    ) -> None:
        self._config = config
        self._already_proposed = already_proposed
        self._settled_columns = settled_columns
        self._cache: EmbeddingCache | None = None
        self._model_id = model_runtime.COL_ENCODER_ID

    def warm_up(self) -> None:
        """Resolve the encoder or self-disable; loads lazily, never downloads."""
        ok, reason = model_runtime.availability(self._model_id)
        if not ok:
            raise BackendUnavailable(reason)
        encoder = model_runtime.get_encoder(self._model_id)
        self._cache = EmbeddingCache(encoder, capacity=self._config.semantic_embedding_cache_size)

    def propose(
        self, df: pd.DataFrame, ctx: SemanticContext, budget: Budget
    ) -> list[SemanticProposal]:
        if self._cache is None:  # pragma: no cover - gather always warms up first
            raise BackendUnavailable("embedding backend used before warm_up()")
        proposals: list[SemanticProposal] = []
        for column in df.columns:
            if budget.exhausted:
                break
            info = ctx.info(str(column))
            if info is None or not self._eligible(str(column), info, ctx):
                continue
            counts = self._distinct_counts(df[column])
            if counts is None or not 2 <= len(counts) <= ctx.max_distinct_values:
                continue
            if info.allowed_values:
                allowed = [v for v in info.allowed_values if isinstance(v, str)]
                if not allowed:
                    continue
                if not budget.try_column(len(counts)):
                    break
                proposals.extend(self._reference_proposals(str(column), info, counts, allowed, budget))
            elif self._clusterable(info, len(counts)):
                if not budget.try_column(len(counts)):
                    break
                proposals.extend(self._cluster_proposals(str(column), info, counts, budget))
        return proposals

    # -- eligibility ---------------------------------------------------------

    def _eligible(self, column: str, info: SemanticColumnInfo, ctx: SemanticContext) -> bool:
        if info.mutable is False or info.preserve or column in ctx.preserve_columns:
            return False  # protected: the model never sees these values
        if column in ctx.id_columns or info.identifier_like or info.role == "id":
            return False
        if ctx.target_column == column or info.role == "target":
            return False
        if info.free_text or info.high_cardinality:
            return False
        if info.numeric_like and not info.allowed_values:
            return False
        if column in self._settled_columns:
            return False  # deterministic already produced an auto-eligible repair
        return True

    def _clusterable(self, info: SemanticColumnInfo, n_distinct: int) -> bool:
        if n_distinct > _CLUSTER_MAX_DISTINCT:
            return False
        return info.role == "categorical" or info.boolean_like or info.semantic_type == "category"

    @staticmethod
    def _distinct_counts(series: pd.Series) -> list[tuple[str, int]] | None:
        """Distinct string values with counts, deterministically ordered."""
        try:
            counts = series.value_counts(dropna=True)
        except TypeError:  # unhashable payloads
            return None
        out = [(v, int(n)) for v, n in counts.items() if isinstance(v, str) and v.strip()]
        # value_counts order ties are insertion-dependent; sort for determinism.
        out.sort(key=lambda item: (-item[1], item[0]))
        return out

    # -- encoding ------------------------------------------------------------

    def _encode(self, texts: list[str], budget: Budget) -> np.ndarray | None:
        assert self._cache is not None
        if not budget.try_model_call():
            return None
        return self._cache.encode(texts)

    def _evidence(self, cosine: float, margin: float, second: str | None) -> tuple:
        encoder = self._cache.encoder if self._cache is not None else None
        sha = getattr(encoder, "model_sha256", "?")
        items = [
            SemanticEvidence(
                "embedding",
                f"cos={cosine:.3f} margin={margin:.3f} model={self._model_id} sha={sha}",
                0.0,
            )
        ]
        if second is not None:
            items.append(
                SemanticEvidence("embedding_candidates", f"second_candidate={second!r}", 0.0)
            )
        return tuple(items)

    # -- allowed-values (reference) proposals ---------------------------------

    def _reference_proposals(
        self,
        column: str,
        info: SemanticColumnInfo,
        counts: list[tuple[str, int]],
        allowed: list[str],
        budget: Budget,
    ) -> list[SemanticProposal]:
        canonical = {a.strip().casefold(): a for a in allowed}
        residual = [
            v
            for v, _ in counts
            if v.strip().casefold() not in canonical and (column, v) not in self._already_proposed
        ]
        if not residual:
            return []
        vectors = self._encode(residual + allowed, budget)
        if vectors is None:
            return []
        value_vecs, allowed_vecs = vectors[: len(residual)], vectors[len(residual) :]
        similarity = value_vecs @ allowed_vecs.T  # unit vectors -> cosine
        count_of = dict(counts)
        out: list[SemanticProposal] = []
        for i, value in enumerate(residual):
            ranked = np.argsort(-similarity[i])
            top = float(similarity[i][ranked[0]])
            second = float(similarity[i][ranked[1]]) if len(allowed) > 1 else -1.0
            margin = top - second
            if top < _REFERENCE_MIN_COSINE or margin < _MARGIN:
                continue  # unknown or ambiguous: abstain, never auto-apply
            target = allowed[int(ranked[0])]
            second_name = allowed[int(ranked[1])] if len(allowed) > 1 else None
            base = min(0.98, 0.55 + 0.35 * top + 1.5 * min(margin, 0.10))
            out.append(
                make_proposal(
                    column=column,
                    raw_value=value,
                    proposed_value=target,
                    issue_type="reference_value",
                    expert="embedding",
                    base_confidence=base,
                    evidence=self._evidence(top, margin, second_name),
                    count=count_of.get(value, 0),
                    rationale=(
                        f"{value!r} matches allowed value {target!r} by embedding "
                        f"similarity (cos={top:.3f}, margin={margin:.3f})"
                    ),
                    info=info,
                    backend=self.name,
                )
            )
        return out

    # -- clustering (suggest-only) proposals ----------------------------------

    def _cluster_proposals(
        self,
        column: str,
        info: SemanticColumnInfo,
        counts: list[tuple[str, int]],
        budget: Budget,
    ) -> list[SemanticProposal]:
        values = [v for v, _ in counts]
        vectors = self._encode(values, budget)
        if vectors is None:
            return []
        similarity = vectors @ vectors.T
        np.fill_diagonal(similarity, -1.0)
        count_of = dict(counts)
        out: list[SemanticProposal] = []
        for i, value in enumerate(values):
            if (column, value) in self._already_proposed:
                continue
            ranked = np.argsort(-similarity[i])
            top = float(similarity[i][ranked[0]])
            second = float(similarity[i][ranked[1]]) if len(values) > 2 else -1.0
            margin = top - second
            if top < _CLUSTER_MIN_COSINE or margin < _MARGIN:
                continue  # dissimilar or ambiguous cluster: abstain
            target = values[int(ranked[0])]
            # The rarer spelling maps onto the modal one, never the reverse.
            if count_of.get(value, 0) >= count_of.get(target, 0):
                continue
            second_name = values[int(ranked[1])] if len(values) > 2 else None
            base = min(0.93, 0.50 + 0.35 * top + 1.0 * min(margin, 0.10))
            out.append(
                make_proposal(
                    column=column,
                    raw_value=value,
                    proposed_value=target,
                    issue_type="category_synonym",
                    expert="embedding",
                    base_confidence=base,
                    evidence=self._evidence(top, margin, second_name),
                    count=count_of.get(value, 0),
                    rationale=(
                        f"{value!r} clusters with modal category {target!r} by embedding "
                        f"similarity (cos={top:.3f}, margin={margin:.3f}); no allowed-values "
                        "evidence, so this is suggest-only"
                    ),
                    # No risk_override: risk_for() already floors category_synonym
                    # at "medium" (and raises low-confidence clusters to "high").
                    info=info,
                    backend=self.name,
                )
            )
        return out
