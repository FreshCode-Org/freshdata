"""Embedding plumbing shared by non-backend consumers (resolver, roles).

Bridges the model runtime into places that must stay model-free at import
time: the context compiler asks for a *scorer* through this module and gets
``(None, None)`` unless the user opted into the embedding backend AND its
encoder is actually available (extra installed + model pulled, or a test
encoder injected). Nothing here downloads anything.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Callable

from ..models.runtime import COL_ENCODER_ID, availability, get_encoder
from .cache import EmbeddingCache

#: (ref, columns) -> ranked (column, cosine) — the resolver-rung contract.
ResolverScorer = Callable[[str, Sequence[str]], "list[tuple[str, float]]"]


def _wants_embedding(config: object) -> bool:
    backends = getattr(config, "semantic_backends", ()) or ()
    return "embedding" in tuple(backends)


def resolver_scorer(config: object) -> tuple[ResolverScorer | None, str | None]:
    """Build the column-matching scorer for the resolver's embedding rung.

    Returns ``(None, None)`` — deterministic ladder only — unless
    ``"embedding"`` is in ``semantic_backends`` and the encoder is available.
    Column-name similarity uses snake_case-normalized names so the model sees
    the same shape the deterministic rungs compare.
    """
    if not _wants_embedding(config):
        return None, None
    ok, _reason = availability(COL_ENCODER_ID)
    if not ok:
        return None, None
    encoder = get_encoder(COL_ENCODER_ID)
    capacity = int(getattr(config, "semantic_embedding_cache_size", 65_536) or 0)
    cache = EmbeddingCache(encoder, capacity=capacity)

    from ..context.normalize import snake_ref  # noqa: PLC0415 - avoid import cycle

    def scorer(ref: str, columns: Sequence[str]) -> list[tuple[str, float]]:
        names = [str(c) for c in columns]
        texts = [snake_ref(ref).replace("_", " ")] + [
            snake_ref(name).replace("_", " ") for name in names
        ]
        vectors = cache.encode(texts)
        ref_vec, col_vecs = vectors[0], vectors[1:]
        sims = col_vecs @ ref_vec
        return [(name, float(sim)) for name, sim in zip(names, sims)]

    return scorer, COL_ENCODER_ID
