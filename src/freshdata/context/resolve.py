"""Column-reference resolution: user phrases -> actual schema columns.

A fixed ladder, cheapest and most trustworthy first:

1. exact label match,
2. snake_case-normalized match (the pipeline's own renaming),
3. alias-lexicon match (``"CustomerID"`` -> ``cust_id``),
4. token-subset match (``"revenue"`` -> ``monthly_revenue``),
5. ``difflib`` similarity at a threshold (default 0.85),
6. optional embedding cosine similarity (only when the caller injects a
   scorer — this module never imports the model runtime, so ``context/``
   stays model-free; without a scorer, behavior is byte-identical to the
   deterministic ladder).

Two candidates that score within ``ambiguity_margin`` of each other are never
chosen between silently — the reference comes back unresolved with the ranked
candidates attached so the user can disambiguate. The embedding rung follows
the same rule and can only *rescue* references the deterministic rungs gave up
on; it can never override an exact/normalized/alias/token/difflib match.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

from .lexicon import alias_group
from .normalize import singular_ref, snake_ref, tokens

#: Minimum difflib ratio for a fuzzy match.
DEFAULT_THRESHOLD = 0.85
#: Two candidates closer than this are ambiguous.
AMBIGUITY_MARGIN = 0.05
#: Minimum cosine similarity for the optional embedding rung.
EMBEDDING_THRESHOLD = 0.60

#: An injected embedding scorer: (ref, columns) -> ranked (column, cosine).
EmbeddingScorer = Callable[[str, Sequence[str]], "list[tuple[str, float]]"]

#: Confidence assigned per ladder rung (fuzzy matches use their actual ratio).
_EXACT_CONFIDENCE = 1.0
_SNAKE_CONFIDENCE = 1.0
_ALIAS_CONFIDENCE = 0.90
_TOKEN_CONFIDENCE = 0.87


@dataclass(frozen=True)
class Resolution:
    """Outcome of resolving one reference against one schema."""

    ref: str
    column: str | None
    #: "exact" | "normalized" | "alias" | "tokens" | "difflib" | "embedding"
    #: | "unresolved"
    method: str
    confidence: float
    #: Ranked ``(column, score)`` evidence; for unresolved refs, the shortlist.
    candidates: tuple[tuple[str, float], ...] = ()
    reason: str = ""
    #: The encoder that produced an "embedding" resolution (None otherwise).
    model_id: str | None = None

    @property
    def resolved(self) -> bool:
        return self.column is not None


def _alias_matches(ref: str, columns: list[str]) -> list[str]:
    """Columns whose alias group equals the reference's alias group."""
    group = alias_group(ref)
    if group is None:
        return []
    return [col for col in columns if alias_group(col) == group]


def _token_matches(ref: str, columns: list[str]) -> list[str]:
    """Columns whose token set contains every token of the reference."""
    ref_tokens = set(tokens(snake_ref(ref)))
    if not ref_tokens:
        return []
    return [col for col in columns if ref_tokens <= set(tokens(snake_ref(col)))]


def _embedding_rescue(
    ref: str,
    cols: list[str],
    fallback: Resolution,
    scorer: EmbeddingScorer | None,
    model_id: str | None,
    ambiguity_margin: float,
) -> Resolution:
    """Rung 6: try the injected embedding scorer on an unresolved reference.

    Same discipline as the deterministic rungs: a near-tie stays unresolved
    (with the embedding shortlist attached as evidence), and any scorer error
    falls back to the deterministic outcome rather than raising mid-compile.
    """
    if scorer is None or not cols:
        return fallback
    try:
        ranked = sorted(scorer(ref, cols), key=lambda pair: (-pair[1], cols.index(pair[0])))
    except Exception:  # pragma: no cover - defensive: scorer bugs never break compile
        return fallback
    if not ranked:
        return fallback
    shortlist = tuple((c, round(s, 4)) for c, s in ranked[:3])
    best_col, best_score = ranked[0]
    if best_score < EMBEDDING_THRESHOLD:
        return fallback
    if len(ranked) > 1 and (best_score - ranked[1][1]) < ambiguity_margin:
        return Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=shortlist,
            reason=(
                f"{fallback.reason}; embedding also ambiguous: {best_col!r} and "
                f"{ranked[1][0]!r} score within {ambiguity_margin:.2f} of each other"
            ),
            model_id=model_id,
        )
    return Resolution(
        ref,
        best_col,
        "embedding",
        round(best_score, 4),
        candidates=shortlist,
        model_id=model_id,
    )


def resolve_reference(
    ref: str,
    columns: list[str] | tuple[str, ...],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
    embedding_scorer: EmbeddingScorer | None = None,
    embedding_model_id: str | None = None,
) -> Resolution:
    """Resolve one user reference against the effective schema *columns*.

    Deterministic; never guesses between near-ties. Column order breaks exact
    ties only when scores are strictly equal at the same ladder rung — in that
    case the reference is reported ambiguous rather than resolved.

    ``embedding_scorer`` (optional, injected by the semantic layer when the
    embedding backend is enabled and its model is available) adds a final
    rescue rung for references the deterministic ladder could not resolve; it
    follows the same ambiguity rules and never overrides earlier rungs.
    """
    cols = [str(c) for c in columns]

    # 1. Exact label.
    if ref in cols:
        return Resolution(ref, ref, "exact", _EXACT_CONFIDENCE)

    # 2. snake_case-normalized (matches the pipeline's own renaming), with a
    #    singular/plural bridge ("Emails" == "emails" -> "email" columns).
    ref_snake = snake_ref(ref)
    ref_singular = singular_ref(ref_snake)
    normalized = {col: snake_ref(col) for col in cols}
    snake_hits = [
        col
        for col, col_snake in normalized.items()
        if col_snake == ref_snake or singular_ref(col_snake) == ref_singular
    ]
    if len(snake_hits) == 1:
        return Resolution(ref, snake_hits[0], "normalized", _SNAKE_CONFIDENCE)
    if len(snake_hits) > 1:
        return Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=tuple((c, _SNAKE_CONFIDENCE) for c in snake_hits),
            reason=f"{len(snake_hits)} columns normalize to {ref_snake!r}",
        )

    # 3. Alias lexicon.
    alias_hits = _alias_matches(ref, cols)
    if len(alias_hits) == 1:
        return Resolution(
            ref,
            alias_hits[0],
            "alias",
            _ALIAS_CONFIDENCE,
            candidates=((alias_hits[0], _ALIAS_CONFIDENCE),),
        )
    if len(alias_hits) > 1:
        return Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=tuple((c, _ALIAS_CONFIDENCE) for c in alias_hits),
            reason=f"{len(alias_hits)} columns share the {alias_group(ref)!r} alias group",
        )

    # 4. Token subset ("revenue" ⊆ "monthly_revenue").
    token_hits = _token_matches(ref, cols)
    if len(token_hits) == 1:
        return Resolution(
            ref,
            token_hits[0],
            "tokens",
            _TOKEN_CONFIDENCE,
            candidates=((token_hits[0], _TOKEN_CONFIDENCE),),
        )
    if len(token_hits) > 1:
        return Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=tuple((c, _TOKEN_CONFIDENCE) for c in token_hits),
            reason=f"{len(token_hits)} columns contain all tokens of {ref!r}",
        )

    # 5. difflib similarity on the normalized forms.
    scored = sorted(
        (
            (col, difflib.SequenceMatcher(None, ref_snake, col_snake).ratio())
            for col, col_snake in normalized.items()
        ),
        key=lambda pair: (-pair[1], cols.index(pair[0])),
    )
    shortlist = tuple((c, round(s, 4)) for c, s in scored[:3])
    if not scored or scored[0][1] < threshold:
        unresolved = Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=shortlist,
            reason=f"no column matches {ref!r} (best similarity "
            f"{scored[0][1]:.2f} < {threshold:.2f})" if scored else "schema is empty",
        )
        return _embedding_rescue(
            ref, cols, unresolved, embedding_scorer, embedding_model_id, ambiguity_margin
        )
    best_col, best_score = scored[0]
    if len(scored) > 1 and (best_score - scored[1][1]) < ambiguity_margin:
        unresolved = Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=shortlist,
            reason=f"ambiguous: {best_col!r} and {scored[1][0]!r} score within "
            f"{ambiguity_margin:.2f} of each other",
        )
        return _embedding_rescue(
            ref, cols, unresolved, embedding_scorer, embedding_model_id, ambiguity_margin
        )
    return Resolution(ref, best_col, "difflib", round(best_score, 4), candidates=shortlist)
