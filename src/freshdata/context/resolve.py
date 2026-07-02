"""Column-reference resolution: user phrases -> actual schema columns.

A fixed ladder, cheapest and most trustworthy first:

1. exact label match,
2. snake_case-normalized match (the pipeline's own renaming),
3. alias-lexicon match (``"CustomerID"`` -> ``cust_id``),
4. token-subset match (``"revenue"`` -> ``monthly_revenue``),
5. ``difflib`` similarity at a threshold (default 0.85).

Two candidates that score within ``ambiguity_margin`` of each other are never
chosen between silently — the reference comes back unresolved with the ranked
candidates attached so the user can disambiguate.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .lexicon import alias_group
from .normalize import singular_ref, snake_ref, tokens

#: Minimum difflib ratio for a fuzzy match.
DEFAULT_THRESHOLD = 0.85
#: Two candidates closer than this are ambiguous.
AMBIGUITY_MARGIN = 0.05

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
    method: str  # "exact" | "normalized" | "alias" | "tokens" | "difflib" | "unresolved"
    confidence: float
    #: Ranked ``(column, score)`` evidence; for unresolved refs, the shortlist.
    candidates: tuple[tuple[str, float], ...] = ()
    reason: str = ""

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


def resolve_reference(
    ref: str,
    columns: list[str] | tuple[str, ...],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    ambiguity_margin: float = AMBIGUITY_MARGIN,
) -> Resolution:
    """Resolve one user reference against the effective schema *columns*.

    Deterministic; never guesses between near-ties. Column order breaks exact
    ties only when scores are strictly equal at the same ladder rung — in that
    case the reference is reported ambiguous rather than resolved.
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
        return Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=shortlist,
            reason=f"no column matches {ref!r} (best similarity "
            f"{scored[0][1]:.2f} < {threshold:.2f})" if scored else "schema is empty",
        )
    best_col, best_score = scored[0]
    if len(scored) > 1 and (best_score - scored[1][1]) < ambiguity_margin:
        return Resolution(
            ref,
            None,
            "unresolved",
            0.0,
            candidates=shortlist,
            reason=f"ambiguous: {best_col!r} and {scored[1][0]!r} score within "
            f"{ambiguity_margin:.2f} of each other",
        )
    return Resolution(ref, best_col, "difflib", round(best_score, 4), candidates=shortlist)
