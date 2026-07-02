"""Deterministic text normalization shared by the parser and the resolver.

The column-name normalization here must agree with the pipeline's own
snake_casing (:func:`freshdata.steps.columns.snake_case`) so that references
resolve against the *effective* post-``column_names`` schema. The regexes are
duplicated rather than imported to keep this package importable without pandas.
"""

from __future__ import annotations

import re

_CAMEL_LOWER_UPPER = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_ACRONYM = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD = re.compile(r"\W+", re.UNICODE)

#: Surrounding quotes/backticks users often put around column names.
_QUOTES = "\"'`“”‘’"


def snake_ref(name: str) -> str:
    """snake_case one reference the same way the cleaning pipeline renames columns.

    >>> snake_ref("CustomerID"), snake_ref(" Phone numbers ")
    ('customer_id', 'phone_numbers')
    """
    s = _CAMEL_ACRONYM.sub("_", _CAMEL_LOWER_UPPER.sub("_", name.strip()))
    s = _NON_WORD.sub("_", s).strip("_").lower()
    return re.sub(r"__+", "_", s)


def strip_quotes(ref: str) -> str:
    """Remove surrounding quotes and stray punctuation from a column phrase."""
    text = ref
    while True:
        stripped = text.strip().strip(_QUOTES).rstrip(".,;:")
        if stripped == text:
            return text
        text = stripped


def singular(token: str) -> str:
    """Cheap, deterministic English singularization for column-name tokens.

    Only the patterns that matter for column vocabulary: ``emails -> email``,
    ``addresses -> address``, ``countries -> country``. Never touches short
    words or words ending in ``ss``.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("sses") or token.endswith("shes") or token.endswith("ches"):
        return token[:-2]
    if token.endswith("ss"):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def singular_ref(snake: str) -> str:
    """Singularize each ``_``-separated token of an already-snake_cased ref."""
    return "_".join(singular(t) for t in snake.split("_") if t)


def tokens(snake: str) -> tuple[str, ...]:
    """Singularized tokens of a snake_cased name (for subset matching)."""
    return tuple(singular(t) for t in snake.split("_") if t)


def split_value_list(raw: str) -> tuple[str, ...]:
    """Split ``"active, inactive or pending"`` into cleaned value tokens."""
    cleaned = re.sub(r"[{}\[\]()]", "", raw)
    parts = re.split(r",|;|\bor\b|\band\b|/", cleaned)
    return tuple(p for p in (strip_quotes(x) for x in parts) if p)


def parse_scalar(raw: str) -> object:
    """Parse ``"18"``/``"1.5"``/``"-3"`` into a number; anything else stays str."""
    text = strip_quotes(raw)
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text
