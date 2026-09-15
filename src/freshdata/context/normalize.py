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


#: A quoted value that starts and ends at a token boundary, so an apostrophe
#: inside a word (``Don't``) never opens a quote.
_QUOTED_VALUE = re.compile(
    r"(?<![^\s,;:(\[{])"
    r"(?:\"[^\"]*\"|'[^']*'|`[^`]*`|“[^”]*”|‘[^’]*’)"
    r"(?![^\s,;:.)\]}])"
)
_PLACEHOLDER = re.compile("\x00(\\d+)\x00")
_LIST_SEPARATOR = re.compile(r"[,;]")
#: ``and``/``or`` between two items (lower-case only, so ``OR`` stays a value).
_CONJUNCTION = re.compile(r"\s+(?:and|or)\s+")
_LEADING_CONJUNCTION = re.compile(r"^(?:and|or)\s+")
#: The last ``and``/``or`` in an item (greedy head), for ``a, b or c``.
_TRAILING_CONJUNCTION = re.compile(r"^(?P<head>.*\S)\s+(?:and|or)\s+(?P<tail>\S.*)$")


def split_value_list(raw: str) -> tuple[str, ...]:
    """Split ``"active, inactive or pending"`` into cleaned value tokens.

    Splitting rules, applied in order:

    - A quoted value (``'N/A'``, ``"Trinidad and Tobago"``) is always kept
      whole, and ``/`` never separates values.
    - When the list contains ``,`` or ``;``, only those separate values; the
      last item may still split once on a trailing ``and``/``or``
      (``a, b or c``), and a leading ``and``/``or`` (``a, b, and c``) is
      dropped.
    - Without ``,``/``;``, ``and``/``or`` separate values
      (``email and phone``).

    A value that itself contains ``and``/``or`` in the position of a list
    separator is ambiguous; quote it.

    >>> split_value_list("Trinidad and Tobago, Chile, N/A")
    ('Trinidad and Tobago', 'Chile', 'N/A')
    >>> split_value_list("active, inactive or pending")
    ('active', 'inactive', 'pending')
    >>> split_value_list("'Bosnia and Herzegovina' or Chile")
    ('Bosnia and Herzegovina', 'Chile')
    """
    quoted: list[str] = []

    def _stash(match: re.Match[str]) -> str:
        quoted.append(match.group(0)[1:-1].strip())
        return f"\x00{len(quoted) - 1}\x00"

    text = re.sub(r"[{}\[\]()]", "", _QUOTED_VALUE.sub(_stash, raw))
    if _LIST_SEPARATOR.search(text):
        parts = [p.strip() for p in _LIST_SEPARATOR.split(text)]
        last = _LEADING_CONJUNCTION.sub("", parts.pop())
        split_last = _TRAILING_CONJUNCTION.match(last)
        if split_last is None:
            parts.append(last)
        else:
            parts.extend([split_last.group("head"), split_last.group("tail")])
    else:
        parts = _CONJUNCTION.split(text)

    def _restore(part: str) -> str:
        return _PLACEHOLDER.sub(lambda m: quoted[int(m.group(1))], strip_quotes(part))

    return tuple(v for v in (_restore(p) for p in parts) if v)


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
