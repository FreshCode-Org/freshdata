"""Tier-0 deterministic parser: sentences in, intent candidates out.

The parser never guesses. Each sentence either matches exactly one lexicon
pattern (first match in a fixed order wins) and becomes an
:class:`~freshdata.context.types.IntentCandidate`, or it becomes an
:class:`~freshdata.context.types.UnparsedSentence`. It never resolves column
names, never reads data, and never touches config — that is the compiler's job.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from .lexicon import FORMAT_PATTERN, REGION_PATTERN, format_for, parse_confidence, region_code
from .normalize import parse_scalar, snake_ref, split_value_list, strip_quotes

#: A column phrase: word-ish tokens, possibly multi-word ("Phone numbers").
_COL = r"[A-Za-z_][A-Za-z0-9_]*(?:[ .\-][A-Za-z0-9_]+)*"
#: A number literal (int or float, optionally signed).
_NUM = r"-?\d+(?:\.\d+)?"

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
#: Sentence enders; a period *between* digits (a decimal like 0.95) never splits.
_SPLIT = re.compile(r"[!?;]|(?<=\D)\.|\.(?=\D|$)")
_WS = re.compile(r"\s+")


def split_sentences(text: str) -> tuple[str, ...]:
    """Deterministically split context prose into candidate sentences.

    Lines are split first (a newline always ends a sentence), then sentence
    punctuation within a line. Bullets/numbering are stripped; whitespace is
    collapsed; empty fragments are dropped.
    """
    out: list[str] = []
    for raw_line in text.splitlines():
        line = _BULLET.sub("", raw_line)
        for fragment in _SPLIT.split(line):
            sentence = _WS.sub(" ", fragment).strip()
            if sentence:
                out.append(sentence)
    return tuple(out)


@dataclass(frozen=True)
class _Match:
    """Intermediate parse result for one sentence."""

    intent: str
    column_refs: tuple[str, ...]
    params: dict[str, object]


_Builder = Callable[["re.Match[str]"], "_Match | None"]
_RULES: list[tuple[re.Pattern[str], _Builder]] = []


def _rule(pattern: str) -> Callable[[_Builder], _Builder]:
    def register(builder: _Builder) -> _Builder:
        _RULES.append((re.compile(pattern, re.IGNORECASE), builder))
        return builder

    return register


def _col(m: re.Match[str], group: str = "col") -> str:
    return strip_quotes(m.group(group))


# -- 1. DOMAIN ---------------------------------------------------------------


@_rule(rf"^this is (?:a|an|the)\s+(?P<dom>{_COL})\s+(?:dataset|data set|data|table)$")
def _domain_this_is(m: re.Match[str]) -> _Match:
    return _Match("domain", (), {"name": snake_ref(m.group("dom"))})


@_rule(
    rf"^(?:the )?dataset (?:is|contains|describes|holds)\s+(?P<dom>{_COL})"
    rf"(?:\s+(?:data|records|rows))?$"
)
def _domain_dataset(m: re.Match[str]) -> _Match:
    return _Match("domain", (), {"name": snake_ref(m.group("dom"))})


@_rule(r"^domain\s*[:=]\s*(?P<dom>.+)$")
def _domain_explicit(m: re.Match[str]) -> _Match:
    return _Match("domain", (), {"name": snake_ref(m.group("dom"))})


# -- 2. PROTECT (before anything that could read "modify X" as a column op) --

_PROTECT_VERB = r"(?:modify|change|alter|touch|edit|update|overwrite|impute|fill)"


@_rule(rf"^never {_PROTECT_VERB}\s+(?:the\s+)?(?P<col>{_COL}?)(?:\s+(?:values|column|field))?$")
@_rule(
    rf"^do(?:\s+not|n'?t) {_PROTECT_VERB}\s+(?:the\s+)?(?P<col>{_COL}?)"
    rf"(?:\s+(?:values|column|field))?$"
)
def _protect_never(m: re.Match[str]) -> _Match:
    return _Match("protected", (_col(m),), {})


@_rule(
    rf"^leave\s+(?:the\s+)?(?P<col>{_COL}?)(?:\s+(?:values|column|field))?"
    rf"\s+(?:alone|as[- ]is|unchanged|untouched)$"
)
@_rule(
    rf"^keep\s+(?:the\s+)?(?P<col>{_COL}?)(?:\s+(?:values|column|field))?"
    rf"\s+(?:unchanged|untouched|as[- ]is|intact)$"
)
@_rule(
    rf"^(?P<col>{_COL})\s+(?:is|are|must (?:remain|stay)|should (?:remain|stay))"
    rf"\s+read[- ]?only$"
)
@_rule(rf"^protect\s+(?:the\s+)?(?P<col>{_COL}?)(?:\s+(?:values|column|field))?$")
def _protect_leave(m: re.Match[str]) -> _Match:
    return _Match("protected", (_col(m),), {})


# -- 3. IMPUTE_IF ------------------------------------------------------------


@_rule(
    rf"^missing\s+(?P<col>{_COL}?)(?:\s+values)?\s+"
    rf"(?:should|may|can|must)\s+(?:only\s+)?be\s+"
    rf"(?:estimated|imputed|filled(?:\s+in)?|predicted|inferred)"
    rf"(?:\s*,?\s*(?:but\s+)?only)?(?:\s+(?:if|when)\s+(?P<cond>.+))?$"
)
@_rule(
    rf"^(?:estimate|impute|fill(?:\s+in)?|predict|infer)\s+missing\s+(?P<col>{_COL}?)"
    rf"(?:\s+values)?(?:\s*,?\s*(?:but\s+)?only)?(?:\s+(?:if|when)\s+(?P<cond>.+))?$"
)
def _impute_if(m: re.Match[str]) -> _Match | None:
    params: dict[str, object] = {}
    cond = m.group("cond")
    if cond is not None:
        confidence = parse_confidence(cond)
        if confidence is None:
            return None  # a condition we cannot represent -> unparsed, not guessed
        params["min_confidence"] = confidence
    return _Match("impute_if", (_col(m),), params)


# -- 4. DEDUP_KEY ------------------------------------------------------------


@_rule(r"^(?:de-?dup(?:licat)?e)\s+(?:rows\s+)?(?:by|on|using)\s+(?P<cols>.+)$")
@_rule(r"^duplicates are (?:identified|keyed|detected)\s+by\s+(?P<cols>.+)$")
@_rule(r"^use\s+(?P<cols>.+?)\s+as (?:the )?(?:de-?dup(?:licat)?e?(?:ion)?|duplicate)\s+key$")
def _dedup_key(m: re.Match[str]) -> _Match:
    cols = split_value_list(m.group("cols"))
    return _Match("dedup_key", cols, {})


# -- 5. ALLOWED_VALUES -------------------------------------------------------


@_rule(rf"^allowed\s+(?P<col>{_COL}?)\s+values\s+are:?\s+(?P<vals>.+)$")
@_rule(rf"^valid\s+(?P<col>{_COL}?)\s+values\s+are:?\s+(?P<vals>.+)$")
@_rule(rf"^valid values for\s+(?P<col>{_COL})\s+are:?\s+(?P<vals>.+)$")
@_rule(
    rf"^(?P<col>{_COL})\s+(?:values\s+)?(?:must|should|can|may)(?:\s+only)?\s+be\s+"
    rf"(?:one of|in|among):?\s+(?P<vals>.+)$"
)
@_rule(rf"^(?P<col>{_COL})\s+can only be\s+(?P<vals>.+)$")
def _allowed_values(m: re.Match[str]) -> _Match | None:
    values = split_value_list(m.group("vals"))
    if not values:
        return None
    return _Match("allowed_values", (_col(m),), {"values": list(values)})


# -- 6. RANGE ----------------------------------------------------------------


@_rule(
    rf"^(?P<col>{_COL})\s+(?:values\s+)?(?:must|should|is|are)(?:\s+be)?\s+"
    rf"between\s+(?P<lo>{_NUM})\s+and\s+(?P<hi>{_NUM})$"
)
@_rule(
    rf"^(?P<col>{_COL})\s+(?:ranges?|values range)\s+from\s+(?P<lo>{_NUM})"
    rf"\s+to\s+(?P<hi>{_NUM})$"
)
def _range_between(m: re.Match[str]) -> _Match:
    return _Match(
        "range",
        (_col(m),),
        {"lo": parse_scalar(m.group("lo")), "hi": parse_scalar(m.group("hi"))},
    )


@_rule(rf"^(?P<col>{_COL})\s+(?:must|should)\s+be\s+at least\s+(?P<lo>{_NUM})$")
def _range_lo(m: re.Match[str]) -> _Match:
    return _Match("range", (_col(m),), {"lo": parse_scalar(m.group("lo")), "hi": None})


@_rule(rf"^(?P<col>{_COL})\s+(?:must|should)\s+(?:be\s+at most|not exceed)\s+(?P<hi>{_NUM})$")
def _range_hi(m: re.Match[str]) -> _Match:
    return _Match("range", (_col(m),), {"lo": None, "hi": parse_scalar(m.group("hi"))})


# -- 7. LOCALE_FORMAT (before UNIQUE/VALID so "are Indian" wins "are X") -----


@_rule(
    rf"^(?P<col>{_COL})\s+(?:numbers\s+)?(?:are|is)\s+(?:all\s+)?(?P<region>{REGION_PATTERN})"
    rf"(?:\s+(?P<fmt>{FORMAT_PATTERN}))?$"
)
@_rule(
    rf"^(?P<col>{_COL})\s+(?:are|is)\s+in\s+(?P<region>{REGION_PATTERN})\s+format$"
)
def _locale_format(m: re.Match[str]) -> _Match:
    col = _col(m)
    fmt_word = m.groupdict().get("fmt")
    fmt = format_for(fmt_word) if fmt_word else format_for(col)
    return _Match(
        "locale_format",
        (col,),
        {"format": fmt, "region": region_code(m.group("region"))},
    )


# -- 8. UNIQUE ---------------------------------------------------------------


@_rule(
    rf"^(?:each\s+|every\s+)?(?P<col>{_COL})\s+"
    rf"(?:is|are|must be|should be|values? (?:are|must be|should be))\s+unique$"
)
@_rule(rf"^no duplicate\s+(?P<col>{_COL}?)(?:\s+values)?(?:\s+allowed)?$")
@_rule(rf"^(?P<col>{_COL})\s+must not (?:repeat|contain duplicates)$")
def _unique(m: re.Match[str]) -> _Match:
    return _Match("unique", (_col(m),), {})


# -- 9. VALID_FORMAT ---------------------------------------------------------


@_rule(
    rf"^(?P<col>{_COL})\s+(?:must|should)\s+(?:be|contain|hold)\s+"
    rf"(?:a\s+)?valid(?:\s+(?P<fmt>{FORMAT_PATTERN}))?(?:\s+(?:values|addresses|entries))?$"
)
@_rule(
    rf"^(?P<col>{_COL})\s+(?:are|is|contains?)\s+(?:valid\s+)?(?P<fmt>{FORMAT_PATTERN})"
    rf"(?:\s+(?:values|addresses|entries))?$"
)
def _valid_format(m: re.Match[str]) -> _Match | None:
    col = _col(m)
    fmt_word = m.groupdict().get("fmt")
    fmt = format_for(fmt_word) if fmt_word else format_for(col)
    if fmt is None:
        return None  # "X must be valid" with no known format is not guessable
    return _Match("valid_format", (col,), {"format": fmt})


# -- 10. DROP_IF -------------------------------------------------------------

_DROP_COND = r"(?:missing|null|empty|blank|na|n/a|negative|zero|invalid)"


@_rule(
    rf"^(?:drop|remove|delete|discard)\s+rows\s+(?:where|if|when|whose)\s+"
    rf"(?P<col>{_COL})\s+(?:is|are|value is)\s+(?P<cond>{_DROP_COND})$"
)
@_rule(
    rf"^(?:drop|remove|delete|discard)\s+rows\s+with\s+(?P<cond>{_DROP_COND})\s+"
    rf"(?P<col>{_COL})(?:\s+values)?$"
)
def _drop_if(m: re.Match[str]) -> _Match:
    cond = m.group("cond").lower()
    if cond in ("null", "empty", "blank", "na", "n/a"):
        cond = "missing"
    return _Match("drop_if", (_col(m),), {"condition": cond})


# -- 11. RENAME --------------------------------------------------------------


@_rule(rf"^rename\s+(?P<col>{_COL})\s+to\s+(?P<new>{_COL})$")
@_rule(rf"^(?P<col>{_COL})\s+should be (?:renamed|called)\s+(?:to\s+)?(?P<new>{_COL})$")
def _rename(m: re.Match[str]) -> _Match:
    return _Match("rename", (_col(m),), {"new_name": snake_ref(m.group("new"))})


# -- 12. MAP -----------------------------------------------------------------

_VAL = r"(?:\"[^\"]+\"|'[^']+'|[\w./+-]+)"


@_rule(rf"^map\s+(?P<a>{_VAL})\s+to\s+(?P<b>{_VAL})\s+in\s+(?P<col>{_COL})$")
@_rule(rf"^replace\s+(?P<a>{_VAL})\s+with\s+(?P<b>{_VAL})\s+in\s+(?P<col>{_COL})$")
@_rule(
    rf"^in\s+(?P<col>{_COL}),?\s+(?:map|replace|treat)\s+(?P<a>{_VAL})\s+"
    rf"(?:with|as|to|->)\s+(?P<b>{_VAL})$"
)
def _map_single(m: re.Match[str]) -> _Match:
    key = strip_quotes(m.group("a"))
    return _Match("map", (_col(m),), {"mapping": {key: strip_quotes(m.group("b"))}})


@_rule(rf"^map\s+(?P<col>{_COL})\s+values?\s*:?\s+(?P<pairs>.+)$")
def _map_pairs(m: re.Match[str]) -> _Match | None:
    mapping: dict[str, object] = {}
    for chunk in re.split(r",|;", m.group("pairs")):
        pair = re.split(r"->|=>|=|\bto\b", chunk, maxsplit=1)
        if len(pair) != 2:
            continue
        key, value = strip_quotes(pair[0]), strip_quotes(pair[1])
        if key and value:
            mapping[key] = value
    if not mapping:
        return None
    return _Match("map", (_col(m),), {"mapping": mapping})


# ---------------------------------------------------------------------------


_COURTESY = re.compile(r"^(?:please|kindly)[, ]+", re.IGNORECASE)


def _match_sentence(sentence: str) -> _Match | None:
    """First lexicon rule that matches wins; ``None`` means unparsed."""
    stripped = _COURTESY.sub("", sentence.rstrip("."))
    for pattern, builder in _RULES:
        m = pattern.match(stripped)
        if m is not None:
            built = builder(m)
            if built is not None:
                return built
    return None


@dataclass(frozen=True)
class ParseResult:
    """Everything the parser extracted from one context text."""

    candidates: tuple[object, ...]  # tuple[IntentCandidate, ...]
    unparsed: tuple[object, ...]  # tuple[UnparsedSentence, ...]
    sentences: tuple[str, ...]

    def __iter__(self) -> Iterator[object]:
        return iter(self.candidates)


def parse_context(text: str) -> ParseResult:
    """Parse free-form context prose into intent candidates.

    Deterministic and model-free: identical text always produces an identical
    result. Sentences that match no pattern are returned as
    :class:`UnparsedSentence` — never silently dropped.
    """
    from .types import IntentCandidate, Provenance, UnparsedSentence  # noqa: PLC0415

    candidates: list[IntentCandidate] = []
    unparsed: list[UnparsedSentence] = []
    sentences = split_sentences(text)
    for sentence in sentences:
        match = _match_sentence(sentence)
        if match is None:
            unparsed.append(UnparsedSentence(sentence=sentence))
            continue
        candidates.append(
            IntentCandidate(
                intent=match.intent,
                column_refs=match.column_refs,
                params=match.params,
                provenance=Provenance(sentence=sentence, tier=0),
            )
        )
    return ParseResult(
        candidates=tuple(candidates),
        unparsed=tuple(unparsed),
        sentences=sentences,
    )
