"""Semantic-type inference over distinct value samples (Phase 3, additive).

Classifies a column into a fixed taxonomy from three signal tiers, strongest
first:

1. an explicit user/policy hint (``semantic_context`` / ``ContextPolicy``
   lowering) — always wins, confidence 1.0;
2. deterministic content detectors over a distinct-value sample, corroborated
   by name heuristics;
3. an optional embedding vote — evidence only: it can raise confidence in an
   undetected type but is *vetoed* by any conflicting content detector, so a
   model label can never certify a type the values contradict.

The default cleaning path does not consume this module — it feeds
``fd.infer_roles`` (three additive columns) and the embedding backend, so
inference here can never change what a model-free install repairs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .types import SemanticEvidence

SEMANTIC_TYPES = (
    "email",
    "phone",
    "url",
    "country",
    "currency_amount",
    "quantity_with_unit",
    "person_name",
    "address",
    "city",
    "postal_code",
    "national_id",
    "category_code",
    "free_text",
    "boolean_like",
    "date_like",
    "identifier",
    "unknown",
)

#: Columns with fewer non-null distinct values than this are "unknown": there
#: is not enough support to certify any semantic type.
MIN_DISTINCT_SUPPORT = 5

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_URL_RE = re.compile(r"^https?://\S+\.\S+", re.I)
_PHONE_RE = re.compile(r"^\+?[\d\s\-().]{7,17}$")
_POSTAL_RE = re.compile(r"^\d{5,6}(-\d{4})?$")
_CATEGORY_CODE_RE = re.compile(r"^[A-Z]{1,4}[-_]?\d{0,4}$")
_PERSON_NAME_RE = re.compile(r"^[A-Za-z][a-z]+(?: [A-Za-z][a-z]+){1,3}\.?$")
_ADDRESS_HINT_RE = re.compile(
    r"\d+.*\b(st|street|road|rd|ave|avenue|lane|ln|block|sector|nagar|marg)\b", re.I
)
_CURRENCY_RE = re.compile(
    r"^[₹$€£¥]\s?-?[\d,]+(\.\d+)?$|^-?[\d,]+(\.\d+)?\s?(inr|usd|eur|gbp)$", re.I
)
_QUANTITY_RE = re.compile(r"^-?[\d.,]+\s?(kg|g|mg|km|m|cm|mm|l|ml|s|ms|h|hz|kwh|mw|%)$", re.I)
_BOOL_VALUES = frozenset(
    {"true", "false", "yes", "no", "y", "n", "0", "1", "t", "f", "on", "off"}
)
_DATE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}([ T].*)?$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$|^\d{1,2} \w{3,9},? \d{4}$"
)

_COUNTRIES = frozenset(
    [
        "india", "china", "japan", "france", "germany", "italy", "spain", "brazil",
        "canada", "australia", "russia", "mexico", "indonesia", "pakistan",
        "bangladesh", "nigeria", "egypt", "turkey", "iran", "thailand", "vietnam",
        "argentina", "colombia", "poland", "ukraine", "kenya", "usa", "uk", "uae",
        "singapore", "malaysia", "nepal", "sri_lanka", "united_states",
        "united_kingdom", "south_africa", "new_zealand", "netherlands",
        "switzerland", "sweden", "norway", "denmark", "finland", "belgium",
        "austria", "portugal", "ireland", "greece", "israel", "qatar", "oman",
        "kuwait", "bahrain",
    ]
)

_NAME_HINTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"e[-_]?mail|email", re.I)),
    ("phone", re.compile(r"phone|mobile|mob[_ ]?no|msisdn|contact[_ ]?no", re.I)),
    ("url", re.compile(r"\burl\b|website|link|homepage", re.I)),
    ("country", re.compile(r"country|nation", re.I)),
    ("city", re.compile(r"\bcity\b|\btown\b", re.I)),
    ("postal_code", re.compile(r"zip|postal|pincode|pin[_ ]?code", re.I)),
    ("address", re.compile(r"address|addr\b|street", re.I)),
    ("person_name", re.compile(r"^(first|last|full|middle)?[_ ]?name$|person", re.I)),
    (
        "national_id",
        re.compile(r"ssn|aadhaar|aadhar|passport|pan[_ ]?(no|number)|tax[_ ]?id", re.I),
    ),
    ("currency_amount", re.compile(r"price|revenue|salary|amount|cost|fee|balance", re.I)),
    ("date_like", re.compile(r"date|_at$|_on$|timestamp|dob|birth", re.I)),
)


@dataclass(frozen=True)
class SemanticTypeResult:
    """Inferred type with a calibratable confidence and an evidence trail."""

    semantic_type: str
    confidence: float
    evidence: tuple[SemanticEvidence, ...] = ()


def _distinct_strings(series: pd.Series, sample_size: int) -> list[str]:
    try:
        sample = series.dropna()
    except TypeError:  # pragma: no cover - exotic payloads
        return []
    if len(sample) > sample_size:
        sample = sample.head(sample_size)
    try:
        distinct = sample.unique()
    except TypeError:
        return []
    return [str(v).strip() for v in distinct if str(v).strip()]


def _share(values: list[str], predicate) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if predicate(v)) / len(values)


def _content_detect(values: list[str]) -> tuple[str, float] | None:
    """The strongest deterministic content signal over the distinct sample."""
    checks: tuple[tuple[str, float], ...] = (
        ("email", _share(values, lambda v: bool(_EMAIL_RE.match(v)))),
        ("url", _share(values, lambda v: bool(_URL_RE.match(v)))),
        ("currency_amount", _share(values, lambda v: bool(_CURRENCY_RE.match(v)))),
        ("quantity_with_unit", _share(values, lambda v: bool(_QUANTITY_RE.match(v)))),
        ("date_like", _share(values, lambda v: bool(_DATE_RE.match(v)))),
        ("boolean_like", _share(values, lambda v: v.casefold() in _BOOL_VALUES)),
        ("country", _share(values, lambda v: v.casefold().replace(" ", "_") in _COUNTRIES)),
        ("postal_code", _share(values, lambda v: bool(_POSTAL_RE.match(v)))),
        (
            "phone",
            _share(
                values,
                lambda v: bool(_PHONE_RE.match(v)) and sum(ch.isdigit() for ch in v) >= 7,
            ),
        ),
        ("address", _share(values, lambda v: bool(_ADDRESS_HINT_RE.search(v)))),
        ("person_name", _share(values, lambda v: bool(_PERSON_NAME_RE.match(v)))),
        ("category_code", _share(values, lambda v: bool(_CATEGORY_CODE_RE.match(v)))),
    )
    best_type, best_share = max(checks, key=lambda pair: pair[1])
    if best_share < 0.6:
        return None
    return best_type, best_share


def _name_hint(name: str) -> str | None:
    for semantic_type, pattern in _NAME_HINTS:
        if pattern.search(name):
            return semantic_type
    return None


def infer_semantic_type(
    name: str,
    series: pd.Series,
    *,
    role: str | None = None,
    hint: str | None = None,
    sample_size: int = 10_000,
    embedding_vote: tuple[str, float] | None = None,
) -> SemanticTypeResult:
    """Infer the semantic type of one column.

    ``hint`` is an explicit user/policy declaration and always wins.
    ``embedding_vote`` — an optional ``(type, score)`` from a model head — is
    consumed as corroborating evidence only and is vetoed by any conflicting
    content detector.
    """
    if hint:
        return SemanticTypeResult(
            hint, 1.0, (SemanticEvidence("context_hint", f"declared semantic_type={hint!r}", 0.0),)
        )

    values = _distinct_strings(series, sample_size)
    evidence: list[SemanticEvidence] = []

    if role == "id":
        return SemanticTypeResult(
            "identifier", 0.9, (SemanticEvidence("column_role", "role inference: id", 0.0),)
        )
    if role == "text":
        return SemanticTypeResult(
            "free_text", 0.8, (SemanticEvidence("column_role", "role inference: text", 0.0),)
        )

    if len(values) < MIN_DISTINCT_SUPPORT:
        return SemanticTypeResult(
            "unknown",
            0.2,
            (
                SemanticEvidence(
                    "pattern", f"only {len(values)} distinct non-null value(s) sampled", 0.0
                ),
            ),
        )

    detected = _content_detect(values)
    name_type = _name_hint(name)

    if detected is not None:
        detected_type, share = detected
        confidence = 0.6 + 0.3 * share
        evidence.append(
            SemanticEvidence(
                "pattern", f"{share:.0%} of distinct values match {detected_type}", 0.0
            )
        )
        if name_type == detected_type:
            confidence += 0.05
            evidence.append(
                SemanticEvidence("pattern", f"column name corroborates {detected_type}", 0.0)
            )
        if embedding_vote is not None:
            vote_type, vote_score = embedding_vote
            if vote_type == detected_type:
                confidence += 0.02
                evidence.append(
                    SemanticEvidence(
                        "embedding", f"model head agrees confidence={vote_score:.2f}", 0.0
                    )
                )
            else:
                # Content detector veto: the conflicting model label is
                # recorded but cannot certify a contradicted type.
                evidence.append(
                    SemanticEvidence(
                        "conflict",
                        f"model head proposed {vote_type!r}; overruled by content detector",
                        0.0,
                    )
                )
        return SemanticTypeResult(detected_type, min(confidence, 0.98), tuple(evidence))

    if embedding_vote is not None:
        vote_type, vote_score = embedding_vote
        if vote_type in SEMANTIC_TYPES and vote_score >= 0.5:
            evidence.append(
                SemanticEvidence(
                    "embedding",
                    f"model head proposed {vote_type!r} confidence={vote_score:.2f} "
                    "(no contradicting content detector)",
                    0.0,
                )
            )
            # Model-only labels are never certain — capped well below the
            # auto thresholds so they inform, not decide.
            return SemanticTypeResult(vote_type, min(0.7, vote_score), tuple(evidence))

    if name_type is not None:
        evidence.append(SemanticEvidence("pattern", f"column name suggests {name_type}", 0.0))
        return SemanticTypeResult(name_type, 0.5, tuple(evidence))

    if role == "categorical":
        return SemanticTypeResult(
            "category_code",
            0.4,
            (SemanticEvidence("column_role", "role inference: categorical", 0.0),),
        )
    return SemanticTypeResult("unknown", 0.2, tuple(evidence))
