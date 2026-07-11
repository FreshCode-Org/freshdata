"""Configurable, field-aware text cleaning with a full audit trail.

``fd.clean_text(df, ...)`` (and the scalar ``fd.clean_text_value``) normalize
messy text — Unicode forms, whitespace, control characters, HTML, URLs,
punctuation, casing — without ever destroying information:

* the input frame is **never modified**; a cleaned copy is returned;
* every changed cell is logged with its original value, cleaned value and the
  exact ordered list of transforms that fired;
* cleaning is **field-aware**: aggressive operations (punctuation stripping,
  case folding) are automatically withheld from field types they would corrupt
  (numbers, identifiers, emails, URLs, dates, tickers).

The pipeline is deterministic, stdlib-only (``unicodedata``, ``html``, ``re``)
and independent of the main cleaning engine, so it can run before domain
validation or entirely on its own.
"""

from __future__ import annotations

import html as _html
import re
import unicodedata
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from html.parser import HTMLParser
from typing import Any, Literal

import pandas as pd

__all__ = [
    "TextCleanConfig",
    "CleanedText",
    "TextCleanReport",
    "clean_text",
    "clean_text_value",
    "config_for_field",
]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_ZERO_WIDTH = "\u200b\u200c\u200d\u2060\ufeff"
_BIDI_MARKS = "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
_IRREGULAR_WS_RE = re.compile("[\u00a0\u202f\u2000-\u200a\u2007\u3000]")
_WS_RE = re.compile(r"\s+")
# Unicode punctuation → ASCII equivalents (smart quotes, dashes, ellipsis).
_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...",
})


class _TextExtractor(HTMLParser):
    """Collect text content, dropping tags, scripts and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self._parts.append(data)

    def text(self) -> str:
        return " ".join(p for p in self._parts if p)


def _strip_html(value: str) -> str:
    if "<" not in value or ">" not in value:
        return _html.unescape(value)
    parser = _TextExtractor()
    try:
        parser.feed(value)
        parser.close()
    except Exception:  # pragma: no cover - HTMLParser is extremely tolerant
        return value
    return parser.text()


@dataclass(frozen=True)
class TextCleanConfig:
    """Which cleaning operations run, in a fixed safe order.

    Every flag defaults to the *safe* choice: lossless normalizations are on,
    lossy operations (HTML/URL removal, case folding, punctuation removal)
    are opt-in.
    """

    unicode_form: Literal["NFC", "NFKC", "NFD", "NFKD"] | None = "NFC"
    strip_control_chars: bool = True
    strip_zero_width: bool = True
    normalize_punctuation: bool = True
    collapse_whitespace: bool = True
    strip: bool = True
    strip_html: bool = False
    strip_urls: bool = False
    case: str | None = None  #: ``None`` | ``"lower"`` | ``"upper"`` | ``"title"``
    remove_punctuation: bool = False
    max_char_repeat: int | None = None  #: cap runs of the same character
    max_length: int | None = None  #: truncate (recorded as a transform)
    custom: tuple[tuple[str, Callable[[str], str]], ...] = ()

    def __post_init__(self) -> None:
        if self.unicode_form not in (None, "NFC", "NFKC", "NFD", "NFKD"):
            raise ValueError(f"unsupported unicode_form: {self.unicode_form!r}")
        if self.case not in (None, "lower", "upper", "title"):
            raise ValueError(f"unsupported case: {self.case!r}")


#: Field types whose values are structural — punctuation, casing and length
#: are meaningful, so lossy operations are withheld even if configured.
_STRUCTURAL_TYPES = frozenset({
    "numeric", "integer", "float", "currency_amount", "rate", "percentage",
    "identifier", "account_number", "national_id", "postal_code",
    "email", "url", "phone", "date_like", "date", "datetime",
    "stock_ticker", "ticker", "category_code", "boolean_like",
})
_ENTITY_TYPES = frozenset({
    "person_name", "company_name", "entity_name", "city", "country", "address",
})


def config_for_field(
    semantic_type: str | None,
    base: TextCleanConfig | None = None,
) -> TextCleanConfig:
    """Return ``base`` restricted to operations safe for ``semantic_type``.

    Structural types (numbers, identifiers, emails, dates, tickers…) keep only
    lossless normalizations; entity names additionally never get punctuation
    stripped or case-folded to lower/upper. Free text passes ``base`` through.
    """
    cfg = base or TextCleanConfig()
    if semantic_type in _STRUCTURAL_TYPES:
        return replace(
            cfg, strip_html=False, strip_urls=False, case=None,
            remove_punctuation=False, max_char_repeat=None, max_length=cfg.max_length,
        )
    if semantic_type in _ENTITY_TYPES:
        case = cfg.case if cfg.case == "title" else None
        return replace(cfg, remove_punctuation=False, case=case)
    return cfg


@dataclass(frozen=True)
class CleanedText:
    """One cleaned value: the original, the result, and what happened."""

    original: Any
    cleaned: Any
    transforms: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.transforms)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "cleaned": self.cleaned,
            "transforms": list(self.transforms),
        }


def clean_text_value(
    value: Any,
    config: TextCleanConfig | None = None,
    *,
    field_type: str | None = None,
) -> CleanedText:
    """Clean a single value; non-strings and nulls pass through untouched."""
    if not isinstance(value, str):
        return CleanedText(value, value)
    cfg = config_for_field(field_type, config) if field_type else (config or TextCleanConfig())

    out = value
    transforms: list[str] = []

    def step(name: str, new: str) -> None:
        nonlocal out
        if new != out:
            transforms.append(name)
            out = new

    if cfg.unicode_form:
        step(f"unicode_{cfg.unicode_form.lower()}", unicodedata.normalize(cfg.unicode_form, out))
    if cfg.strip_html:
        step("strip_html", _strip_html(out))
    if cfg.strip_urls:
        step("strip_urls", _URL_RE.sub(" ", out))
    if cfg.strip_control_chars:
        cleaned = "".join(
            c for c in out
            if not (unicodedata.category(c) == "Cc" and c not in "\t\n\r")
            and c not in _BIDI_MARKS
        )
        step("strip_control_chars", cleaned)
    if cfg.strip_zero_width:
        step("strip_zero_width", "".join(c for c in out if c not in _ZERO_WIDTH))
    if cfg.normalize_punctuation:
        step("normalize_punctuation", _IRREGULAR_WS_RE.sub(" ", out.translate(_PUNCT_MAP)))
    if cfg.max_char_repeat is not None and cfg.max_char_repeat >= 1:
        n = cfg.max_char_repeat
        pattern = r"(.)\1{" + str(n) + ",}"
        step("collapse_repeats", re.sub(pattern, lambda m: m.group(1) * n, out))
    if cfg.remove_punctuation:
        step("remove_punctuation", "".join(
            c for c in out if not unicodedata.category(c).startswith("P")))
    if cfg.case:
        step(f"case_{cfg.case}", getattr(out, cfg.case)())
    for name, fn in cfg.custom:
        step(f"custom_{name}", str(fn(out)))
    if cfg.collapse_whitespace:
        step("collapse_whitespace", _WS_RE.sub(" ", out))
    if cfg.strip:
        step("strip", out.strip())
    if cfg.max_length is not None and len(out) > cfg.max_length:
        step("truncate", out[: cfg.max_length])

    return CleanedText(value, out, tuple(transforms))


@dataclass
class TextCleanReport:
    """Every change made by :func:`clean_text`, cell by cell."""

    changes: list = field(default_factory=list)  #: list of per-cell change dicts
    columns_cleaned: list = field(default_factory=list)
    values_seen: int = 0

    def __len__(self) -> int:
        return len(self.changes)

    def __bool__(self) -> bool:
        return bool(self.changes)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.changes, columns=["row", "column", "original", "cleaned", "transforms"]
        )

    def transform_counts(self) -> dict:
        counts: dict = {}
        for ch in self.changes:
            for t in ch["transforms"]:
                counts[t] = counts.get(t, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            f"freshdata text clean — {len(self.changes)} value(s) changed across "
            f"{len(self.columns_cleaned)} column(s), {self.values_seen:,} values seen"
        ]
        for t, n in sorted(self.transform_counts().items(), key=lambda kv: -kv[1]):
            lines.append(f"  {t}: {n}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def clean_text(
    df: pd.DataFrame,
    *,
    columns: list | None = None,
    config: TextCleanConfig | None = None,
    field_types: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, TextCleanReport]:
    """Clean the string cells of ``df``; return ``(cleaned_copy, report)``.

    Parameters
    ----------
    df:
        Input frame — never modified.
    columns:
        Columns to clean; defaults to all object/string columns.
    config:
        Base :class:`TextCleanConfig` (safe lossless defaults).
    field_types:
        Optional ``{column: semantic_type}`` map; each column's config is
        restricted via :func:`config_for_field` so e.g. punctuation stripping
        never runs on an amount or identifier column.
    """
    if columns is None:
        cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "string"]
    else:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"columns not in frame: {missing}")
        cols = list(columns)

    out = df.copy()
    report = TextCleanReport(columns_cleaned=[str(c) for c in cols])
    types = dict(field_types or {})

    for col in cols:
        cfg = config_for_field(types[col], config) if col in types else (
            config or TextCleanConfig())
        series = df[col]
        report.values_seen += int(series.notna().sum())
        # ponytail: per-cell python loop; vectorize per-op if profiling demands
        cleaned_values = {}
        for idx, val in series.items():
            if not isinstance(val, str):
                continue
            result = clean_text_value(val, cfg)
            if result.changed:
                cleaned_values[idx] = result.cleaned
                report.changes.append({
                    "row": idx, "column": str(col),
                    "original": val, "cleaned": result.cleaned,
                    "transforms": list(result.transforms),
                })
        if cleaned_values:
            out[col] = series.copy()
            out.loc[list(cleaned_values), col] = pd.Series(cleaned_values)
    return out, report
