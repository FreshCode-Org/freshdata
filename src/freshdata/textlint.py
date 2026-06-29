"""Mixed-language and encoding linting for text columns.

``fd.lint_text_encoding(df, columns=..., locale_hints=...)`` flags text-quality
problems that silently corrupt joins, displays and analytics: mixed scripts,
mojibake-like artifacts, Unicode normalization inconsistencies, RTL/LTR rendering
risk, locale-ambiguous dates/numbers, replacement characters and stray
control/zero-width whitespace.

It is **diagnostic only** — nothing is modified. Each issue carries a severity,
example values, a suggested repair, and whether that repair is safe to automate.
Stdlib only (``unicodedata`` + ``re``); part of the light core.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ._util import stringlike_columns
from .render import html as H
from .render.mixins import SimpleHtmlReport

_SEVERITY = {
    "mixed_script": "high",
    "mojibake": "high",
    "replacement_char": "high",
    "control_chars": "medium",
    "nfc_nfd_inconsistency": "medium",
    "rtl_ltr_risk": "medium",
    "ambiguous_date": "medium",
    "ambiguous_number": "low",
    "irregular_whitespace": "low",
}

# Telltale UTF-8-decoded-as-Latin1/CP1252 sequences.
_MOJIBAKE_RE = re.compile(
    r"Ã[\x80-\xbf]|Â[\xa0-\xbf]|â€[\x99\x9c\x9d\x93\x94\xa6]|Ã©|Ã¨|Ã¼|Ã±|Ã |â€™|â€œ"
)
_DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b")
_IRREGULAR_WS = {"\u00a0", "\u202f", "\u2007", "\u200b", "\u200c", "\u200d", "\ufeff"}
_BIDI_MARKS = {"\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e"}


def _script_of(ch: str) -> str | None:
    """Coarse Unicode script family of a *letter*, or ``None`` for non-letters."""
    if not ch.isalpha():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    head = name.split(" ", 1)[0]
    # Collapse the CJK/Kana families that legitimately co-occur in Japanese.
    if head in ("CJK", "HIRAGANA", "KATAKANA"):
        return "CJK_JP"
    return head


@dataclass(frozen=True)
class TextIssue:
    """One detected text-quality problem for a column."""

    column: str
    issue_type: str
    severity: str
    count: int
    examples: list[str]
    suggested_repair: str
    auto_repair_safe: bool
    human_review: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "count": self.count,
            "examples": list(self.examples),
            "suggested_repair": self.suggested_repair,
            "auto_repair_safe": self.auto_repair_safe,
            "human_review": self.human_review,
        }


@dataclass
class TextLintReport(SimpleHtmlReport):
    """Result of :func:`lint_text_encoding`."""

    issues: list[TextIssue] = field(default_factory=list)
    columns_checked: list[str] = field(default_factory=list)
    values_scanned: int = 0
    locale_hints: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.issues)

    def __len__(self) -> int:
        return len(self.issues)

    @property
    def n_high(self) -> int:
        return sum(1 for i in self.issues if i.severity == "high")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [i.to_dict() for i in self.issues],
            columns=["column", "issue_type", "severity", "count", "examples",
                     "suggested_repair", "auto_repair_safe", "human_review"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns_checked": list(self.columns_checked),
            "values_scanned": self.values_scanned,
            "locale_hints": list(self.locale_hints),
            "n_issues": len(self.issues),
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        lines = [
            f"freshdata text/encoding lint — {len(self.issues)} issue(s) across "
            f"{len(self.columns_checked)} column(s), {self.values_scanned:,} values scanned",
        ]
        for i in sorted(self.issues, key=lambda x: (x.severity != "high", x.column)):
            flag = " [review]" if i.human_review else ""
            ex = ", ".join(repr(e) for e in i.examples[:2])
            lines.append(
                f"  [{i.severity}] {i.column}: {i.issue_type} ({i.count}){flag} — "
                f"e.g. {ex}; fix: {i.suggested_repair}"
            )
        if not self.issues:
            lines.append("  no text-quality issues detected")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    # -- HTML ----------------------------------------------------------------

    def _html_title(self) -> str:
        return "freshdata text / encoding lint"

    def _html_subtitle(self) -> str | None:
        return (f"{len(self.issues)} issue(s) · {self.values_scanned:,} values scanned"
                + (f" · locale hints: {', '.join(self.locale_hints)}"
                   if self.locale_hints else ""))

    def _html_sections(self) -> list[str]:
        if not self.issues:
            return ["<div class='fd-meta'>no text-quality issues detected</div>"]
        rows = []
        for i in self.issues:
            safe = "✓ safe" if i.auto_repair_safe else "✗ manual"
            review = H.badge("review", "#9a6700") if i.human_review else ""
            rows.append([
                str(i.column), str(i.issue_type), H.risk_badge(i.severity),
                str(i.count), "; ".join(i.examples[:3]), i.suggested_repair,
                f"{safe} {review}",
            ])
        tbl = H.filterable_table(
            "fd-textlint",
            ["column", "issue", "severity", "count", "examples", "suggested repair",
             "auto-repair"],
            rows, filters={"column": 0, "issue": 1}, raw_columns=[2, 6])
        dl = H.json_download("text_lint.json", self.to_dict(), "⬇ JSON")
        return [dl, tbl]


def _scan_column(name: str, values: list[str], locale_hints: list[str]) -> list[TextIssue]:
    counters: dict[str, int] = {}
    examples: dict[str, list[str]] = {}

    def hit(kind: str, value: str) -> None:
        counters[kind] = counters.get(kind, 0) + 1
        ex = examples.setdefault(kind, [])
        if len(ex) < 3 and value not in ex:
            ex.append(value)

    for v in values:
        scripts = {s for s in (_script_of(c) for c in v) if s}
        if len(scripts) > 1:
            hit("mixed_script", v)
        if _MOJIBAKE_RE.search(v):
            hit("mojibake", v)
        if "�" in v:
            hit("replacement_char", v)
        if any(unicodedata.category(c) == "Cc" and c not in "\t\n\r" for c in v) or (
                _BIDI_MARKS & set(v)):
            hit("control_chars", v)
        if _IRREGULAR_WS & set(v):
            hit("irregular_whitespace", v)
        if v != unicodedata.normalize("NFC", v):
            hit("nfc_nfd_inconsistency", v)
        bidi = {unicodedata.bidirectional(c) for c in v}
        if ({"R", "AL"} & bidi) and ({"L"} & bidi):
            hit("rtl_ltr_risk", v)
        m = _DATE_RE.search(v)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a != b and a <= 12 and b <= 12:
                hit("ambiguous_date", v)
        if "," in v and "." in v and any(c.isdigit() for c in v):
            hit("ambiguous_number", v)

    repairs = {
        "mixed_script": ("normalize the column to a single script / transliterate", False, True),
        "mojibake": ("re-decode as UTF-8 (likely mis-decoded latin-1/cp1252)", False, True),
        "replacement_char": ("re-ingest from source; � marks lost bytes", False, True),
        "control_chars": ("strip control / bidi-override characters", True, False),
        "irregular_whitespace": ("replace NBSP/zero-width with a normal space", True, False),
        "nfc_nfd_inconsistency": ("apply Unicode NFC normalization", True, False),
        "rtl_ltr_risk": ("add explicit bidi isolates or split mixed-direction text", False, True),
        "ambiguous_date": ("parse with an explicit dayfirst/locale format", False, True),
        "ambiguous_number": ("parse with an explicit thousands/decimal locale", False, True),
    }
    out = []
    for kind, count in counters.items():
        repair, safe, review = repairs[kind]
        out.append(TextIssue(
            column=name, issue_type=kind, severity=_SEVERITY[kind], count=count,
            examples=examples[kind], suggested_repair=repair,
            auto_repair_safe=safe, human_review=review,
        ))
    return out


def lint_text_encoding(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    locale_hints: list[str] | None = None,
    sample: int | None = 20_000,
) -> TextLintReport:
    """Lint text columns for mixed-language and encoding problems.

    Parameters
    ----------
    df:
        The frame to inspect (never modified).
    columns:
        Columns to check; defaults to all string-like columns.
    locale_hints:
        Optional locale tags (e.g. ``["en_IN", "ar_AE"]``) recorded on the report
        and used to frame ambiguity messages.
    sample:
        Max non-null values scanned per column (``None`` for all), keeping the
        pass cheap on large frames.

    Returns
    -------
    TextLintReport
    """
    hints = list(locale_hints or [])
    if columns is None:
        cols = list(stringlike_columns(df))
    else:
        cols = [c for c in columns if c in df.columns]

    issues: list[TextIssue] = []
    scanned = 0
    for col in cols:
        s = df[col].dropna()
        if sample is not None and len(s) > sample:
            s = s.head(sample)
        values = [str(x) for x in s.tolist()]
        scanned += len(values)
        issues.extend(_scan_column(str(col), values, hints))

    issues.sort(key=lambda i: (i.severity != "high", i.severity != "medium", i.column))
    return TextLintReport(
        issues=issues, columns_checked=[str(c) for c in cols],
        values_scanned=scanned, locale_hints=hints,
    )
