"""Stakeholder-safe, business-language summaries of a cleaning run.

``fd.stakeholder_summary(report, audience="business", format="markdown")`` turns
a :class:`~freshdata.CleanReport` into plain-English narrative for non-technical
readers: what changed, why, what was preserved, and what still needs review —
with completeness and duplicate deltas and the columns worth a human look.

Light core: depends only on the already-computed report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .render import html as H
from .render.mixins import SimpleHtmlReport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .report import CleanReport

_AUDIENCES = ("business", "technical")
_FORMATS = ("markdown", "html")


def _pct(part: int, whole: int) -> float:
    return 100.0 * (1 - part / whole) if whole else 100.0


@dataclass
class StakeholderSummary(SimpleHtmlReport):
    """Business-language summary; export as Markdown or HTML."""

    headline: str
    audience: str
    format: str
    what_changed: list[str] = field(default_factory=list)
    what_preserved: list[str] = field(default_factory=list)
    needs_review: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    # -- exports -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "audience": self.audience,
            "metrics": dict(self.metrics),
            "what_changed": list(self.what_changed),
            "what_preserved": list(self.what_preserved),
            "needs_review": list(self.needs_review),
        }

    def to_markdown(self) -> str:
        lines = ["# Data quality summary", "", self.headline, ""]
        if self.what_changed:
            lines += ["## What changed", ""]
            lines += [f"- {x}" for x in self.what_changed] + [""]
        if self.what_preserved:
            lines += ["## What was preserved", ""]
            lines += [f"- {x}" for x in self.what_preserved] + [""]
        if self.needs_review:
            lines += ["## What needs review", ""]
            lines += [f"- {x}" for x in self.needs_review] + [""]
        return "\n".join(lines).rstrip() + "\n"

    def summary(self) -> str:
        """Plain-text version (same content as Markdown, without ``#`` syntax)."""
        out = [self.headline]
        for title, items in (("What changed", self.what_changed),
                             ("What was preserved", self.what_preserved),
                             ("What needs review", self.needs_review)):
            if items:
                out.append(f"\n{title}:")
                out.extend(f"  - {x}" for x in items)
        return "\n".join(out)

    def render(self) -> str:
        """Render in the configured ``format`` (markdown or html)."""
        return self.to_html() if self.format == "html" else self.to_markdown()

    def __str__(self) -> str:
        return self.summary()

    # -- HTML ----------------------------------------------------------------

    def _html_title(self) -> str:
        return "Data quality summary"

    def _html_subtitle(self) -> str | None:
        return self.headline

    def _html_sections(self) -> list[str]:
        cards = H.scorecards([(k, v) for k, v in self.metrics.items()])
        out = [cards] if self.metrics else []
        for title, items in (("What changed", self.what_changed),
                             ("What was preserved", self.what_preserved),
                             ("What needs review", self.needs_review)):
            if items:
                body = "<ul>" + "".join(f"<li>{H.esc(x)}</li>" for x in items) + "</ul>"
                out.append(H.section(title, body))
        dl = (H.json_download("stakeholder_summary.json", self.to_dict(), "⬇ JSON")
              + H.data_uri_download("stakeholder_summary.md", self.to_markdown(),
                                    "text/markdown", "⬇ Markdown"))
        out.append(dl)
        return out


def stakeholder_summary(
    report: CleanReport,
    *,
    audience: str = "business",
    format: str = "markdown",
) -> StakeholderSummary:
    """Build a business-language summary from a :class:`CleanReport`.

    Parameters
    ----------
    report:
        A report from ``fd.clean(df, return_report=True)``.
    audience:
        ``"business"`` (default, no jargon) or ``"technical"`` (adds step detail).
    format:
        Default export format for :meth:`StakeholderSummary.render` —
        ``"markdown"`` or ``"html"``. Both exports are always available.
    """
    if audience not in _AUDIENCES:
        raise ValueError(f"audience must be one of {_AUDIENCES}, got {audience!r}")
    if format not in _FORMATS:
        raise ValueError(f"format must be one of {_FORMATS}, got {format!r}")

    cells_before = report.rows_before * report.cols_before
    cells_after = report.rows_after * report.cols_after
    comp_before = _pct(report.missing_before, cells_before)
    comp_after = _pct(report.missing_after, cells_after)

    changed: list[str] = []
    if report.missing_before != report.missing_after:
        direction = "rose" if comp_after > comp_before else "fell"
        changed.append(
            f"Overall data completeness {direction} from {comp_before:.1f}% to "
            f"{comp_after:.1f}%.")
    if report.duplicates_removed:
        changed.append(f"{report.duplicates_removed:,} duplicate record(s) were removed.")
    if report.columns_imputed:
        changed.append(
            f"Missing values were filled in {len(report.columns_imputed)} column(s): "
            f"{', '.join(report.columns_imputed[:6])}.")
    if report.columns_dropped:
        changed.append(
            f"{len(report.columns_dropped)} unusable column(s) were removed: "
            f"{', '.join(report.columns_dropped[:6])}.")
    n_changed_cols = len({a.column for a in report.actions if a.column})
    if n_changed_cols:
        changed.append(f"{n_changed_cols} column(s) changed meaningfully.")
    if audience == "technical":
        steps = sorted({a.step for a in report.actions if a.count})
        if steps:
            changed.append("Steps applied: " + ", ".join(steps) + ".")

    preserved: list[str] = []
    if report.outliers_handled:
        preserved.append(
            f"{report.outliers_handled:,} unusual value(s) were flagged but kept for review "
            "rather than silently altered.")
    if report.columns_preserved:
        preserved.append(
            f"{len(report.columns_preserved)} column(s) were deliberately left untouched: "
            f"{', '.join(report.columns_preserved[:6])}.")
    if not preserved:
        preserved.append("Identifiers and key fields were left unchanged.")

    review: list[str] = list(report.warnings) + list(report.recommendations)

    headline = (
        f"Cleaning kept {comp_after:.1f}% of fields complete across "
        f"{report.rows_after:,} record(s); {len(review)} item(s) need review.")

    metrics = {
        "records": f"{report.rows_after:,}",
        "completeness": f"{comp_after:.1f}%",
        "duplicates removed": f"{report.duplicates_removed:,}",
        "needs review": len(review),
    }

    return StakeholderSummary(
        headline=headline, audience=audience, format=format,
        what_changed=changed, what_preserved=preserved, needs_review=review,
        metrics=metrics,
    )
