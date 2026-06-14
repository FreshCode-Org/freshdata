"""Data Trust Score and the automated quality / execution report.

The trust score blends four measurable dimensions into a single 0-100 number:

- **completeness** — share of cells that are present (not missing),
- **validity** — share of *present* cells already in a clean, parseable form
  (no stray whitespace, no sentinel-as-value, parses to its inferred dtype, not
  a statistical outlier),
- **uniqueness** — share of rows that are not exact duplicates,
- **consistency** — share of columns free of structural defects (mixed types,
  constant columns, duplicate labels).

Validity reuses the *exact* inference helpers the cleaning pipeline uses
(:func:`normalize_text`, :func:`suggest_conversion`, :func:`_bounds`), so the
score reflects what :func:`freshdata.clean` would actually find and repair.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd
from pandas.api.types import infer_dtype, is_bool_dtype, is_numeric_dtype

from .._util import _is_stringlike_dtype
from ..adapters.polars import to_pandas
from ..config import CleanConfig
from ..steps.dtypes import suggest_conversion
from ..steps.outliers import _bounds
from ..steps.strings import active_sentinels, normalize_text
from .config import TrustScoreWeights

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..report import CleanReport

#: Minimum non-null count before outliers are scored (matches profile.py).
_MIN_FOR_OUTLIERS = 20


@dataclass(frozen=True)
class ColumnTrust:
    """Per-column completeness and validity, with the issues that lowered it."""

    name: str
    completeness: float
    validity: float
    missing_pct: float
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class TrustScore:
    """A 0-100 data trust score with its four component dimensions.

    Render with ``print(score)``, export with :meth:`to_dict` /
    :meth:`to_markdown`. ``grade`` maps the overall score to an A-F letter.
    """

    overall: float
    completeness: float
    validity: float
    uniqueness: float
    consistency: float
    n_rows: int
    n_cols: int
    columns: tuple[ColumnTrust, ...] = ()

    @property
    def grade(self) -> str:
        """Letter grade: A (>=90), B (>=80), C (>=70), D (>=60), else F."""
        for cutoff, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):
            if self.overall >= cutoff:
                return letter
        return "F"

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 2),
            "grade": self.grade,
            "dimensions": {
                "completeness": round(self.completeness, 2),
                "validity": round(self.validity, 2),
                "uniqueness": round(self.uniqueness, 2),
                "consistency": round(self.consistency, 2),
            },
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [
                {
                    "name": c.name,
                    "completeness": round(c.completeness, 2),
                    "validity": round(c.validity, 2),
                    "missing_pct": round(c.missing_pct, 2),
                    "issues": list(c.issues),
                }
                for c in self.columns
            ],
        }

    def to_markdown(self) -> str:
        """Markdown table of the score and its dimensions."""
        rows = [
            ("Completeness", self.completeness),
            ("Validity", self.validity),
            ("Uniqueness", self.uniqueness),
            ("Consistency", self.consistency),
        ]
        body = [_md_table_row((dim, f"{val:.1f}")) for dim, val in rows]
        lines = [
            f"### Data Trust Score: **{self.overall:.1f} / 100**  (grade {self.grade})",
            "",
            _md_table_row(("Dimension", "Score")),
            _md_table_row(("---", "---:")),
            *body,
            _md_table_row(("**Overall**", f"**{self.overall:.1f}**")),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return (
            f"Data Trust Score {self.overall:.1f}/100 (grade {self.grade}) — "
            f"completeness {self.completeness:.0f}, validity {self.validity:.0f}, "
            f"uniqueness {self.uniqueness:.0f}, consistency {self.consistency:.0f}"
        )

    def __repr__(self) -> str:
        return f"<TrustScore {self.overall:.1f}/100 grade={self.grade}>"


def _md_table_row(cells: tuple[str, ...]) -> str:
    return "| " + " | ".join(cells) + " |"


def _column_validity(
    s: pd.Series, config: CleanConfig, sentinels: frozenset
) -> tuple[int, list[str]]:
    """Count representation-invalid cells in one column; return (count, issues).

    "Invalid" = stray surrounding whitespace, sentinel-as-value, values that
    will not parse to the column's inferred dtype, or statistical outliers.
    Mirrors the detections surfaced by :func:`freshdata.profile`.
    """
    invalid = 0
    issues: list[str] = []
    non_null = int(s.notna().sum())
    if non_null == 0:
        return 0, issues

    if _is_stringlike_dtype(s.dtype):
        normalized, n_stripped, n_sentinels = normalize_text(s, config, sentinels)
        if n_stripped:
            invalid += n_stripped
            issues.append(f"{n_stripped} whitespace")
        if n_sentinels:
            invalid += n_sentinels
            issues.append(f"{n_sentinels} sentinel")
        _, converted, n_coerced = suggest_conversion(normalized, config)
        if converted is not None and n_coerced:
            invalid += n_coerced
            issues.append(f"{n_coerced} unparseable")
    elif is_numeric_dtype(s) and not is_bool_dtype(s) and non_null >= _MIN_FOR_OUTLIERS:
        bounds = _bounds(s, config)
        if bounds is not None:
            n_out = int(((s < bounds[0]) | (s > bounds[1])).sum())
            if n_out:
                invalid += n_out
                issues.append(f"{n_out} outlier")

    if infer_dtype(s, skipna=True) in ("mixed", "mixed-integer"):
        issues.append("mixed types")
    return min(invalid, non_null), issues


def _is_structurally_inconsistent(s: pd.Series, n_rows: int) -> bool:
    kind = infer_dtype(s, skipna=True)
    if kind in ("mixed", "mixed-integer"):
        return True
    try:
        if n_rows > 1 and int(s.nunique(dropna=True)) <= 1:
            return True
    except TypeError:
        return False
    return False


def compute_trust_score(
    df: pd.DataFrame,
    *,
    weights: TrustScoreWeights | None = None,
    config: CleanConfig | None = None,
) -> TrustScore:
    """Profile *df* (pandas or polars) and compute its Data Trust Score.

    Read-only: the input is never modified.
    """
    frame = to_pandas(df)
    cfg = config or CleanConfig()
    w = (weights or TrustScoreWeights()).normalized()
    sentinels = active_sentinels(cfg)

    n_rows = len(frame)
    n_cols = frame.shape[1]
    n_cells = int(frame.size)

    if n_cells == 0:
        return TrustScore(100.0, 100.0, 100.0, 100.0, 100.0, n_rows, n_cols, ())

    missing_cells = int(frame.isna().sum().sum())
    completeness = 100.0 * (1.0 - missing_cells / n_cells)

    total_non_null = 0
    total_invalid = 0
    inconsistent_cols = 0
    col_trust: list[ColumnTrust] = []
    for i in range(n_cols):
        s = frame.iloc[:, i]
        non_null = int(s.notna().sum())
        invalid, issues = _column_validity(s, cfg, sentinels)
        total_non_null += non_null
        total_invalid += invalid
        if _is_structurally_inconsistent(s, n_rows):
            inconsistent_cols += 1
        col_missing_pct = 100.0 * (1.0 - non_null / n_rows) if n_rows else 0.0
        col_validity = 100.0 * (1.0 - invalid / non_null) if non_null else 100.0
        col_trust.append(
            ColumnTrust(
                name=str(frame.columns[i]),
                completeness=100.0 - col_missing_pct,
                validity=col_validity,
                missing_pct=col_missing_pct,
                issues=tuple(issues),
            )
        )

    validity = 100.0 * (1.0 - total_invalid / total_non_null) if total_non_null else 100.0

    try:
        dup_rows = int(frame.duplicated().sum())
        uniqueness = 100.0 * (1.0 - dup_rows / n_rows) if n_rows else 100.0
    except TypeError:  # pragma: no cover - unhashable cells (rare; mirrors profile.py)
        uniqueness = 100.0

    dup_labels = int(frame.columns.duplicated().sum())
    flagged = min(n_cols, inconsistent_cols + dup_labels)
    consistency = 100.0 * (1.0 - flagged / n_cols) if n_cols else 100.0

    overall = (
        w["completeness"] * completeness
        + w["validity"] * validity
        + w["uniqueness"] * uniqueness
        + w["consistency"] * consistency
    )
    return TrustScore(
        overall=overall,
        completeness=completeness,
        validity=validity,
        uniqueness=uniqueness,
        consistency=consistency,
        n_rows=n_rows,
        n_cols=n_cols,
        columns=tuple(col_trust),
    )


@dataclass
class QualityReport:
    """Before/after quality summary for one cleaning run.

    Pairs the trust score before and after cleaning with the structured
    :class:`~freshdata.CleanReport`, and exposes the whole thing as JSON (for
    logging / orchestration) and as a console-friendly markdown report.
    """

    trust_before: TrustScore
    trust_after: TrustScore
    clean_report: CleanReport
    actor: str = "unknown"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def percent_clean(self) -> float:
        """Headline 'how clean is it now' metric — the post-clean trust score."""
        return round(self.trust_after.overall, 2)

    @property
    def trust_delta(self) -> float:
        """Change in overall trust score from cleaning (post minus pre)."""
        return round(self.trust_after.overall - self.trust_before.overall, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "actor": self.actor,
            "percent_clean": self.percent_clean,
            "trust_delta": self.trust_delta,
            "trust_before": self.trust_before.to_dict(),
            "trust_after": self.trust_after.to_dict(),
            "clean_report": self.clean_report.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        rep = self.clean_report
        before, after = self.trust_before, self.trust_after
        lines = [
            "# freshdata Quality Report",
            "",
            f"- **Generated:** {self.generated_at}",
            f"- **Actor:** {self.actor}",
            f"- **Percent clean (trust after):** {self.percent_clean:.1f}% "
            f"(grade {after.grade})",
            f"- **Trust delta:** {self.trust_delta:+.1f}",
            "",
            "## Trust dimensions (before → after)",
            "",
            _md_table_row(("Dimension", "Before", "After")),
            _md_table_row(("---", "---:", "---:")),
            _md_table_row(("Completeness", f"{before.completeness:.1f}",
                           f"{after.completeness:.1f}")),
            _md_table_row(("Validity", f"{before.validity:.1f}", f"{after.validity:.1f}")),
            _md_table_row(("Uniqueness", f"{before.uniqueness:.1f}",
                           f"{after.uniqueness:.1f}")),
            _md_table_row(("Consistency", f"{before.consistency:.1f}",
                           f"{after.consistency:.1f}")),
            _md_table_row(("**Overall**", f"**{before.overall:.1f}**",
                           f"**{after.overall:.1f}**")),
            "",
            "## What changed",
            "",
            f"- rows: {rep.rows_before:,} → {rep.rows_after:,}",
            f"- columns: {rep.cols_before:,} → {rep.cols_after:,}",
            f"- missing cells: {rep.missing_before:,} → {rep.missing_after:,}",
            f"- cells altered: {rep.cells_changed:,}",
            f"- duplicates removed: {rep.duplicates_removed:,}",
            f"- outliers handled: {rep.outliers_handled:,}",
        ]
        if rep.actions:
            lines += ["", "## Actions", "",
                      _md_table_row(("Step", "Column", "Description", "Count")),
                      _md_table_row(("---", "---", "---", "---:"))]
            lines += [
                _md_table_row((a.step, a.column or "—",
                               a.description.replace("|", "\\|"), f"{a.count:,}"))
                for a in rep.actions
            ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


def build_quality_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
    clean_report: CleanReport,
    *,
    weights: TrustScoreWeights | None = None,
    config: CleanConfig | None = None,
    actor: str = "unknown",
) -> QualityReport:
    """Build a :class:`QualityReport` from the input/output frames and report."""
    return QualityReport(
        trust_before=compute_trust_score(before, weights=weights, config=config),
        trust_after=compute_trust_score(after, weights=weights, config=config),
        clean_report=clean_report,
        actor=actor,
    )
