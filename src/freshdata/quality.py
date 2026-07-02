"""Soft-warning quality-debt ledger — a middle ground between hard failure and
ignoring data-quality issues.

    cleaned, gate = fd.evaluate_quality_debt(
        df, debt_policy="warn_then_fail", ledger="quality_debt.sqlite")

Cleans *df*, scores quality debt across nine issue dimensions, **persists the
history** to a server-free SQLite ledger, and escalates from *warn* to *fail*
when an issue repeats or worsens across runs. Emits both human-readable and
machine-readable reports and integrates with the existing trust gate.

Light core: stdlib ``sqlite3`` + the in-process cleaner. PII scoring uses the
enterprise detector lazily and degrades gracefully if unavailable.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd

from .render import html as H
from .render.mixins import SimpleHtmlReport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .report import CleanReport

_POLICIES = ("warn", "fail", "warn_then_fail")

#: The nine debt dimensions and their default "in debt" thresholds (0..1 score).
DEBT_DIMENSIONS = (
    "missingness", "duplicates", "schema_drift", "type_instability",
    "outlier_spikes", "pii_risk", "category_churn", "failed_repairs",
    "human_review_backlog",
)
_DEFAULT_THRESHOLDS = dict.fromkeys(DEBT_DIMENSIONS, 0.1)
_DEFAULT_THRESHOLDS.update({"failed_repairs": 0.0, "human_review_backlog": 0.0,
                            "pii_risk": 0.0, "schema_drift": 0.0})


@dataclass
class DebtItem:
    dimension: str
    score: float
    threshold: float
    detail: str
    previous: float | None = None

    @property
    def over(self) -> bool:
        return self.score > self.threshold

    @property
    def worsening(self) -> bool:
        return self.previous is not None and self.score > self.previous

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": round(self.score, 4),
            "threshold": self.threshold,
            "over_threshold": self.over,
            "previous": None if self.previous is None else round(self.previous, 4),
            "worsening": self.worsening,
            "detail": self.detail,
        }


@dataclass
class QualityDebtGate(SimpleHtmlReport):
    """Outcome of :func:`evaluate_quality_debt`."""

    status: str  # "pass" | "warn" | "fail"
    policy: str
    items: list[DebtItem] = field(default_factory=list)
    total_score: float = 0.0
    run_at: str = ""
    ledger_path: str | None = None
    trust_score: float | None = None

    @property
    def passed(self) -> bool:
        return self.status != "fail"

    @property
    def warned(self) -> list[DebtItem]:
        return [i for i in self.items if i.over]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [i.to_dict() for i in self.items],
            columns=["dimension", "score", "threshold", "over_threshold",
                     "previous", "worsening", "detail"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "policy": self.policy,
            "passed": self.passed,
            "total_score": round(self.total_score, 4),
            "trust_score": self.trust_score,
            "run_at": self.run_at,
            "ledger_path": self.ledger_path,
            "items": [i.to_dict() for i in self.items],
        }

    def summary(self) -> str:
        lines = [
            f"freshdata quality-debt gate: {self.status.upper()} "
            f"(policy={self.policy}, total debt {self.total_score:.2f})",
        ]
        if self.trust_score is not None:
            lines.append(f"  trust score: {self.trust_score:.1f}")
        for i in sorted(self.items, key=lambda x: x.score, reverse=True):
            if not i.over:
                continue
            flag = " WORSENING" if i.worsening else ""
            lines.append(f"  ! {i.dimension}: {i.score:.2f} "
                         f"(threshold {i.threshold:.2f}){flag} — {i.detail}")
        if not self.warned:
            lines.append("  no quality debt over threshold")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    # -- HTML ----------------------------------------------------------------

    def _html_title(self) -> str:
        return "freshdata quality-debt gate"

    def _html_subtitle(self) -> str | None:
        return f"{self.status.upper()} · policy {self.policy} · total debt {self.total_score:.2f}"

    def _html_sections(self) -> list[str]:
        color = {"pass": "#1a7f37", "warn": "#9a6700", "fail": "#cf222e"}[self.status]
        cards = H.scorecards([
            ("status", self.status.upper()),
            ("total debt", f"{self.total_score:.2f}"),
            ("over threshold", len(self.warned)),
            *([("trust", f"{self.trust_score:.1f}")] if self.trust_score is not None else []),
        ])
        rows = []
        for i in sorted(self.items, key=lambda x: x.score, reverse=True):
            sev = "high" if (i.over and i.worsening) else "medium" if i.over else "low"
            rows.append([i.dimension, f"{i.score:.2f}", f"{i.threshold:.2f}",
                         H.risk_badge(sev),
                         "↑" if i.worsening else "", i.detail])
        tbl = H.filterable_table(
            "fd-debt", ["dimension", "score", "threshold", "severity", "trend", "detail"],
            rows, filters={"dimension": 0}, raw_columns=[3])
        dl = H.json_download("quality_debt.json", self.to_dict(), "⬇ JSON")
        banner = (f"<div style='padding:.4rem .6rem;border-radius:8px;color:#fff;"
                  f"background:{color};display:inline-block'>{self.status.upper()}</div>")
        return [banner, cards, tbl, dl]


def _score_debt(
    df: pd.DataFrame, cleaned: pd.DataFrame, report: CleanReport,
    baseline: pd.DataFrame | None,
) -> dict[str, tuple[float, str]]:
    """Return ``{dimension: (score, detail)}`` for the current run."""
    rows = max(1, report.rows_after)
    cells = max(1, report.rows_after * max(1, report.cols_after))
    out: dict[str, tuple[float, str]] = {}

    miss = report.missing_after / cells
    out["missingness"] = (miss, f"{report.missing_after:,} missing cell(s) remain")

    dup = report.duplicates_removed / max(1, report.rows_before)
    out["duplicates"] = (dup, f"{report.duplicates_removed:,} duplicate row(s) removed")

    outl = report.outliers_handled / rows
    out["outlier_spikes"] = (outl, f"{report.outliers_handled:,} outlier(s) flagged")

    failed = sum(1 for a in report.actions if a.risk == "high")
    out["failed_repairs"] = (failed / max(1, len(report.actions) or 1),
                             f"{failed} high-risk action(s)")

    review = len(report.recommendations) + sum(
        1 for a in report.actions if getattr(a, "human_review", False))
    out["human_review_backlog"] = (min(1.0, review / 10.0),
                                    f"{review} item(s) awaiting review")

    # Type instability: text columns the profiler still wants to retype.
    try:
        from .config import CleanConfig  # noqa: PLC0415
        from .profile import build_profile  # noqa: PLC0415

        prof = build_profile(cleaned, CleanConfig())
        retype = sum(1 for c in prof.columns
                     if c.suggested_dtype and c.suggested_dtype != c.dtype)
        out["type_instability"] = (retype / max(1, report.cols_after),
                                   f"{retype} column(s) with unstable types")
    except Exception:  # pragma: no cover - profiling is best-effort
        out["type_instability"] = (0.0, "not assessed")

    # PII risk (best-effort, lazy enterprise import).
    try:
        from .enterprise.privacy import detect_pii  # noqa: PLC0415

        scan = detect_pii(cleaned)
        n_pii = len(getattr(scan, "entities", []) or [])
        out["pii_risk"] = (min(1.0, n_pii / max(1, report.cols_after)),
                           f"{n_pii} potential PII column(s)")
    except Exception:
        out["pii_risk"] = (0.0, "PII scan unavailable")

    # Schema drift + category churn need a baseline.
    if baseline is not None:
        added = set(map(str, df.columns)) - set(map(str, baseline.columns))
        removed = set(map(str, baseline.columns)) - set(map(str, df.columns))
        denom = max(1, len(set(map(str, baseline.columns))))
        out["schema_drift"] = ((len(added) + len(removed)) / denom,
                               f"{len(added)} added, {len(removed)} removed column(s)")
        churn = _category_churn(baseline, df)
        out["category_churn"] = (churn, "category distribution churn vs baseline")
    else:
        out["schema_drift"] = (0.0, "no baseline supplied")
        out["category_churn"] = (0.0, "no baseline supplied")

    return out


def _category_churn(baseline: pd.DataFrame, current: pd.DataFrame) -> float:
    """Mean fraction of new categorical values across shared low-cardinality cols."""
    shared = [c for c in baseline.columns if c in current.columns]
    fracs = []
    for c in shared:
        try:
            bvals = set(baseline[c].dropna().astype(str).unique())
            cvals = set(current[c].dropna().astype(str).unique())
        except Exception:  # pragma: no cover - unhashable
            continue
        if 0 < len(bvals) <= 50:
            new = cvals - bvals
            fracs.append(len(new) / max(1, len(bvals)))
    return round(sum(fracs) / len(fracs), 4) if fracs else 0.0


def _ledger_setup(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS debt_runs "
                 "(run_id INTEGER PRIMARY KEY AUTOINCREMENT, run_at TEXT, "
                 "policy TEXT, status TEXT, total_score REAL)")
    conn.execute("CREATE TABLE IF NOT EXISTS debt_items "
                 "(run_id INTEGER, dimension TEXT, score REAL, over_threshold INTEGER)")


def _previous_scores(conn: sqlite3.Connection) -> dict[str, float]:
    row = conn.execute("SELECT run_id FROM debt_runs ORDER BY run_id DESC LIMIT 1").fetchone()
    if not row:
        return {}
    return {d: float(s) for d, s in conn.execute(
        "SELECT dimension, score FROM debt_items WHERE run_id=?", (row[0],)).fetchall()}


def _over_threshold_history(conn: sqlite3.Connection, dimension: str, n: int = 1) -> int:
    """How many of the last *n* runs had this dimension over threshold."""
    rows = conn.execute(
        "SELECT over_threshold FROM debt_items WHERE dimension=? "
        "ORDER BY run_id DESC LIMIT ?", (dimension, n)).fetchall()
    return sum(int(r[0]) for r in rows)


def evaluate_quality_debt(
    df: pd.DataFrame,
    *,
    debt_policy: str = "warn_then_fail",
    ledger: str | None = None,
    baseline: pd.DataFrame | None = None,
    thresholds: dict[str, float] | None = None,
    include_trust_score: bool = True,
    **clean_options: Any,
) -> tuple[pd.DataFrame, QualityDebtGate]:
    """Clean *df*, score quality debt, persist history, and gate per *debt_policy*.

    Parameters
    ----------
    df:
        Frame to clean and assess.
    debt_policy:
        ``"warn"`` (never fail), ``"fail"`` (fail on any debt over threshold), or
        ``"warn_then_fail"`` (default: warn first, fail when an issue repeats or
        worsens across runs).
    ledger:
        Path to a SQLite file for the persistent debt history (created if absent).
        ``None`` keeps the run in memory only (no escalation history).
    baseline:
        Optional prior frame enabling schema-drift and category-churn scoring.
    thresholds:
        Per-dimension overrides of the default "in debt" thresholds.
    **clean_options:
        Forwarded to :func:`freshdata.clean`.

    Returns
    -------
    (cleaned_df, QualityDebtGate)
    """
    if debt_policy not in _POLICIES:
        raise ValueError(f"debt_policy must be one of {_POLICIES}, got {debt_policy!r}")
    from .api import clean  # noqa: PLC0415

    cleaned, report = clean(df, report=True, **clean_options)
    thr = dict(_DEFAULT_THRESHOLDS)
    thr.update(thresholds or {})
    scores = _score_debt(df, cleaned, report, baseline)

    conn = sqlite3.connect(ledger) if ledger else None
    previous: dict[str, float] = {}
    if conn is not None:
        _ledger_setup(conn)
        previous = _previous_scores(conn)

    items: list[DebtItem] = []
    for dim in DEBT_DIMENSIONS:
        score, detail = scores.get(dim, (0.0, "not assessed"))
        items.append(DebtItem(dim, score, thr[dim], detail, previous.get(dim)))

    over = [i for i in items if i.over]
    # Decide status under the policy.
    if not over:
        status = "pass"
    elif debt_policy == "warn":
        status = "warn"
    elif debt_policy == "fail":
        status = "fail"
    else:  # warn_then_fail
        escalate = False
        for i in over:
            repeated = conn is not None and _over_threshold_history(conn, i.dimension, 1) >= 1
            if i.worsening or repeated:
                escalate = True
                break
        status = "fail" if escalate else "warn"

    total = round(sum(i.score for i in items), 4)
    run_at = datetime.now(timezone.utc).isoformat()

    trust = None
    if include_trust_score:
        try:
            from .enterprise.metrics import compute_trust_score  # noqa: PLC0415

            trust = float(compute_trust_score(cleaned).overall)
        except Exception:  # pragma: no cover - optional
            trust = None

    if conn is not None:
        cur = conn.execute(
            "INSERT INTO debt_runs(run_at, policy, status, total_score) VALUES(?,?,?,?)",
            (run_at, debt_policy, status, total))
        run_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO debt_items(run_id, dimension, score, over_threshold) VALUES(?,?,?,?)",
            [(run_id, i.dimension, i.score, int(i.over)) for i in items])
        conn.commit()
        conn.close()

    gate = QualityDebtGate(
        status=status, policy=debt_policy, items=items, total_score=total,
        run_at=run_at, ledger_path=ledger, trust_score=trust,
    )
    return cleaned, gate
