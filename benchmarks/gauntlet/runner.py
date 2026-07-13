"""Drive FreshData's validation surfaces over one gauntlet fixture.

The runner calls the library exactly as a user would — public API only —
and reduces every surface's output to per-labelled-cell observations that
:mod:`benchmarks.gauntlet.metrics` scores against the gold dispositions.

Surfaces exercised:

1. ``fd.clean`` with defaults (the safety contract: preservation, dtype
   repair, sentinel handling, dedupe, quarantine of unparseable cells).
2. ``fd.validate_fields`` with the fixture schema (per-cell detection).
3. ``fd.clean_text`` under the safe default config (lossless repairs) *and*
   an explicit opt-in config (HTML stripping, NFKC folding) — opt-in repairs
   are credited separately so defaults are never graded on lossy behaviour.
4. The semantic layer in ``auto`` mode (high-confidence applied repairs).
5. ``fd.lint_text_encoding`` (mojibake / mixed-script / control detection).
6. ``detect_pii`` (labelled PII cells).
7. The domain pack, when the fixture declares one.
"""

from __future__ import annotations

import numbers
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

import freshdata as fd
from freshdata.enterprise.privacy import detect_pii
from freshdata.textclean import TextCleanConfig

from .fixtures import DEFAULT_ROWS, DEFAULT_SEED, FIXTURES, GauntletFixture, GoldCell

#: Opt-in text config used for the second clean_text pass: lossy-but-audited
#: operations a caller must ask for. Punctuation/case stay untouched.
AGGRESSIVE_TEXT = TextCleanConfig(unicode_form="NFKC", strip_html=True)


@dataclass
class CellObservation:
    """Everything the surfaces said about one labelled cell."""

    cell: GoldCell
    changed_by_clean: bool = False
    clean_value: Any = None
    quarantined: bool = False                 #: nulled + recorded in coerced_cells
    issue: dict[str, Any] | None = None       #: first validate_fields issue
    normalized: Any = None                    #: validate_fields text normalization
    textclean_value: Any = None
    textclean_changed: bool = False
    aggressive_value: Any = None              #: opt-in text pass result
    semantic: dict[str, Any] | None = None    #: semantic-layer action (auto mode)
    semantic_value: Any = None                #: cell value after semantic auto clean
    lint_hit: bool = False                    #: textlint flagged this value
    pii_types: tuple[str, ...] = ()
    audit_covered: bool = False               #: mutation has an audit record


@dataclass
class FixtureRun:
    fixture: GauntletFixture
    observations: list[CellObservation]
    false_positive_cells: list[dict[str, Any]]  #: error issues on unlabelled cells
    n_clean_cells_checked: int
    duplicates_removed: int
    deterministic: bool
    trust_pristine: float
    trust_dirty: float
    clean_seconds: float
    validate_seconds: float
    peak_memory_mb: float
    audit_mutations: int = 0
    audit_recorded: int = 0
    domain_findings: int = 0
    warnings: list[str] = field(default_factory=list)


def _canon(value: Any) -> Any:
    """Loose canonical form so 45 == 45.0 == '45' == Timestamp('45')…"""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        f = float(value)
        return int(f) if f == int(f) else f
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return _canon(float(s))
        except ValueError:
            ts = pd.to_datetime(s, errors="coerce") if _dateish(s) else None
            return ts.isoformat() if ts is not None and not pd.isna(ts) else s
    return value


def _dateish(s: str) -> bool:
    return len(s) >= 8 and s[:4].isdigit() and s.count("-") >= 2


def _values_equal(a: Any, b: Any) -> bool:
    return _canon(a) == _canon(b)


def _timed_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, Any, float, float]:
    tracemalloc.start()
    t0 = time.perf_counter()
    out, report = fd.clean(df, return_report=True)
    seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, report, seconds, peak / 1e6


def _observe_clean(obs: dict, cleaned: pd.DataFrame, report: Any) -> tuple[int, int]:
    """Fold the default fd.clean results into the observations."""
    audit_columns = {a.column for a in report.actions if a.column} | {
        w.split("'")[1] for w in report.warnings if "'" in w
    }
    mutations = recorded = 0
    for (row, col), o in obs.items():
        if col not in cleaned.columns or row not in cleaned.index:
            o.changed_by_clean = True  # cell no longer addressable (dropped)
            continue
        o.clean_value = cleaned.at[row, col]
        o.changed_by_clean = not _values_equal(o.clean_value, o.cell.dirty)
        o.quarantined = (pd.isna(o.clean_value)
                         and row in report.coerced_cells.get(col, {}))
        if o.changed_by_clean:
            mutations += 1
            if col in audit_columns:
                recorded += 1
                o.audit_covered = True
    return mutations, recorded


def _observe_validate(obs: dict, vf: Any) -> list[dict[str, Any]]:
    fp_cells: list[dict[str, Any]] = []
    for issue in vf.issues:
        entry = {
            "row": issue.row, "column": issue.column,
            "classification": issue.classification, "severity": issue.severity,
            "action": issue.action, "reason": issue.reason,
            "confidence": issue.confidence, "suggestion": issue.suggestion,
        }
        key = (issue.row, issue.column)
        if key in obs:
            if obs[key].issue is None:
                obs[key].issue = entry
        elif issue.severity == "error":
            fp_cells.append(entry)
    for norm in vf.normalized_cells:
        key = (norm["row"], norm["column"])
        if key in obs:
            obs[key].normalized = norm["cleaned"]
    return fp_cells


def _observe_text(obs: dict, fx: GauntletFixture, df: pd.DataFrame) -> None:
    text_cols = [c for c in fx.field_types if c in df.columns]
    if not text_cols:
        return
    safe, _ = fd.clean_text(df, columns=text_cols, field_types=fx.field_types)
    hard, _ = fd.clean_text(df, columns=text_cols, field_types=fx.field_types,
                            config=AGGRESSIVE_TEXT)
    for (row, col), o in obs.items():
        if col in text_cols:
            o.textclean_value = safe.at[row, col]
            o.textclean_changed = not _values_equal(o.textclean_value, o.cell.dirty)
            o.aggressive_value = hard.at[row, col]

    lint = fd.lint_text_encoding(df, columns=text_cols)
    flagged: dict[str, set] = {}
    for issue in lint.issues:
        flagged.setdefault(issue.column, set()).update(issue.examples)
    for (_row, col), o in obs.items():
        examples = flagged.get(col, ())
        if isinstance(o.cell.dirty, str) and any(
                str(o.cell.dirty) in ex or ex in str(o.cell.dirty)
                for ex in examples):
            o.lint_hit = True


def _observe_semantic(obs: dict, df: pd.DataFrame) -> None:
    out, report = fd.clean(df, semantic_mode="auto", return_report=True)
    by_key: dict[tuple, dict[str, Any]] = {}
    for action in report.actions:
        if not action.step.startswith("semantic"):
            continue
        meta = action.metadata or {}
        row = meta.get("row")
        if row is None:
            continue
        by_key.setdefault((row, action.column), {
            "status": action.status,
            "confidence": action.confidence,
            "description": action.description,
            "rationale": action.rationale,
        })
    for (row, col), o in obs.items():
        if (row, col) in by_key:
            o.semantic = by_key[(row, col)]
        if col in out.columns and row in out.index:
            o.semantic_value = out.at[row, col]


def _observe_pii(obs: dict, df: pd.DataFrame) -> None:
    for entity in detect_pii(df).entities:
        meta = entity.metadata or {}
        key = (meta.get("row"), meta.get("column"))
        if key in obs:
            o = obs[key]
            o.pii_types = (*o.pii_types, entity.entity_type)


def _check_determinism(fx: GauntletFixture, cleaned: pd.DataFrame, vf: Any) -> bool:
    cleaned2, report2, _, _ = _timed_clean(fx.df)
    vf2 = fd.validate_fields(fx.df, schema=fx.schema)
    return bool(
        cleaned.equals(cleaned2)
        and len(vf.issues) == len(vf2.issues)
        and all(a.reason == b.reason and a.row == b.row and a.column == b.column
                for a, b in zip(vf.issues, vf2.issues))
    )


def run_fixture(fx: GauntletFixture) -> FixtureRun:
    df = fx.df
    obs = {(c.row, c.column): CellObservation(cell=c) for c in fx.cells}

    cleaned, report, clean_s, peak_mb = _timed_clean(df)
    mutations, recorded = _observe_clean(obs, cleaned, report)

    t0 = time.perf_counter()
    vf = fd.validate_fields(df, schema=fx.schema)
    validate_s = time.perf_counter() - t0
    fp_cells = _observe_validate(obs, vf)

    _observe_text(obs, fx, df)
    _observe_semantic(obs, df)
    _observe_pii(obs, df)

    domain_findings = 0
    if fx.domain is not None:
        _, dom_report = fd.clean(df, domain=fx.domain, return_report=True)
        domain_findings = len(dom_report.domain_findings or [])

    deterministic = _check_determinism(fx, cleaned, vf)

    trust_pristine = float(fd.compute_trust_score(fx.pristine()).overall)
    trust_dirty = float(fd.compute_trust_score(df).overall)

    return FixtureRun(
        fixture=fx,
        observations=list(obs.values()),
        false_positive_cells=fp_cells,
        n_clean_cells_checked=int(df.shape[0] * df.shape[1]) - len(fx.cells),
        duplicates_removed=int(report.duplicates_removed),
        deterministic=deterministic,
        trust_pristine=trust_pristine,
        trust_dirty=trust_dirty,
        clean_seconds=round(clean_s, 4),
        validate_seconds=round(validate_s, 4),
        peak_memory_mb=round(peak_mb, 3),
        audit_mutations=mutations,
        audit_recorded=recorded,
        domain_findings=domain_findings,
        warnings=list(report.warnings),
    )


def run_gauntlet(n_rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED,
                 fixtures: list[str] | None = None) -> dict[str, FixtureRun]:
    from .fixtures import build_fixture

    names = fixtures or sorted(FIXTURES)
    return {name: run_fixture(build_fixture(name, n_rows, seed)) for name in names}
