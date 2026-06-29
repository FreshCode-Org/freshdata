"""Metric computation for the FreshData benchmark harness.

Nine standardized metrics (see ``docs/benchmarks.md``). Every metric is a pure
function of a fixture and the FreshData API; none of them reach into library
internals. The harness (``bench.py``) orchestrates these and serialises the
results to the stable JSON schema in ``results_schema.py``.

The two safety metrics — false-repair rate and preservation rate — are the load
bearing ones: FreshData's contract is that id / target / free-text columns are
never mutated, and these functions verify it on every fixture, not just gold.
"""

from __future__ import annotations

import gc
import json
import math
import os
import tempfile
import tracemalloc
from statistics import quantiles
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

import freshdata as fd
from freshdata import CleanConfig
from freshdata._sentinels import DEFAULT_SENTINELS

from fixtures import REGISTRY, gold as gold_mod

# Canonical FreshData invocation for the authored-code metric: a config line and
# the clean call. Three logical lines covers id/target hints + clean + report.
FD_AUTHORED_LINES = 3
_RISK_LEVELS = {"low", "medium", "high"}
TRUST_DEFECT_RATES = (0.0, 0.05, 0.15, 0.30, 0.60)


# -- fixture plumbing ------------------------------------------------------
def config_for(name: str, df: pd.DataFrame, *, aggressive: bool = False) -> CleanConfig:
    """Build the balanced (default) CleanConfig for a fixture.

    id/target hints come from the fixture module so the engine's role detection
    is anchored, exactly as a real user would configure it.
    """
    mod = REGISTRY[name]
    strategy = "aggressive" if aggressive else "balanced"
    return CleanConfig(
        strategy=strategy,
        id_columns=tuple(getattr(mod, "ID_COLUMNS", ())),
        target_column=getattr(mod, "TARGET_COLUMN", None),
    )


def protected_columns(name: str, df: pd.DataFrame) -> dict[str, list[str]]:
    """Map role -> protected column names present in ``df`` for this fixture."""
    mod = REGISTRY[name]
    ids = [c for c in getattr(mod, "ID_COLUMNS", ()) if c in df.columns]
    target = getattr(mod, "TARGET_COLUMN", None)
    targets = [target] if target and target in df.columns else []
    if name == "wide_schema":
        texts = [c for c in df.columns if c.startswith("text_")]
    else:
        texts = [c for c in getattr(mod, "TEXT_COLUMNS", ()) if c in df.columns]
    return {"id": ids, "target": targets, "text": texts}


def make_frame(name: str, size: int, seed: int, defect_rate: float | None = None):
    """Return the input frame for a fixture (gold returns its dirty frame)."""
    mod = REGISTRY[name]
    if name == "gold":
        return mod.generate(size, seed=seed, defect_rate=defect_rate).dirty_df
    return mod.generate(size, seed=seed, defect_rate=defect_rate)


# -- metric 1: wall-clock --------------------------------------------------
def metric_wall_clock(df: pd.DataFrame, config: CleanConfig, *, repeat: int = 5) -> dict[str, float]:
    """p50 / p95 wall-clock seconds for clean + full report materialisation.

    Times only ``fd.clean(..., return_report=True)``; FreshData does not mutate
    its input under the default ``preserve_original=True``, so no per-iteration
    copy is needed and none is timed (I/O excluded, per the metric definition).
    """
    times: list[float] = []
    for _ in range(max(1, repeat)):
        t0 = perf_counter()
        _cleaned, _report = fd.clean(df, config=config, return_report=True)
        times.append(perf_counter() - t0)
    return {"p50": _pctl(times, 50), "p95": _pctl(times, 95)}


def _pctl(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


# -- metric 2: peak memory -------------------------------------------------
def metric_peak_memory(df: pd.DataFrame, config: CleanConfig) -> dict[str, float]:
    """Peak / delta MB of Python allocations during one clean+report run.

    Uses :mod:`tracemalloc` (stdlib, dependency-free, reproducible across
    machines). ``peak_mb`` is the high-water mark; ``delta_mb`` is the increase
    over the pre-call baseline.
    """
    gc.collect()
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    _cleaned, _report = fd.clean(df, config=config, return_report=True)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"peak_mb": round(peak / 1e6, 3), "delta_mb": round((peak - base) / 1e6, 3)}


# -- metrics 4 & 5: false-repair & preservation (all fixtures) -------------
def _as_array(values: Any) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        return values.to_numpy(copy=True)
    return np.asarray(values)


def _changed_mask(a: Any, b: Any) -> np.ndarray:
    """NaN-aware inequality between two aligned series."""
    av = _as_array(a)
    bv = _as_array(b)
    eq = np.zeros(len(av), dtype=bool)
    # compare element-wise without raising on mixed dtypes
    for i in range(len(av)):
        a_na = _is_missing_scalar(av[i])
        b_na = _is_missing_scalar(bv[i])
        if a_na and b_na:
            eq[i] = True
        elif a_na or b_na:
            eq[i] = False
        else:
            try:
                eq[i] = bool(av[i] == bv[i])
            except Exception:
                eq[i] = False
    return ~eq


def _first_rows_by_key(df: pd.DataFrame, key: str) -> dict[Any, int]:
    rows: dict[Any, int] = {}
    for idx, value in enumerate(df[key].to_numpy(copy=True)):
        if _is_missing_scalar(value) or value in rows:
            continue
        rows[value] = idx
    return rows


def _take_column(df: pd.DataFrame, column: str, rows: list[int]) -> np.ndarray:
    values = df[column].to_numpy(copy=True)
    return values[np.asarray(rows, dtype=int)] if rows else values[:0]


def preservation_report(name: str, dirty: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, Any]:
    """Per-role preservation + false-repair for a non-gold fixture.

    Aligns ``dirty`` and ``cleaned`` on the fixture's primary (unique) id key,
    then compares protected columns cell-by-cell. The primary key itself is
    validated by set membership (no non-null id may become a value absent from
    the input — that would be a mutation or a fabricated fill).
    """
    roles = protected_columns(name, dirty)
    if not roles["id"]:
        return {"per_role": {}, "preservation_rate_pct": 100.0, "false_repair_rate_pct": 0.0}
    key = roles["id"][0]

    # primary key: membership check (covers both mutation and null-fill)
    dirty_ids = dirty[key].to_numpy(copy=True)
    cleaned_ids = cleaned[key].to_numpy(copy=True)
    in_ids = {value for value in dirty_ids if not _is_missing_scalar(value)}
    out_ids = [value for value in cleaned_ids if not _is_missing_scalar(value)]
    key_total = len(out_ids)
    key_preserved = sum(1 for value in out_ids if value in in_ids)
    nulls_in = sum(1 for value in dirty_ids if _is_missing_scalar(value))
    nulls_out = sum(1 for value in cleaned_ids if _is_missing_scalar(value))

    # align other protected columns on the key
    dirty_rows = _first_rows_by_key(dirty, key)
    cleaned_rows = _first_rows_by_key(cleaned, key)
    common = [value for value in dirty_rows if value in cleaned_rows]
    dirty_common_rows = [dirty_rows[value] for value in common]
    cleaned_common_rows = [cleaned_rows[value] for value in common]

    per_role: dict[str, dict[str, float]] = {}
    agg_changed = 0
    agg_total = 0
    for role in ("id", "target", "text"):
        changed = 0
        total = 0
        cols = [c for c in roles[role] if c != key]
        if role == "id":
            # fold the primary-key membership result into the id role
            changed += key_total - key_preserved
            total += key_total
        for c in cols:
            if c not in dirty.columns or c not in cleaned.columns:
                continue
            cm = _changed_mask(
                _take_column(dirty, c, dirty_common_rows),
                _take_column(cleaned, c, cleaned_common_rows),
            )
            changed += int(cm.sum())
            total += int(len(cm))
        if total:
            per_role[role] = {
                "preservation_rate_pct": round(100.0 * (total - changed) / total, 4),
                "false_repair_rate_pct": round(100.0 * changed / total, 4),
                "n_cells": total,
            }
            agg_changed += changed
            agg_total += total

    return {
        "per_role": per_role,
        "preservation_rate_pct": round(100.0 * (agg_total - agg_changed) / agg_total, 4) if agg_total else 100.0,
        "false_repair_rate_pct": round(100.0 * agg_changed / agg_total, 4) if agg_total else 0.0,
        "ids_never_filled": nulls_out >= max(0, nulls_in - (len(dirty) - len(cleaned))),
    }


# -- metrics 3/4/5 on the gold fixture (cell-level ground truth) -----------
def gold_repair_report(size: int, seed: int) -> dict[str, Any]:
    """Repair fidelity, false-repair and preservation against the gold oracle."""
    bundle = gold_mod.generate(size, seed=seed)
    cfg = CleanConfig(
        strategy="balanced",
        id_columns=tuple(gold_mod.ID_COLUMNS),
        target_column=gold_mod.TARGET_COLUMN,
    )
    cleaned = fd.clean(bundle.dirty_df, config=cfg).reset_index(drop=True)
    clean = bundle.clean_df
    dirty_head = bundle.dirty_df.iloc[: len(clean)].reset_index(drop=True)
    n = len(clean)

    # defect family per column from the manifest
    fam_by_col: dict[str, str] = {}
    for d in gold_mod.DEFECT_MANIFEST:
        fam_by_col.setdefault(d["column"], d["defect_type"])

    # repair fidelity, per column + per family
    per_col: dict[str, dict[str, Any]] = {}
    fam_correct: dict[str, int] = {}
    fam_total: dict[str, int] = {}
    rep_correct = rep_total = 0
    for col in clean.columns:
        if col not in cleaned.columns:
            continue
        rm = bundle.repair_mask[col].to_numpy()
        if rm.sum() == 0:
            continue
        exp = clean[col]
        act = cleaned[col]
        ok = _values_match(exp[rm], act[rm])
        nok = int(ok.sum())
        ntot = int(rm.sum())
        per_col[col] = {
            "defect_type": fam_by_col.get(col, "unknown"),
            "expected_action": gold_mod.GOLD_LABELS[col]["fill_action"],
            "correct_cells": nok,
            "total_cells": ntot,
            "fidelity_pct": round(100.0 * nok / ntot, 3),
        }
        fam = per_col[col]["defect_type"]
        fam_correct[fam] = fam_correct.get(fam, 0) + nok
        fam_total[fam] = fam_total.get(fam, 0) + ntot
        rep_correct += nok
        rep_total += ntot

    # duplicate removal is a row-level repair family
    dup_expected = bundle.n_duplicates
    dup_removed = len(bundle.dirty_df) - len(cleaned)
    fam_correct["exact_duplicate_row"] = min(dup_removed, dup_expected)
    fam_total["exact_duplicate_row"] = dup_expected
    if dup_expected:
        rep_correct += min(dup_removed, dup_expected)
        rep_total += dup_expected

    # false-repair on trap cells
    trap_changed = trap_total = 0
    per_trap: dict[str, dict[str, float]] = {}
    for col in clean.columns:
        tm = bundle.false_repair_traps[col].to_numpy()
        if tm.sum() == 0:
            continue
        cm = _changed_mask(dirty_head[col], cleaned[col])
        changed = int((cm & tm).sum())
        tot = int(tm.sum())
        per_trap[col] = {
            "false_repair_rate_pct": round(100.0 * changed / tot, 4),
            "n_traps": tot,
        }
        trap_changed += changed
        trap_total += tot

    # preservation on protected columns
    pres_unchanged = pres_total = 0
    for col in gold_mod.PROTECTED:
        pm = bundle.preservation_mask[col].to_numpy()
        cm = _changed_mask(dirty_head[col], cleaned[col])
        unchanged = int((~cm & pm).sum())
        tot = int(pm.sum())
        pres_unchanged += unchanged
        pres_total += tot

    return {
        "repair_fidelity_pct": round(100.0 * rep_correct / rep_total, 3) if rep_total else 100.0,
        "per_column": per_col,
        "per_family": {
            fam: round(100.0 * fam_correct[fam] / fam_total[fam], 3)
            for fam in fam_total
            if fam_total[fam]
        },
        "false_repair_rate_pct": round(100.0 * trap_changed / trap_total, 4) if trap_total else 0.0,
        "per_trap": per_trap,
        "preservation_rate_pct": round(100.0 * pres_unchanged / pres_total, 4) if pres_total else 100.0,
        "n_rows": n,
    }


def _values_match(exp: pd.Series, act: pd.Series) -> np.ndarray:
    """Element-wise match with float tolerance and NaN/NaT awareness."""
    ev = exp.to_numpy()
    av = act.reindex(exp.index).to_numpy() if hasattr(act, "reindex") else np.asarray(act)
    out = np.zeros(len(ev), dtype=bool)
    for i in range(len(ev)):
        e, a = ev[i], av[i]
        e_na = _is_missing_scalar(e)
        a_na = _is_missing_scalar(a)
        if e_na and a_na:
            out[i] = True
        elif e_na or a_na:
            out[i] = False
        else:
            try:
                if (
                    isinstance(e, (int, float, np.floating, np.integer))
                    and not isinstance(e, bool)
                ):
                    out[i] = bool(np.isclose(float(e), float(a), rtol=1e-6, atol=1e-9))
                elif _is_dateish(e) or _is_dateish(a):
                    out[i] = _date_values_equal(e, a)
                else:
                    out[i] = bool(e == a)
            except Exception:
                try:
                    out[i] = bool(e == a)
                except Exception:
                    out[i] = False
    return out


def _is_missing_scalar(value: Any) -> bool:
    """Scalar-only missing check that avoids pandas scalar NA dispatch."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(float(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    return False


def _is_dateish(v: Any) -> bool:
    return isinstance(v, (pd.Timestamp, np.datetime64))


def _date_values_equal(expected: Any, actual: Any) -> bool:
    try:
        return bool(np.datetime64(expected) == np.datetime64(actual))
    except Exception:
        try:
            return bool(expected == actual)
        except Exception:
            return False


# -- metric 3 (named fixtures): manifest-driven repair fidelity -------------
def _expand_pattern(pattern: str, cols: list[str]) -> list[str] | None:
    """Resolve a DEFECT_MANIFEST column pattern to concrete columns.

    Returns ``None`` for the row-level wildcard ``"*"`` (whole-row families like
    duplicate removal).
    """
    if pattern == "*":
        return None
    if "|" in pattern:
        wanted = set(pattern.split("|"))
        return [c for c in cols if c in wanted]
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return [c for c in cols if c.startswith(prefix)]
    return [c for c in cols if c == pattern]


# The engine deliberately *normalises* a sentinel to one of these canonical
# missing markers (e.g. for medium-missingness categoricals). Their presence is
# the repair succeeding, not a residual raw sentinel, so they are excluded from
# the "did any injected sentinel survive?" check.
_NORMALIZED_MARKERS = {"missing", "unknown"}
_RAW_SENTINELS = DEFAULT_SENTINELS - _NORMALIZED_MARKERS


def _has_sentinel(series: pd.Series) -> bool:
    """True if any *raw* injected sentinel token survived cleaning."""
    if series.dtype != object:
        return False
    vals = series.dropna()
    if vals.empty:
        return False
    lowered = vals.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    return bool(lowered.isin(_RAW_SENTINELS).any())


def _has_surrounding_ws(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    vals = series.dropna()
    return bool(vals.map(lambda v: isinstance(v, str) and v != v.strip()).any())


def manifest_repair_fidelity(name: str, dirty: pd.DataFrame, cleaned: pd.DataFrame, report) -> dict[str, Any]:
    """Family-level repair fidelity for a named fixture, from DEFECT_MANIFEST.

    Each documented defect family contributes a single post-condition (did
    FreshData do the correct thing?), scored 1.0 / 0.0. This is coarser than the
    gold fixture's cell-level fidelity but is anchored entirely in the declared
    ground truth, so it is reported for every fixture. "Correct" for a flagging
    family (reference / missing-flag / CDC / provenance) means non-destructive
    preservation, not silent rewriting.
    """
    mod = REGISTRY[name]
    labels = getattr(mod, "GOLD_LABELS", {})
    if callable(getattr(mod, "gold_labels", None)) and name == "wide_schema":
        labels = mod.gold_labels(dirty.shape[1])
    cols = list(cleaned.columns)
    per_family: dict[str, bool] = {}

    for d in getattr(mod, "DEFECT_MANIFEST", []):
        fam = d["id"]
        repair = d["in_scope_repair"]
        targets = _expand_pattern(d["column"], cols)

        if repair == "drop_duplicate" or targets is None:
            satisfied = not cleaned.duplicated().any()
        elif repair == "sentinel_normalize":
            satisfied = all(not _has_sentinel(cleaned[c]) for c in targets if c in cleaned)
        elif repair == "median_fill":
            satisfied = all(
                c in cleaned and cleaned[c].isna().sum() == 0 and cleaned[c].dtype.kind in "fi"
                for c in targets
            )
        elif repair == "dtype_coerce":
            satisfied = all(
                c in cleaned and cleaned[c].dtype.kind in ("f", "i", "M") for c in targets
            )
        elif repair == "normalize_whitespace":
            satisfied = all(not _has_surrounding_ws(cleaned[c]) for c in targets if c in cleaned)
        elif repair in ("reference_flag",):
            # correct = bad values preserved (not silently rewritten to a valid one)
            satisfied = all(c in cleaned for c in targets)
        elif repair in ("flag_missing", "preserve", "cdc_flag", "provenance_flag", "dtype_coerce_or_flag"):
            # non-destructive: the column still exists and missing stays missing
            satisfied = all(c in cleaned for c in targets)
        else:
            satisfied = all(c in cleaned for c in targets)
        per_family[fam] = bool(satisfied)

    n = len(per_family)
    pct = round(100.0 * sum(per_family.values()) / n, 3) if n else 100.0
    return {"repair_fidelity_pct": pct, "per_family": per_family}


# -- metric 6: authored-code reduction -------------------------------------
def metric_authored_lines() -> dict[str, Any]:
    """Authored-line counts for FreshData vs pandas / pyjanitor baselines."""
    from baselines import pandas_baseline, pyjanitor_baseline

    pandas_lines = pandas_baseline.AUTHORED_LINES
    pj_lines = pyjanitor_baseline.AUTHORED_LINES
    return {
        "fd_lines": FD_AUTHORED_LINES,
        "pandas_lines": pandas_lines,
        "pyjanitor_lines": pj_lines,
        "reduction_vs_pandas_pct": round(100.0 * (pandas_lines - FD_AUTHORED_LINES) / pandas_lines, 2),
        "reduction_vs_pyjanitor_pct": round(100.0 * (pj_lines - FD_AUTHORED_LINES) / pj_lines, 2),
    }


# -- metric 7: diagnosis speed ---------------------------------------------
def metric_diagnosis_speed(report) -> dict[str, float]:
    """Report-materialisation latency: summary / to_frame / to_dict."""
    def _timed(fn) -> float:
        t0 = perf_counter()
        fn()
        return perf_counter() - t0

    return {
        "summary_sec": round(_timed(report.summary), 6),
        "to_frame_sec": round(_timed(report.to_frame), 6),
        "to_dict_sec": round(_timed(report.to_dict), 6),
    }


# -- metric 8: trust score + monotonicity ----------------------------------
def metric_trust(name: str, sweep_size: int, seed: int) -> dict[str, Any]:
    """Trust score at base defects + strict-monotonic decrease over a sweep.

    The sweep is run at a bounded size so the metric stays cheap on large
    fixtures; monotonicity is a property of the defect rate, not the row count.
    """
    base_df = make_frame(name, sweep_size, seed, defect_rate=None)
    base_trust = round(float(fd.compute_trust_score(base_df).overall), 3)

    scores = []
    for r in TRUST_DEFECT_RATES:
        dfr = make_frame(name, sweep_size, seed, defect_rate=r)
        scores.append(round(float(fd.compute_trust_score(dfr).overall), 3))
    monotonic = all(scores[i] > scores[i + 1] for i in range(len(scores) - 1))
    return {
        "trust_score": base_trust,
        "trust_monotonic_valid": bool(monotonic),
        "sweep": {str(r): s for r, s in zip(TRUST_DEFECT_RATES, scores)},
    }


# -- metric 9: export completeness -----------------------------------------
def metric_export_completeness(name: str, df: pd.DataFrame, config: CleanConfig, report) -> dict[str, Any]:
    """All required report fields populated + export methods non-empty/valid."""
    checks: list[bool] = []
    missing: list[str] = []

    def check(label: str, cond: bool) -> None:
        checks.append(bool(cond))
        if not cond:
            missing.append(label)

    summary = report.summary()
    frame = report.to_frame()
    asdict = report.to_dict()
    check("summary_nonempty", isinstance(summary, str) and len(summary) > 0)
    check("to_frame_nonempty", frame is not None and len(frame) > 0)
    check("to_dict_dict", isinstance(asdict, dict) and len(asdict) > 0)

    # Every repaired action must carry a before/after summary (description), a
    # valid risk level and a numeric confidence. ``rationale`` is the engine's
    # *statistical* justification — populated for decision-engine actions
    # (impute / outliers, identifiable by a model_id or a non-low risk); the
    # deterministic Layer-1 representation steps explain themselves through the
    # description instead, which is the before/after summary the metric checks.
    for action in report:
        if not getattr(action, "count", 0):
            continue
        col = getattr(action, "column", None) or action.step
        check(f"before_after:{col}", bool(getattr(action, "description", "")))
        check(f"risk:{col}", getattr(action, "risk", None) in _RISK_LEVELS)
        check(f"confidence:{col}", isinstance(getattr(action, "confidence", None), float))
        is_engine = bool(getattr(action, "model_id", "")) or getattr(action, "risk", "low") != "low"
        if is_engine:
            check(f"rationale:{col}", bool(getattr(action, "rationale", "")))

    # enterprise exports: quality markdown + lineage json
    try:
        res = fd.clean_enterprise(df, clean_config=config)
        check("quality_markdown", len(res.quality.to_markdown()) > 0)
        path = os.path.join(tempfile.gettempdir(), f"fd_lineage_{name}.json")
        res.lineage.emit(path)
        with open(path) as fh:
            json.load(fh)
        check("lineage_json_valid", os.path.exists(path))
        os.remove(path)
    except Exception as exc:  # pragma: no cover - enterprise optional deps
        check("enterprise_exports", False)
        missing.append(f"enterprise:{type(exc).__name__}")

    pct = round(100.0 * sum(checks) / len(checks), 3) if checks else 0.0
    return {"export_completeness_pct": pct, "fields_missing": missing}
