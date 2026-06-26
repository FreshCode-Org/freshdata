"""Lightweight, bounded schema- and distribution-drift detection between batches.

Compares an incoming (already representation-repaired) batch against the schema
baseline and the running per-column state. Findings are cheap to compute from the
state that streaming already maintains — no extra passes over historical rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._config import StreamingCleanConfig
    from ._state import StreamingState


@dataclass(frozen=True)
class DriftFinding:
    """One detected drift event for a single column (or the schema as a whole)."""

    #: One of: new_column, missing_column, dtype_change, missing_rate, cardinality, distribution.
    kind: str
    column: str | None
    message: str
    risk: str = "medium"


def detect_drift(
    df: pd.DataFrame, state: StreamingState, config: StreamingCleanConfig
) -> list[DriftFinding]:
    """Return drift findings comparing *df* to the baseline + running *state*.

    Called after the baseline is locked (i.e. from the second batch on); the first
    batch establishes the baseline and never reports drift.
    """
    findings: list[DriftFinding] = []
    if not state.schema_baseline:
        return findings

    batch_cols = [str(c) for c in df.columns]
    baseline = set(state.schema_baseline)
    seen = set(batch_cols)

    for col in batch_cols:
        if col not in baseline:
            findings.append(DriftFinding(
                "new_column", col,
                f"new column '{col}' not present in the schema baseline",
            ))
    for col in state.schema_baseline:
        if col not in seen:
            findings.append(DriftFinding(
                "missing_column", col,
                f"baseline column '{col}' is absent from this batch",
            ))

    for col in batch_cols:
        if col not in baseline:
            continue
        s = df[col]
        baseline_dtype = state.baseline_dtypes.get(col)
        if baseline_dtype is not None and str(s.dtype) != baseline_dtype:
            findings.append(DriftFinding(
                "dtype_change", col,
                f"dtype of '{col}' changed: baseline {baseline_dtype} -> {s.dtype}",
            ))
        cstate = state.columns.get(col)
        if cstate is None:
            continue
        _missing_rate_drift(s, cstate, config, findings)
        _cardinality_drift(s, cstate, config, findings)
        _distribution_drift(s, cstate, config, findings)
    return findings


def _missing_rate_drift(s, cstate, config, findings) -> None:
    if cstate.seen == 0:
        return
    batch_ratio = float(s.isna().mean()) if len(s) else 0.0
    if abs(batch_ratio - cstate.missing_ratio) > config.drift_missing_jump:
        findings.append(DriftFinding(
            "missing_rate", cstate.name,
            f"missing rate of '{cstate.name}' shifted to {batch_ratio:.0%} "
            f"(running {cstate.missing_ratio:.0%})",
        ))


def _cardinality_drift(s, cstate, config, findings) -> None:
    running_mean = cstate.batch_nunique_mean
    if running_mean <= 0 or cstate.first_seen_batch == cstate.seen:
        return
    try:
        batch_nunique = int(s.nunique(dropna=True))
    except TypeError:
        return
    if batch_nunique > config.drift_cardinality_factor * running_mean and batch_nunique > 20:
        findings.append(DriftFinding(
            "cardinality", cstate.name,
            f"cardinality of '{cstate.name}' jumped to {batch_nunique} "
            f"(running avg {running_mean:.0f} distinct/batch)",
        ))


def _distribution_drift(s, cstate, config, findings) -> None:
    if not (is_numeric_dtype(s) and not is_bool_dtype(s)):
        return
    snap = cstate.numeric_snapshot()
    if snap.count < 2 or snap.std <= 0:
        return
    nonnull = pd.to_numeric(s, errors="coerce").dropna()
    if nonnull.empty:
        return
    z = abs(float(nonnull.mean()) - snap.mean) / snap.std
    if z > config.drift_zscore:
        findings.append(DriftFinding(
            "distribution", cstate.name,
            f"mean of '{cstate.name}' shifted {z:.1f}σ from the running mean "
            f"({snap.mean:.4g})",
            risk="high",
        ))
