"""CleanBench metrics and release gates.

All metrics operate on three frames with the **same shape and column set**:

- ``truth`` — the known-clean frame the fixture started from;
- ``corrupted`` — what the cleaner was given;
- ``repaired`` — what the cleaner returned.

(Row-count-changing corruptions — duplicate injection — are measured at the
row level by the caller, not by these cell metrics.) Comparison is by
canonical string form so ``25`` vs ``25.0`` vs ``"25"`` after a dtype repair
does not count as a modification.
"""

from __future__ import annotations

import numbers

import pandas as pd

#: Release gates for the Phase-2 fixtures (see README.md).
GATE_PROTECTED_VIOLATION_RATE = 0.0
GATE_FALSE_MODIFICATION_RATE = 0.001  # 0.1%
GATE_RUNTIME_SLOWDOWN = 0.20  # informational; needs the timing harness


def _canon(value: object) -> object:
    """Canonical comparable form of one cell (NaN-safe, numeric-tolerant)."""
    if value is None:
        return None
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if pd.isna(value):
            return None
        as_float = float(value)
        return as_float if as_float != int(as_float) else int(as_float)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return _canon(float(stripped)) if stripped else stripped
        except ValueError:
            return stripped
    return value


def _check_aligned(*frames: pd.DataFrame) -> None:
    first = frames[0]
    for frame in frames[1:]:
        if frame.shape != first.shape or list(frame.columns) != list(first.columns):
            raise ValueError(
                "CleanBench cell metrics need identically-shaped frames with "
                f"the same columns; got {first.shape}/{list(first.columns)} vs "
                f"{frame.shape}/{list(frame.columns)}"
            )


def _cells(frame: pd.DataFrame):
    for col in frame.columns:
        values = frame[col].tolist()
        for pos, value in enumerate(values):
            yield (pos, str(col)), value


def _cell_sets(truth: pd.DataFrame, corrupted: pd.DataFrame, repaired: pd.DataFrame):
    truth_map = {k: _canon(v) for k, v in _cells(truth)}
    corrupted_map = {k: _canon(v) for k, v in _cells(corrupted)}
    repaired_map = {k: _canon(v) for k, v in _cells(repaired)}
    corrupted_cells = {k for k, v in corrupted_map.items() if v != truth_map[k]}
    changed_cells = {k for k, v in repaired_map.items() if v != corrupted_map[k]}
    return truth_map, corrupted_map, repaired_map, corrupted_cells, changed_cells


def false_modification_rate(
    truth: pd.DataFrame, corrupted: pd.DataFrame, repaired: pd.DataFrame
) -> float:
    """Share of *already-correct* cells that the cleaner changed anyway."""
    _check_aligned(truth, corrupted, repaired)
    truth_map, _, repaired_map, corrupted_cells, changed_cells = _cell_sets(
        truth, corrupted, repaired
    )
    clean_cells = set(truth_map) - corrupted_cells
    if not clean_cells:
        return 0.0
    false_mods = {
        k for k in changed_cells & clean_cells if repaired_map[k] != truth_map[k]
    }
    return len(false_mods) / len(clean_cells)


def protected_column_violation_rate(
    corrupted: pd.DataFrame,
    repaired: pd.DataFrame,
    protected_columns: list[str] | tuple[str, ...],
) -> float:
    """Share of protected columns whose values did not survive byte-identical."""
    if not protected_columns:
        return 0.0
    violations = 0
    for col in protected_columns:
        if col not in repaired.columns:
            violations += 1
            continue
        before, after = corrupted[col], repaired[col]
        aligned = before
        if len(after) < len(before):
            try:
                aligned = before.loc[after.index]
            except KeyError:
                violations += 1
                continue
        if str(aligned.dtype) != str(after.dtype) or not aligned.equals(after):
            violations += 1
    return violations / len(protected_columns)


def cell_repair_precision(
    truth: pd.DataFrame, corrupted: pd.DataFrame, repaired: pd.DataFrame
) -> float:
    """Of the cells the cleaner changed, the share it changed to the truth."""
    _check_aligned(truth, corrupted, repaired)
    truth_map, _, repaired_map, _, changed_cells = _cell_sets(
        truth, corrupted, repaired
    )
    if not changed_cells:
        return 1.0
    correct = {k for k in changed_cells if repaired_map[k] == truth_map[k]}
    return len(correct) / len(changed_cells)


def cell_repair_recall(
    truth: pd.DataFrame, corrupted: pd.DataFrame, repaired: pd.DataFrame
) -> float:
    """Of the corrupted cells, the share restored exactly to the truth."""
    _check_aligned(truth, corrupted, repaired)
    truth_map, _, repaired_map, corrupted_cells, _ = _cell_sets(
        truth, corrupted, repaired
    )
    if not corrupted_cells:
        return 1.0
    restored = {k for k in corrupted_cells if repaired_map[k] == truth_map[k]}
    return len(restored) / len(corrupted_cells)


def cell_repair_f1(
    truth: pd.DataFrame, corrupted: pd.DataFrame, repaired: pd.DataFrame
) -> float:
    """Harmonic mean of :func:`cell_repair_precision` and :func:`cell_repair_recall`."""
    precision = cell_repair_precision(truth, corrupted, repaired)
    recall = cell_repair_recall(truth, corrupted, repaired)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------- #
# Calibration metrics (Phase 3)
# --------------------------------------------------------------------------- #

#: Phase-3 release gates on the mini-suite fixtures. Full-suite targets are
#: stricter (ECE <= 0.03, precision@0.95 >= 0.99) and documented in
#: docs/benchmarks.md as the roadmap goal, not a current gate.
GATE_ECE = 0.05
GATE_PRECISION_AT_95 = 0.98

_OUTCOME_STATUSES = frozenset({"automatic", "suggested", "approved", "accepted"})


def confidence_outcomes(
    report: object, truth: pd.DataFrame, corrupted: pd.DataFrame
) -> list[tuple[float, bool]]:
    """Extract ``(confidence, correct)`` pairs from a report's semantic actions.

    A proposal is *correct* when every cell holding its raw value in the
    corrupted frame equals its proposed value in the truth frame (canonical
    comparison). Flag-only actions (no proposed value) and skipped decisions
    carry no repair claim and are excluded.
    """
    pairs: list[tuple[float, bool]] = []
    for action in getattr(report, "actions", []) or []:
        if getattr(action, "step", None) != "semantic":
            continue
        if str(getattr(action, "status", "")) not in _OUTCOME_STATUSES:
            continue
        metadata = getattr(action, "metadata", {}) or {}
        raw = metadata.get("raw_value")
        proposed = metadata.get("proposed_value")
        column = getattr(action, "column", None)
        if proposed is None or column is None or column not in corrupted.columns:
            continue
        raw_canon = _canon(raw)
        matches = [
            pos for pos, value in enumerate(corrupted[column].tolist())
            if _canon(value) == raw_canon
        ]
        if not matches:
            continue
        truth_values = truth[column].tolist()
        proposed_canon = _canon(proposed)
        correct = all(_canon(truth_values[pos]) == proposed_canon for pos in matches)
        pairs.append((float(getattr(action, "confidence", 0.0)), bool(correct)))
    return pairs


def expected_calibration_error(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    """Standard equal-width-bin ECE over ``(confidence, correct)`` pairs.

    Zero when there is nothing to score — an empty benchmark is not evidence
    of miscalibration.
    """
    if not pairs:
        return 0.0
    total = len(pairs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [ok for conf, ok in pairs if (conf > lo or b == 0) and conf <= hi]
        if not bucket:
            continue
        accuracy = sum(bucket) / len(bucket)
        mean_conf = sum(conf for conf, _ in pairs if (conf > lo or b == 0) and conf <= hi)
        mean_conf /= len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - mean_conf)
    return ece


def precision_at_confidence_bucket(
    pairs: list[tuple[float, bool]], floor: float = 0.95
) -> float:
    """Precision among proposals whose confidence is at least *floor*.

    1.0 when the bucket is empty: claiming nothing at high confidence is not a
    precision failure (coverage measures that axis instead).
    """
    bucket = [ok for conf, ok in pairs if conf >= floor]
    if not bucket:
        return 1.0
    return sum(bucket) / len(bucket)


def coverage_at_precision(
    pairs: list[tuple[float, bool]], target_precision: float = 0.98
) -> float:
    """Largest share of proposals acceptable at *target_precision* or better.

    Sweeps confidence thresholds (each observed confidence value) and returns
    the best coverage whose bucket precision clears the target — the
    abstention-quality curve reduced to one number.
    """
    if not pairs:
        return 0.0
    best = 0.0
    for threshold in sorted({conf for conf, _ in pairs}):
        bucket = [ok for conf, ok in pairs if conf >= threshold]
        if not bucket:
            continue
        precision = sum(bucket) / len(bucket)
        if precision >= target_precision:
            best = max(best, len(bucket) / len(pairs))
    return best
