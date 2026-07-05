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


# --------------------------------------------------------------------------- #
# Full-suite metrics and release gates (Phase 5)
# --------------------------------------------------------------------------- #

#: Full-suite release gates (docs/benchmarks.md). The mini-suite constants
#: above are unchanged so the Phase-2/3 CI tests keep their contract.
FULL_GATE_ECE = 0.03
FULL_GATE_PRECISION_AT_95 = 0.99
FULL_GATE_RUNTIME_SLOWDOWN = 0.20
FULL_GATE_MEMORY_OVERHEAD = 0.15
FULL_GATE_FALSE_MODIFICATION_RATE = 0.001
FULL_GATE_PROTECTED_VIOLATION_RATE = 0.0
FULL_GATE_PRIVACY_LEAK_COUNT = 0


def explainability_rubric_score(report: object) -> float:
    """Deterministic 0..1 rubric over a report's semantic actions.

    Each action earns equal-weight credit for: a non-empty rationale or
    evidence trail, a decision status, and a numeric confidence. Reports
    with no semantic actions score 1.0 (nothing needed explaining).
    """
    actions = [a for a in (getattr(report, "actions", []) or [])
               if getattr(a, "step", None) == "semantic"]
    if not actions:
        return 1.0
    total = 0.0
    for action in actions:
        metadata = getattr(action, "metadata", {}) or {}
        has_reason = bool(getattr(action, "reason", "") or metadata.get("evidence"))
        has_status = bool(getattr(action, "status", ""))
        has_confidence = isinstance(getattr(action, "confidence", None), (int, float))
        total += (has_reason + has_status + has_confidence) / 3.0
    return total / len(actions)


def context_policy_accuracy(policy: object, expected: list[tuple[str, str]]) -> float:
    """Share of expected ``(column, rule)`` constraints the compiler produced."""
    if not expected:
        return 1.0
    produced = {
        (str(getattr(c, "column", "")), str(getattr(c, "rule", "")))
        for c in getattr(policy, "constraints", []) or []
    }
    hit = sum(1 for pair in expected if pair in produced)
    return hit / len(expected)


def policy_slot_f1(policy: object, expected_params: dict[tuple[str, str], dict]) -> float:
    """Micro-F1 over constraint params vs the expected slot values."""
    tp = fp = fn = 0
    produced: dict[tuple[str, str], dict] = {}
    for c in getattr(policy, "constraints", []) or []:
        produced[(str(getattr(c, "column", "")), str(getattr(c, "rule", "")))] = dict(
            getattr(c, "params", {}) or {})
    for key, want in expected_params.items():
        got = produced.get(key, {})
        want_pairs = {(k, str(v).lower()) for k, v in _flatten(want)}
        got_pairs = {(k, str(v).lower()) for k, v in _flatten(got)}
        tp += len(want_pairs & got_pairs)
        fp += len(got_pairs - want_pairs)
        fn += len(want_pairs - got_pairs)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 1.0


def _flatten(params: dict):
    for key, value in params.items():
        if isinstance(value, (list, tuple)):
            for item in value:
                yield key, item
        elif isinstance(value, dict):
            for k, v in value.items():
                yield f"{key}.{k}", v
        else:
            yield key, value


def profile_replay_lift(
    truth: pd.DataFrame,
    corrupted: pd.DataFrame,
    repaired_without: pd.DataFrame,
    repaired_with: pd.DataFrame,
) -> float:
    """Repair-F1 gain from replaying a learned profile on a fresh batch."""
    return cell_repair_f1(truth, corrupted, repaired_with) - cell_repair_f1(
        truth, corrupted, repaired_without)


def privacy_leak_count(profile_json: str, planted_literals: tuple[str, ...]) -> int:
    """How many planted sensitive literals appear verbatim in a saved profile."""
    return sum(1 for literal in planted_literals if literal in profile_json)


#: FreshData's own tracks must never touch the network (offline-by-default).
GATE_NETWORK_CALL_COUNT = 0


def runtime_network_call_count() -> int:
    """Network calls made by the FreshData runtime during a CleanBench track.

    Always ``0``: the default install performs no network I/O in ``fd.clean``,
    ``fd.learn``, or ``fd.compile_context`` — the only network call anywhere in
    the package is the explicit, opt-in ``fd.models.pull(...)``, which no
    CleanBench track invokes. This is a disclosure, not a measurement; the
    "no runtime LLM / no cloud call" README claim is enforced separately by
    ``tests/test_no_network_in_runtime.py`` (patches ``socket`` to fail loudly).
    """
    return 0


def determinism_score(scores: list[float]) -> float:
    """``1.0`` when every repeated run produced the identical score, else the
    fraction of runs that agree with the modal (most common) score.

    Used for the disclosed LLM-agent baseline (``REPEATS`` runs of the same
    prompt) to measure run-to-run determinism honestly instead of assuming it.
    """
    if not scores:
        return 0.0
    rounded = [round(s, 6) for s in scores]
    modal_count = max(rounded.count(v) for v in set(rounded))
    return modal_count / len(rounded)


def cost_usd_per_1m_rows(cost_usd: float | None, rows: int) -> float | None:
    """Extrapolate a measured run cost to a per-million-row rate, or ``None``
    when cost is unknown (e.g. a baseline that was skipped)."""
    if cost_usd is None or rows <= 0:
        return None
    return round(cost_usd * (1_000_000 / rows), 6)
