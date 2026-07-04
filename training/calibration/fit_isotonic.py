"""Pool-adjacent-violators isotonic regression per (backend, issue family).

Pure numpy, deterministic. Curves are clamped monotone non-decreasing and
reduced to a compact set of knots compatible with the runtime's
``_IsotonicTable`` JSON format. Deterministic-backend proposals keep an
identity mapping by default (matching the packaged default's contract that
calibration never changes a default install's decisions) unless
``--fit-deterministic`` is passed.

CLI::

    python -m training.calibration.fit_isotonic
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from ..common import BUILD_DIR, read_jsonl, write_json

OUT_DIR = BUILD_DIR / "calibration"
#: Embedding scores may never calibrate to certainty.
EMBEDDING_CEILING = 0.975
MIN_SAMPLES_PER_CURVE = 8


def pav(scores: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: monotone fit of P(correct | score)."""
    order = np.argsort(scores, kind="stable")
    x = scores[order].astype(np.float64)
    y = outcomes[order].astype(np.float64)
    # Blocks: (mean, weight); merge while decreasing.
    means: list[float] = []
    weights: list[float] = []
    xs: list[float] = []
    for xi, yi in zip(x, y):
        means.append(yi)
        weights.append(1.0)
        xs.append(xi)
        while len(means) > 1 and means[-2] > means[-1]:
            total = weights[-2] + weights[-1]
            merged = (means[-2] * weights[-2] + means[-1] * weights[-1]) / total
            means[-2:] = [merged]
            weights[-2:] = [total]
            xs[-2:] = [xs[-1]]
        # xs keeps the right edge of each block
    return np.array(xs), np.array(means)


def fit_curve(
    scores: list[float], outcomes: list[bool], *, ceiling: float = 1.0
) -> dict[str, list[float]] | None:
    """A compact monotone curve ``{"x": [...], "y": [...]}`` or None."""
    if len(scores) < MIN_SAMPLES_PER_CURVE:
        return None
    knot_x, knot_y = pav(np.array(scores), np.array(outcomes, dtype=float))
    knot_y = np.clip(knot_y, 0.0, ceiling)
    xs = [0.0, *[round(float(v), 4) for v in knot_x]]
    ys = [0.0, *[round(float(v), 4) for v in knot_y]]
    # De-duplicate x while keeping monotonicity.
    dedup_x: list[float] = []
    dedup_y: list[float] = []
    for xv, yv in zip(xs, ys):
        if dedup_x and xv <= dedup_x[-1]:
            dedup_y[-1] = max(dedup_y[-1], yv)
            continue
        dedup_x.append(xv)
        dedup_y.append(max(yv, dedup_y[-1] if dedup_y else 0.0))
    if dedup_x[-1] < 1.0:
        dedup_x.append(1.0)
        dedup_y.append(dedup_y[-1])
    if len(dedup_x) < 2:
        return None
    return {"x": dedup_x, "y": dedup_y}


def fit_tables(
    records: list[dict[str, Any]], *, fit_deterministic: bool = False
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Curves per (backend, issue_type) plus a per-backend ``"*"`` fallback."""
    grouped: dict[tuple[str, str], tuple[list[float], list[bool]]] = {}
    for record in records:
        backend = str(record["backend"])
        if backend == "deterministic" and not fit_deterministic:
            continue  # identity by design; never change default decisions
        key = (backend, str(record["issue_type"]))
        scores, outcomes = grouped.setdefault(key, ([], []))
        scores.append(float(record["raw_score"]))
        outcomes.append(bool(record["correct"]))
    tables: dict[str, dict[str, dict[str, list[float]]]] = {}
    for (backend, issue_type), (scores, outcomes) in sorted(grouped.items()):
        ceiling = EMBEDDING_CEILING if backend == "embedding" else 1.0
        curve = fit_curve(scores, outcomes, ceiling=ceiling)
        if curve is not None:
            tables.setdefault(backend, {})[issue_type] = curve
    # Per-backend fallback from all of that backend's records.
    for backend in list(tables):
        scores = [float(r["raw_score"]) for r in records if r["backend"] == backend]
        outcomes = [bool(r["correct"]) for r in records if r["backend"] == backend]
        ceiling = EMBEDDING_CEILING if backend == "embedding" else 1.0
        fallback = fit_curve(scores, outcomes, ceiling=ceiling)
        if fallback is not None:
            tables[backend]["*"] = fallback
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.calibration.fit_isotonic")
    parser.add_argument("--features", default=str(OUT_DIR / "features.jsonl"))
    parser.add_argument("--out", default=str(OUT_DIR / "curves.json"))
    parser.add_argument("--fit-deterministic", action="store_true")
    args = parser.parse_args(argv)
    records = read_jsonl(args.features)
    tables = fit_tables(records, fit_deterministic=args.fit_deterministic)
    write_json(Path(args.out), {"tables": tables, "n_records": len(records)})
    n_curves = sum(len(v) for v in tables.values())
    print(f"isotonic fit: {n_curves} curves from {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
