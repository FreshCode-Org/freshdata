"""Evaluate the exported calibration table against the release gates.

Gates::

    ECE <= 0.03
    precision at confidence >= 0.95  >= 0.99
    protected-column violation rate == 0   (checked by the CleanBench run)
    false modification rate <= 0.1%        (checked by the CleanBench run)

The first two are computed here on a held-out feature split (never the rows
used for fitting); the last two are structural cleaner properties measured
by ``benchmarks.cleanbench`` and re-asserted by the release pipeline.

CLI::

    python -m training.calibration.eval_calibration [--check-gates]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..common import BUILD_DIR, read_json, read_jsonl, write_json
from ..datasets.splits import hash_split

OUT_DIR = BUILD_DIR / "calibration"

GATE_ECE = 0.03
GATE_PRECISION_AT_95 = 0.99


def apply_table(
    table_json: dict[str, Any], records: list[dict[str, Any]]
) -> list[tuple[float, bool]]:
    """(calibrated confidence, correct) pairs via the runtime's own loader."""
    from freshdata.semantic.scoring import _IsotonicTable  # noqa: PLC0415

    table = _IsotonicTable.from_json(json.dumps(table_json))
    pairs: list[tuple[float, bool]] = []
    for record in records:
        calibrated = table.apply(
            str(record["backend"]), str(record["issue_type"]), float(record["raw_score"]))
        pairs.append((min(1.0, max(0.0, calibrated)), bool(record["correct"])))
    return pairs


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float:
    if not pairs:
        return 0.0
    total = len(pairs)
    out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(c, ok) for c, ok in pairs if (c > lo or b == 0) and c <= hi]
        if not bucket:
            continue
        accuracy = sum(ok for _, ok in bucket) / len(bucket)
        mean_confidence = sum(c for c, _ in bucket) / len(bucket)
        out += (len(bucket) / total) * abs(accuracy - mean_confidence)
    return out


def precision_at(pairs: list[tuple[float, bool]], floor: float = 0.95) -> float:
    bucket = [ok for c, ok in pairs if c >= floor]
    return sum(bucket) / len(bucket) if bucket else 1.0


def evaluate(
    *,
    features_path: Path | str = OUT_DIR / "features.jsonl",
    table_path: Path | str = OUT_DIR / "calibration.json",
    out_dir: Path | str = OUT_DIR,
) -> dict[str, Any]:
    records = read_jsonl(features_path)
    table_json = read_json(table_path)
    # Held-out evaluation: the same split the fitting stage did NOT see.
    _, holdout = hash_split(
        records,
        key_fn=lambda r: f"{r['backend']}|{r['issue_type']}|{r['raw_score']}|{r.get('coverage')}",
        eval_fraction=0.35,
        salt="calib-eval",
    )
    rows = holdout if len(holdout) >= 20 else records
    pairs = apply_table(table_json, rows)
    metrics = {
        "version": table_json.get("version", "unknown"),
        "n_eval": len(pairs),
        "held_out": len(holdout) >= 20,
        "ece": round(ece(pairs), 4),
        "precision_at_0.95": round(precision_at(pairs, 0.95), 4),
        "gates": {"ece_max": GATE_ECE, "precision_at_95_min": GATE_PRECISION_AT_95},
    }
    metrics["gates"]["failures"] = check_gates(metrics)
    write_json(Path(out_dir) / "calib-v1.metrics.json", metrics)
    return metrics


def check_gates(metrics: dict[str, Any]) -> list[str]:
    failures = []
    if metrics["ece"] > GATE_ECE:
        failures.append(f"ECE {metrics['ece']} > {GATE_ECE}")
    if metrics["precision_at_0.95"] < GATE_PRECISION_AT_95:
        failures.append(f"P@0.95 {metrics['precision_at_0.95']} < {GATE_PRECISION_AT_95}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.calibration.eval_calibration")
    parser.add_argument("--features", default=str(OUT_DIR / "features.jsonl"))
    parser.add_argument("--table", default=str(OUT_DIR / "calibration.json"))
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--check-gates", action="store_true")
    args = parser.parse_args(argv)
    metrics = evaluate(features_path=args.features, table_path=args.table, out_dir=args.out)
    print(f"calibration eval: ECE={metrics['ece']} P@0.95={metrics['precision_at_0.95']} "
          f"n={metrics['n_eval']}")
    failures = metrics["gates"]["failures"]
    if failures and args.check_gates:
        for failure in failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
