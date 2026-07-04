"""Extract calibration features from CleanBench runs.

Each record is one semantic proposal with the deterministic feature set the
architecture specifies, plus its machine-verified outcome (``correct``)
derived from the fixture's ground truth — never from teacher output.

CLI::

    python -m training.calibration.build_features [--seeds 5]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..common import BUILD_DIR, REPO_ROOT, write_jsonl

OUT_DIR = BUILD_DIR / "calibration"

_OUTCOME_STATUSES = frozenset({"automatic", "suggested", "approved", "accepted"})


def _cleanbench():
    bench_dir = str(REPO_ROOT / "benchmarks")
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    import cleanbench  # noqa: PLC0415

    return cleanbench


def _canon(value: object) -> object:
    from cleanbench.metrics import _canon as canon  # noqa: PLC0415

    return canon(value)


def action_features(action: Any, corrupted: pd.DataFrame, semantic_mode: str) -> dict[str, Any]:
    metadata = getattr(action, "metadata", {}) or {}
    evidence = metadata.get("evidence", []) or []
    kinds = {e.get("kind") for e in evidence if isinstance(e, dict)}
    margin = None
    semantic_type_confidence = None
    for e in evidence:
        detail = str(e.get("detail", "")) if isinstance(e, dict) else ""
        if e.get("kind") == "embedding" and "margin=" in detail:
            try:
                margin = float(detail.split("margin=")[1].split()[0])
            except (IndexError, ValueError):
                margin = None
        if e.get("kind") == "semantic_type" and "confidence=" in detail:
            try:
                semantic_type_confidence = float(detail.split("confidence=")[1].split()[0])
            except (IndexError, ValueError):
                semantic_type_confidence = None
    column = getattr(action, "column", None)
    column_signature = metadata.get("column_signature", {}) or {}
    calibration = metadata.get("calibration", {}) or {}
    raw_score = calibration.get("raw", getattr(action, "confidence", 0.0))
    series = corrupted[column] if column in corrupted.columns else pd.Series(dtype=object)
    return {
        "raw_score": float(raw_score),
        "backend": str(metadata.get("backend", "deterministic")),
        "issue_type": str(metadata.get("issue_type", "unknown")),
        "risk": str(getattr(action, "risk", metadata.get("risk", "unknown"))),
        "role_confidence": 1.0 if column_signature.get("role") else 0.0,
        "semantic_type_confidence": semantic_type_confidence,
        "distinct_support": int(series.nunique()) if len(series) else None,
        "coverage": None,
        "memory_support_count": sum(
            1 for e in evidence if isinstance(e, dict) and e.get("kind") == "memory_replay"),
        "profile_support_count": sum(
            1 for e in evidence
            if isinstance(e, dict) and str(e.get("kind", "")).startswith("profile")),
        "learned_precision": metadata.get("learned_precision"),
        "policy_rule_present": bool(column_signature.get("policy_rule")),
        "allowed_values_present": "reference" in str(metadata.get("expert", ""))
                                  or metadata.get("issue_type") == "reference_value",
        "margin_to_second_candidate": margin,
        "semantic_mode": semantic_mode,
        "evidence_kinds": sorted(k for k in kinds if k),
    }


def extract_from_run(
    report: Any, truth: pd.DataFrame, corrupted: pd.DataFrame, *, semantic_mode: str = "auto"
) -> list[dict[str, Any]]:
    """(features, correct) records for every scoreable semantic action."""
    records: list[dict[str, Any]] = []
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
        matches = [pos for pos, value in enumerate(corrupted[column].tolist())
                   if _canon(value) == raw_canon]
        if not matches:
            continue
        truth_values = truth[column].tolist()
        proposed_canon = _canon(proposed)
        correct = all(_canon(truth_values[pos]) == proposed_canon for pos in matches)
        features = action_features(action, corrupted, semantic_mode)
        features["coverage"] = round(len(matches) / max(1, len(corrupted)), 6)
        features["correct"] = bool(correct)
        features["confidence_reported"] = float(getattr(action, "confidence", 0.0))
        records.append(features)
    return records


def build_features(*, seeds: int = 5, out_dir: Path | str = OUT_DIR) -> list[dict[str, Any]]:
    import freshdata as fd  # noqa: PLC0415

    cleanbench = _cleanbench()
    from freshdata.models import runtime as model_runtime  # noqa: PLC0415

    records: list[dict[str, Any]] = []
    for seed in range(seeds):
        for maker in (cleanbench.make_t2_semantic_fixture, cleanbench.make_t3_context_fixture):
            truth, corrupted, kwargs = maker(seed=seed * 10)
            _, report = fd.clean(corrupted, return_report=True, **kwargs)
            records.extend(extract_from_run(report, truth, corrupted))
        # Embedding-backend runs (stub encoder): the interesting curves.
        truth, corrupted, kwargs = cleanbench.make_phase3_embedding_fixture(seed=seed * 10 + 5)
        model_runtime.set_encoder_factory(lambda _mid: cleanbench.BigramStubEncoder())
        try:
            _, report = fd.clean(corrupted, return_report=True, **kwargs)
        finally:
            model_runtime.set_encoder_factory(None)
        records.extend(extract_from_run(report, truth, corrupted))
    write_jsonl(Path(out_dir) / "features.jsonl", records)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.calibration.build_features")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)
    records = build_features(seeds=args.seeds, out_dir=args.out)
    backends = sorted({r["backend"] for r in records})
    print(f"calibration features: {len(records)} records, backends={backends}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
