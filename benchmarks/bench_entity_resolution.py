"""Benchmark for freshdata probabilistic entity resolution.

Generates a synthetic dataset with *known* duplicate labels, runs the
rule-weighted linkage pipeline, and reports the metrics that matter for a
record-linkage system:

* **candidate-pair reduction ratio** — how much blocking shrinks the O(n^2)
  comparison space (1.0 means everything was pruned, 0.0 means full cartesian).
* **runtime** — wall-clock for the resolve call.
* **precision / recall / F1** — measured against the ground-truth duplicate
  pairs (available because the synthetic generator tags each row's true entity).

It runs both the ``pandas`` and ``duckdb`` backends so they can be compared
side by side (DuckDB is skipped automatically if not installed).

> **Splink-class linkage target; current implementation is rule-weighted, not
> EM-trained unless calibration is enabled.**

    python benchmarks/bench_entity_resolution.py --rows 5000 --dup-rate 0.3 --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

import freshdata as fd
from freshdata.enterprise import BlockingRule, ComparisonLevel, EntityResolutionConfig

_FIRST = ["alice", "bob", "carol", "dan", "erin", "frank", "grace", "heidi"]
_LAST = ["smith", "jones", "lee", "patel", "kim", "garcia", "khan", "obrien"]


def synth(rows: int, dup_rate: float, *, seed: int = 0) -> pd.DataFrame:
    """Build *rows* records; a ``dup_rate`` fraction are noisy copies of earlier
    records. ``entity_label`` is the ground-truth identity.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    labels: list[int] = []
    next_label = 0
    for i in range(rows):
        make_dup = records and rng.random() < dup_rate
        if make_dup:
            src_idx = int(rng.integers(0, len(records)))
            base = records[src_idx]
            label = labels[src_idx]
            email = base["email"]
            name = base["name"]
            if rng.random() < 0.4:  # perturb the name slightly
                name = name + "e" if not name.endswith("e") else name[:-1]
            if rng.random() < 0.3:  # perturb the email casing
                email = email.upper()
            rec = {
                "name": name,
                "email": email,
                "dob": base["dob"],
                "phone": base["phone"],
            }
        else:
            label = next_label
            next_label += 1
            fn = rng.choice(_FIRST)
            ln = rng.choice(_LAST)
            rec = {
                "name": f"{fn} {ln}",
                "email": f"{fn}.{ln}{label}@mail.test",
                "dob": f"19{rng.integers(60, 99):02d}-{rng.integers(1, 13):02d}-"
                       f"{rng.integers(1, 28):02d}",
                "phone": f"555{rng.integers(1000000, 9999999)}",
            }
        rec["id"] = i
        rec["entity_label"] = label
        records.append(rec)
        labels.append(label)
    df = pd.DataFrame(records)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _config(backend: str) -> EntityResolutionConfig:
    return EntityResolutionConfig(
        enabled=True,
        backend=backend,  # type: ignore[arg-type]
        unique_id_column="id",
        blocking_rules=(
            BlockingRule("lower(l.email) = lower(r.email)", "same email"),
            BlockingRule(
                "l.dob = r.dob and substr(lower(l.name), 1, 3) = "
                "substr(lower(r.name), 1, 3)",
                "same dob + name prefix",
            ),
        ),
        comparisons=(
            ComparisonLevel("name", "jaro_winkler", threshold=0.85, weight=2.0),
            ComparisonLevel("email", "jaro_winkler", threshold=0.90, weight=3.0),
            ComparisonLevel("dob", "exact", weight=2.0),
            ComparisonLevel("phone", "levenshtein", threshold=0.85, weight=1.0),
        ),
        match_threshold=0.82,
        clerical_review_threshold=0.6,
    )


def truth_pairs(df: pd.DataFrame) -> set[tuple]:
    """All ground-truth duplicate id-pairs (order-independent)."""
    out: set[tuple] = set()
    for _, group in df.groupby("entity_label"):
        ids = sorted(group["id"].tolist())
        for a, b in combinations(ids, 2):
            out.add((a, b))
    return out


def _norm(a, b) -> tuple:
    return (a, b) if a <= b else (b, a)


def evaluate(df: pd.DataFrame, backend: str) -> dict:
    config = _config(backend)
    t0 = time.perf_counter()
    _, report = fd.resolve_entities(df, config=config)
    runtime = time.perf_counter() - t0

    n = len(df)
    total_pairs = n * (n - 1) // 2
    reduction = 1.0 - (report.n_candidate_pairs / total_pairs) if total_pairs else 0.0

    predicted = {_norm(p.left_id, p.right_id) for p in report.pairs if p.decision == "match"}
    truth = truth_pairs(df)
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "backend": backend,
        "n_records": n,
        "total_pairs": total_pairs,
        "candidate_pairs": report.n_candidate_pairs,
        "reduction_ratio": round(reduction, 6),
        "runtime_s": round(runtime, 4),
        "matches": report.n_matches,
        "clusters": report.n_clusters,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--dup-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["pandas", "duckdb", "both"], default="both")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    df = synth(args.rows, args.dup_rate, seed=args.seed)
    backends = ["pandas", "duckdb"] if args.backend == "both" else [args.backend]

    results = []
    for backend in backends:
        if backend == "duckdb" and importlib.util.find_spec("duckdb") is None:
            print("duckdb not installed — skipping duckdb backend", file=sys.stderr)
            continue
        results.append(evaluate(df, backend))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            print(
                f"[{r['backend']:>6}] n={r['n_records']} "
                f"pairs {r['candidate_pairs']}/{r['total_pairs']} "
                f"(reduction {r['reduction_ratio']:.4f}) "
                f"runtime {r['runtime_s']}s | "
                f"P={r['precision']} R={r['recall']} F1={r['f1']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
