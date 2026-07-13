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


# =====================================================================
# Labelled-dataset mode (benchmarks/data/er_labelled_*.csv)
# =====================================================================


def _labelled_config(null_policy: str, mode: str) -> EntityResolutionConfig:
    """Linkage config for the labelled dataset schema (see gen_er_labelled.py)."""
    return EntityResolutionConfig(
        enabled=True,
        backend="pandas",
        unique_id_column="id",
        blocking_rules=(
            BlockingRule("lower(l.email) = lower(r.email)", "same email"),
            BlockingRule(
                "l.dob = r.dob and substr(lower(l.last_name), 1, 2) = "
                "substr(lower(r.last_name), 1, 2)",
                "same dob + surname prefix",
            ),
            BlockingRule("l.phone = r.phone", "same phone"),
        ),
        comparisons=(
            ComparisonLevel("full_name", "token_set", weight=2.0),
            ComparisonLevel("full_name", "jaro_winkler", threshold=0.85, weight=1.5),
            ComparisonLevel("last_name", "metaphone", weight=1.0),
            ComparisonLevel("email", "jaro_winkler", threshold=0.90, weight=2.5),
            ComparisonLevel("dob", "exact", weight=2.0),
            ComparisonLevel("first_name", "jaro_winkler", threshold=0.85, weight=1.0),
        ),
        null_policy=null_policy,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
    )


def _false_merge_count(report, df: pd.DataFrame) -> int:
    """Clusters that merged more than one true entity (the costly failure)."""
    label_of = dict(zip(df["id"], df["entity_id"]))
    return sum(
        1
        for c in report.clusters
        if len({label_of[rid] for rid in c.record_ids if rid in label_of}) > 1
    )


def _peak_rss_mb() -> float | None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1 << 20) if sys.platform == "darwin" else peak / 1024, 1)
    except ImportError:  # pragma: no cover
        return None


def evaluate_labelled(df: pd.DataFrame, null_policy: str, mode: str) -> dict:
    config = _labelled_config(null_policy, mode)
    t0 = time.perf_counter()
    _, report = fd.resolve_entities(df, config=config)
    runtime = time.perf_counter() - t0

    n = len(df)
    total_pairs = n * (n - 1) // 2
    predicted = {_norm(p.left_id, p.right_id) for p in report.pairs if p.decision == "match"}
    truth = truth_pairs(df.rename(columns={"entity_id": "entity_label"}))
    tp, fp, fn = (
        len(predicted & truth),
        len(predicted - truth),
        len(truth - predicted),
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "config": f"null_policy={null_policy}, mode={mode}",
        "null_policy": null_policy,
        "mode": mode,
        "n_records": n,
        "candidate_pairs": report.n_candidate_pairs,
        "reduction_ratio": round(1.0 - report.n_candidate_pairs / total_pairs, 6)
        if total_pairs
        else 0.0,
        "true_pairs": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merged_clusters": _false_merge_count(report, df),
        "n_clusters": report.n_clusters,
        "runtime_s": round(runtime, 4),
        "peak_rss_mb": _peak_rss_mb(),
    }


def evaluate_exact_baseline(df: pd.DataFrame) -> dict:
    """pandas drop_duplicates on identity fields — the honesty baseline."""
    fields = ["full_name", "email", "phone", "dob"]
    t0 = time.perf_counter()
    dup_mask = df.duplicated(subset=fields, keep=False)
    runtime = time.perf_counter() - t0
    predicted: set[tuple] = set()
    for _, group in df[dup_mask].groupby(fields, dropna=False):
        ids = sorted(group["id"].tolist())
        for a, b in combinations(ids, 2):
            predicted.add((a, b))
    truth = truth_pairs(df.rename(columns={"entity_id": "entity_label"}))
    tp, fp, fn = (
        len(predicted & truth),
        len(predicted - truth),
        len(truth - predicted),
    )
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "config": "pandas exact drop_duplicates (baseline)",
        "n_records": len(df),
        "true_pairs": len(truth),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merged_clusters": 0,
        "runtime_s": round(runtime, 4),
    }


_SWEEP = [
    ("penalize", "balanced"),
    ("neutral", "balanced"),
    ("neutral", "precision"),
    ("neutral", "recall"),
]


def run_labelled(path: str, write_results: str | None) -> list[dict]:
    df = pd.read_csv(path)
    results = [evaluate_exact_baseline(df)]
    results += [evaluate_labelled(df, np_, m) for np_, m in _SWEEP]
    if write_results:
        import platform

        payload = {
            "dataset": path,
            "generator": "benchmarks/gen_er_labelled.py",
            "machine": f"{platform.system()} {platform.machine()}",
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "freshdata": fd.__version__,
            "method": "rule-weighted linkage (normalized weighted average; "
            "no EM / Fellegi-Sunter)",
            "results": results,
        }
        with open(write_results, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--dup-rate", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["pandas", "duckdb", "both"], default="both")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--labelled",
        metavar="CSV",
        help="run the labelled-dataset accuracy sweep (see gen_er_labelled.py)",
    )
    ap.add_argument(
        "--write-results", metavar="JSON", help="with --labelled: write results here"
    )
    args = ap.parse_args(argv)

    if args.labelled:
        results = run_labelled(args.labelled, args.write_results)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(
                    f"[{r['config']:<44}] P={r['precision']:.4f} "
                    f"R={r['recall']:.4f} F1={r['f1']:.4f} "
                    f"FP={r['fp']} FN={r['fn']} "
                    f"false-merges={r['false_merged_clusters']}"
                )
        return 0

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
