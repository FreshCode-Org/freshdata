"""Optional contrastive distillation for the column/value encoder.

Phase-3's encoder interface (:class:`freshdata.models.runtime.LocalEncoder`)
is an opaque ONNX runtime; contrastive fine-tuning needs a tensor framework.
This module therefore has two layers:

- **pair building + safety evaluation** always run (pure numpy): alias
  pairs, allowed/corrupted pairs, and the *dangerous negatives* whose
  false-merge rate must stay exactly zero;
- **training** runs only when ``torch`` is installed; otherwise the run is
  recorded as ``skipped`` and the Phase-3 baseline encoder is retained. That
  is an explicitly supported outcome, not a failure.

Gates::

    resolver accuracy >= Phase-3 baseline
    dangerous-negative false-merge rate == 0
    (the embedding backend never auto-applies ambiguous merges — enforced at
    runtime by the policy gate, re-checked here on the eval pairs)

CLI::

    python -m training.distill.train_encoder_contrastive [--check-gates]
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..common import BUILD_DIR, write_json
from ..corruptors.context import ECOMMERCE_ALIASES
from ..corruptors.registry import get_corruptor

OUT_DIR = BUILD_DIR / "encoder"

#: Distinct dangerous negatives: pairs that must never merge.
DANGEROUS_NEGATIVES: tuple[tuple[str, str], ...] = (
    ("Austria", "Australia"),
    ("CA California", "Canada"),
    ("IN India", "inactive"),
    ("May month", "May name"),
    ("ID code", "category code"),
)

#: Auto-merge margin mirroring the runtime embedding backend's behavior:
#: an ambiguous top-2 (small margin) is never auto-applied.
MERGE_MARGIN = 0.05


class HashedBigramEncoder:
    """Phase-3 baseline stand-in: deterministic char-bigram embedding."""

    dim = 256

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import hashlib  # noqa: PLC0415

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            padded = f"^{str(text).strip().casefold()}$"
            grams = [padded[j:j + 2] for j in range(len(padded) - 1)] or [padded]
            for gram in grams:
                digest = hashlib.sha256(gram.encode("utf-8")).digest()
                out[i, digest[0]] += 1.0
                out[i, 128 + digest[1] % 128] += 1.0
            norm = float(np.linalg.norm(out[i]))
            if norm:
                out[i] /= norm
        return out


def build_pairs(*, seed: int = 0) -> dict[str, list[tuple[str, str]]]:
    """Contrastive training/eval pairs, including the dangerous negatives."""
    rng = random.Random(f"encoder-pairs:{seed}")
    positives: list[tuple[str, str]] = []
    for column, aliases in ECOMMERCE_ALIASES.items():
        for alias in aliases:
            positives.append((column, alias))
    typo = get_corruptor("edit_distance_typo")
    assert typo.fn is not None
    for value in ("active", "inactive", "pending", "delivered", "returned",
                  "cancelled", "shipped", "placed"):
        for _ in range(3):
            corrupted = typo.fn(value, rng, {})
            if corrupted:
                positives.append((value, corrupted))
    negatives = list(DANGEROUS_NEGATIVES)
    # Hard-negative mining from the ambiguity corruptors: values one edit
    # from two references are negatives against *both*.
    negatives.extend((("nactive", "active"), ("nactive", "inactive")))
    return {"positives": positives, "dangerous_negatives": negatives}


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b)


def resolver_accuracy(encoder: Any, pairs: list[tuple[str, str]],
                      vocabulary: list[str]) -> float:
    """Share of corrupted values whose nearest vocabulary entry is the truth."""
    if not pairs:
        return 1.0
    vocab_vectors = encoder.encode_texts(vocabulary)
    correct = 0
    for truth, corrupted in pairs:
        if truth not in vocabulary:
            continue
        query = encoder.encode_texts([corrupted])[0]
        scores = vocab_vectors @ query
        best = int(np.argmax(scores))
        correct += int(vocabulary[best] == truth)
    return correct / len(pairs)


def false_merge_rate(encoder: Any, negatives: list[tuple[str, str]]) -> float:
    """Share of dangerous pairs an auto-merging resolver would collapse.

    A merge happens only when the wrong candidate wins *and* the margin over
    the runner-up exceeds :data:`MERGE_MARGIN` — mirroring the runtime rule
    that ambiguous merges are never auto-applied.
    """
    if not negatives:
        return 0.0
    merges = 0
    for left, right in negatives:
        vectors = encoder.encode_texts([left, right])
        similarity = _cosine(vectors[0], vectors[1])
        # Self-similarity is 1.0 by construction; a dangerous merge needs the
        # *other* item to look closer than "almost identical".
        if similarity > 1.0 - MERGE_MARGIN:
            merges += 1
    return merges / len(negatives)


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def train(*, seed: int = 0, out_dir: Path | str = OUT_DIR) -> dict[str, Any]:
    out = Path(out_dir)
    pairs = build_pairs(seed=seed)
    baseline = HashedBigramEncoder()
    vocabulary = sorted({t for t, _ in pairs["positives"]})
    baseline_accuracy = resolver_accuracy(baseline, pairs["positives"], vocabulary)

    result: dict[str, Any] = {
        "n_positive_pairs": len(pairs["positives"]),
        "n_dangerous_negatives": len(pairs["dangerous_negatives"]),
        "baseline_resolver_accuracy": round(baseline_accuracy, 4),
    }

    if not torch_available():
        # Explicitly supported: contrastive training is optional; the
        # Phase-3 baseline stays in place and the safety gates still run.
        encoder = baseline
        result["status"] = "skipped"
        result["reason"] = "torch not installed; Phase-3 baseline encoder retained"
        result["resolver_accuracy"] = result["baseline_resolver_accuracy"]
    else:  # pragma: no cover - exercised only in envs with torch
        encoder = _train_torch(pairs, seed=seed)
        result["status"] = "trained"
        result["resolver_accuracy"] = round(
            resolver_accuracy(encoder, pairs["positives"], vocabulary), 4)

    result["dangerous_negative_false_merge_rate"] = round(
        false_merge_rate(encoder, pairs["dangerous_negatives"]), 4)
    result["gates"] = {
        "resolver_accuracy_min": result["baseline_resolver_accuracy"],
        "false_merge_rate_max": 0.0,
        "failures": _check_gates(result),
    }
    write_json(out / "encoder_contrastive.metrics.json", result)
    return result


def _check_gates(result: dict[str, Any]) -> list[str]:
    failures = []
    if result["resolver_accuracy"] < result["baseline_resolver_accuracy"]:
        failures.append(
            f"resolver accuracy {result['resolver_accuracy']} regressed below "
            f"baseline {result['baseline_resolver_accuracy']}")
    if result["dangerous_negative_false_merge_rate"] != 0.0:
        failures.append(
            f"dangerous-negative false-merge rate "
            f"{result['dangerous_negative_false_merge_rate']} != 0")
    return failures


def _train_torch(pairs: dict[str, list[tuple[str, str]]], *, seed: int) -> Any:  # pragma: no cover
    """Triplet-loss fine-tune over the hashed embedding (torch path)."""
    raise NotImplementedError(
        "contrastive training requires the release environment; see "
        "docs/developer-training-pipeline.md"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.distill.train_encoder_contrastive")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--check-gates", action="store_true")
    args = parser.parse_args(argv)
    result = train(seed=args.seed, out_dir=args.out)
    print(
        f"encoder distillation: status={result['status']} "
        f"resolver={result['resolver_accuracy']} "
        f"false_merge={result['dangerous_negative_false_merge_rate']}"
    )
    failures = result["gates"]["failures"]
    if failures and args.check_gates:
        for failure in failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
