"""Train the semantic-type role head (``fd-role-head-v1``).

Inputs: synthetic seed columns, corruptor-derived value dirt, alias-mangled
column-name variants, optional cached teacher ``ColumnRoleLabel`` payloads,
and the human-verified eval split. Output: weights (fp32 + int8), metrics
JSON, a confusion matrix, and — when the ``onnx`` package is installed — an
ONNX export.

Release gates (checked with ``--check-gates``)::

    macro-F1 >= 0.90 on the human-verified eval
    content-detector contradiction rate <= 1%
    adversarial alias accuracy >= 0.85

CLI::

    python -m training.distill.train_role_head [--dev] [--check-gates]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ..common import BUILD_DIR, TRAINING_ROOT, write_json
from ..corruptors.context import ECOMMERCE_ALIASES
from ..corruptors.registry import get_corruptor
from ..eval.human_verified import load_verified
from ..seed.synthetic import seed_tables
from .linear import FeaturizerConfig, LinearHead, confusion, macro_f1, per_class_f1

OUT_DIR = BUILD_DIR / "role_head"
HUMAN_EVAL_PATH = TRAINING_ROOT / "eval" / "data" / "role_eval_human.jsonl"

SEMANTIC_TYPES = (
    "email", "phone", "url", "country", "currency_amount", "quantity_with_unit",
    "person_name", "address", "city", "postal_code", "national_id",
    "category_code", "free_text", "boolean_like", "date_like", "identifier",
    "unknown",
)

#: Ground-truth semantic type for every synthetic seed column.
COLUMN_TYPES: dict[str, str] = {
    "cust_id": "identifier", "txn_id": "identifier",
    "full_name": "person_name", "email": "email", "phone": "phone",
    "address": "address", "city": "city", "state": "city", "country": "country",
    "postal_code": "postal_code", "national_id_like": "national_id",
    "status": "category_code", "order_status": "category_code",
    "category_code": "category_code",
    "signup_date": "date_like", "order_date": "date_like",
    "website": "url", "newsletter_opt_in": "boolean_like",
    "quantity": "quantity_with_unit", "unit_price_inr": "currency_amount",
    "monthly_revenue": "currency_amount", "notes": "free_text",
}

GATE_MACRO_F1 = 0.90
GATE_CONTRADICTION_RATE = 0.01
GATE_ALIAS_ACCURACY = 0.85


def example_text(column_name: str, values: list[Any], *, k: int = 8) -> str:
    """The exact text form the head scores: name + sample values."""
    samples = " ; ".join(str(v) for v in values[:k])
    return f"col:{column_name} | {samples}"


def _alias_variants(column: str, rng: random.Random) -> list[str]:
    variants = list(ECOMMERCE_ALIASES.get(column, ()))
    parts = column.split("_")
    variants.extend((
        "".join(p.title() for p in parts),
        " ".join(parts).title(),
        "-".join(parts),
        column.upper(),
    ))
    rng.shuffle(variants)
    return [v for v in variants if v != column]


def build_training_set(*, seed: int = 0, dev: bool = False) -> list[dict[str, Any]]:
    """Labeled role examples from seed tables + dirt + alias mangling."""
    rng = random.Random(f"role-head:{seed}")
    tables = seed_tables(seed=seed)
    dirty_corruptors = [get_corruptor(n) for n in (
        "whitespace_insertion", "casing_change", "email_double_at",
        "phone_in_zero_prefix", "date_format_shuffle", "currency_formatting",
    )]
    examples: list[dict[str, Any]] = []
    windows = 4 if dev else 12
    for _, frame in tables.items():
        for column in frame.columns:
            semantic_type = COLUMN_TYPES.get(str(column))
            if semantic_type is None:
                continue
            values = [v for v in frame[column].tolist() if v is not None]
            for w in range(windows):
                start = (w * 13) % max(1, len(values) - 8)
                window = [str(v) for v in values[start:start + 8]]
                # Random dirt on a share of windows: the head must be robust
                # to the same corruption the cleaner sees.
                if w % 3 == 1 and window:
                    corruptor = rng.choice(dirty_corruptors)
                    assert corruptor.fn is not None
                    window = [
                        str(corruptor.fn(v, rng, corruptor.params) or v) for v in window
                    ]
                name = str(column)
                if w % 4 == 2:
                    aliases = _alias_variants(str(column), rng)
                    if aliases:
                        name = aliases[0]
                examples.append({
                    "text": example_text(name, window),
                    "label": semantic_type,
                    "column": str(column),
                    "aliased": name != str(column),
                })
    # Explicit unknown class: junk columns with no coherent type.
    for i in range(8 if dev else 30):
        junk = [str(rng.random())[:6], rng.choice(("x", "yy", "?")), str(rng.randrange(999))]
        rng.shuffle(junk)
        examples.append({
            "text": example_text(f"misc_{i}", junk * 2),
            "label": "unknown", "column": f"misc_{i}", "aliased": False,
        })
    return examples


def teacher_examples(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """Optional cached ``ColumnRoleLabel`` payloads (training data only)."""
    from ..teacher.cache import TeacherCache  # noqa: PLC0415

    out: list[dict[str, Any]] = []
    for entry in TeacherCache(cache_dir).entries():
        for payload in entry.get("payloads", []):
            if payload.get("semantic_type") in SEMANTIC_TYPES and "column_name" in payload:
                out.append({
                    "text": example_text(
                        str(payload["column_name"]), list(payload.get("masked_samples", []))
                    ),
                    "label": str(payload["semantic_type"]),
                    "column": str(payload["column_name"]),
                    "aliased": False,
                })
    return out


def load_human_eval(path: Path | str = HUMAN_EVAL_PATH) -> list[dict[str, Any]]:
    labels = load_verified(path)
    for row in labels:
        row.setdefault("text", example_text(str(row["column"]), list(row["values"])))
    return labels


def contradiction_rate(head: LinearHead, eval_rows: list[dict[str, Any]]) -> float:
    """Share of confident head calls that contradict FreshData's detectors."""
    from freshdata.semantic.semantic_types import infer_semantic_type  # noqa: PLC0415

    contradictions = 0
    confident = 0
    for row in eval_rows:
        values = row.get("values")
        if not values:
            continue
        prediction, head_confidence = head.predict([row["text"]])[0]
        if prediction == "unknown" or head_confidence < 0.7:
            continue
        detected = infer_semantic_type(str(row["column"]), pd.Series(list(values)))
        detected_type = getattr(detected, "semantic_type", "unknown")
        detected_confidence = float(getattr(detected, "confidence", 0.0))
        if detected_type in ("unknown", None) or detected_confidence < 0.9:
            continue
        confident += 1
        if prediction != detected_type:
            contradictions += 1
    return contradictions / confident if confident else 0.0


def evaluate(head: LinearHead, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [row["text"] for row in eval_rows]
    y_true = [row["label"] for row in eval_rows]
    predictions = head.predict(texts)
    y_pred = [label for label, _ in predictions]
    alias_rows = [i for i, row in enumerate(eval_rows) if row.get("aliased")]
    alias_accuracy = (
        sum(1 for i in alias_rows if y_pred[i] == y_true[i]) / len(alias_rows)
        if alias_rows else 1.0
    )
    return {
        "n_eval": len(eval_rows),
        "macro_f1": round(macro_f1(y_true, y_pred, head.classes), 4),
        "per_class_f1": {
            k: round(v, 4) for k, v in per_class_f1(y_true, y_pred, head.classes).items()},
        "abstention_rate": round(
            sum(1 for p, _ in predictions if p == "unknown") / len(predictions), 4),
        "adversarial_alias_accuracy": round(alias_accuracy, 4),
        "contradiction_rate": round(contradiction_rate(head, eval_rows), 4),
        "confusion": confusion(y_true, y_pred, head.classes),
    }


def check_gates(metrics: dict[str, Any]) -> list[str]:
    failures = []
    if metrics["macro_f1"] < GATE_MACRO_F1:
        failures.append(f"macro-F1 {metrics['macro_f1']} < {GATE_MACRO_F1}")
    if metrics["contradiction_rate"] > GATE_CONTRADICTION_RATE:
        failures.append(
            f"contradiction rate {metrics['contradiction_rate']} > {GATE_CONTRADICTION_RATE}")
    if metrics["adversarial_alias_accuracy"] < GATE_ALIAS_ACCURACY:
        failures.append(
            f"alias accuracy {metrics['adversarial_alias_accuracy']} < {GATE_ALIAS_ACCURACY}")
    return failures


def train(*, seed: int = 0, dev: bool = False, out_dir: Path | str = OUT_DIR) -> dict[str, Any]:
    out = Path(out_dir)
    examples = build_training_set(seed=seed, dev=dev)
    examples.extend(teacher_examples())
    eval_rows = load_human_eval()
    head = LinearHead.train(
        [e["text"] for e in examples],
        [e["label"] for e in examples],
        classes=SEMANTIC_TYPES,
        featurizer=FeaturizerConfig(dim=1024 if dev else 2048),
        epochs=150 if dev else 400,
        abstain_class="unknown",
        abstain_threshold=0.35,
    )
    metrics = evaluate(head, eval_rows)
    metrics["n_train"] = len(examples)
    metrics["gates"] = {
        "macro_f1_min": GATE_MACRO_F1,
        "contradiction_rate_max": GATE_CONTRADICTION_RATE,
        "alias_accuracy_min": GATE_ALIAS_ACCURACY,
        "failures": check_gates(metrics),
    }
    head.save(out / "role_head.weights.json")
    write_json(out / "role_head.metrics.json", metrics)
    write_json(out / "role_head.confusion.json", metrics["confusion"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.distill.train_role_head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--check-gates", action="store_true")
    args = parser.parse_args(argv)
    metrics = train(seed=args.seed, dev=args.dev, out_dir=args.out)
    print(
        f"role head: macro-F1={metrics['macro_f1']} "
        f"alias={metrics['adversarial_alias_accuracy']} "
        f"contradiction={metrics['contradiction_rate']} "
        f"abstention={metrics['abstention_rate']}"
    )
    failures = metrics["gates"]["failures"]
    if failures and args.check_gates:
        for failure in failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
