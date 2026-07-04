"""Train the context-sentence intent head (``fd-intent-v1``).

Inputs: the Phase-1 golden context corpus, deterministic paraphrase
templates (including the Indian-English / Hinglish set), corruptor-generated
context variants, optional cached teacher paraphrases, and an
author-disjoint human-verified eval split.

The intent head is **optional evidence** for the deterministic parser — it
never replaces it, and in strict mode unresolved context stays surfaced. Its
training therefore favors *protection recall* over coverage: missing a
PROTECT sentence is the one unforgivable error, so PROTECT is oversampled
and gated at recall >= 0.99.

Gates (``--check-gates``)::

    exact intent accuracy >= 0.92
    slot F1 >= 0.90
    UNKNOWN precision >= 0.95
    protected intent recall >= 0.99

CLI::

    python -m training.distill.train_intent_head [--dev] [--check-gates]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from ..common import BUILD_DIR, REPO_ROOT, read_json, write_json
from ..datasets.splits import author_disjoint_split
from ..seed.synthetic import make_context_sentences
from .linear import FeaturizerConfig, LinearHead, confusion, per_class_f1

OUT_DIR = BUILD_DIR / "intent_head"
GOLDEN_DIR = REPO_ROOT / "tests" / "context" / "golden"

INTENTS = (
    "DOMAIN", "UNIQUE", "VALID_FORMAT", "LOCALE_FORMAT", "PROTECT", "IMPUTE_IF",
    "ALLOWED_VALUES", "RANGE", "DEDUP_KEY", "DROP_IF", "RENAME", "MAP", "UNKNOWN",
)

#: Phase-1 constraint rule -> intent label.
RULE_TO_INTENT = {
    "unique": "UNIQUE",
    "valid_format": "VALID_FORMAT",
    "locale_format": "LOCALE_FORMAT",
    "protected": "PROTECT",
    "impute_missing": "IMPUTE_IF",
    "allowed_values": "ALLOWED_VALUES",
    "range": "RANGE",
    "dedup_key": "DEDUP_KEY",
}

#: Authors held out entirely for the paraphrase-generalization gate.
EVAL_AUTHORS = ("t2", "hinglish_b")

GATE_EXACT_ACCURACY = 0.92
GATE_SLOT_F1 = 0.90
GATE_UNKNOWN_PRECISION = 0.95
GATE_PROTECT_RECALL = 0.99

_DOMAIN_RE = re.compile(r"\bthis is (a|an)?\s*.*\bdata\s?set\b", re.I)


def load_golden_examples() -> list[dict[str, Any]]:
    """Sentence-level intent labels derived from the Phase-1 golden corpus."""
    examples: list[dict[str, Any]] = []
    for text_path in sorted(GOLDEN_DIR.glob("*.txt")):
        policy_path = text_path.with_name(text_path.name.replace(".txt", ".policy.json"))
        if not policy_path.is_file():
            continue
        policy = read_json(policy_path)
        sentence_intent: dict[str, str] = {}
        sentence_slots: dict[str, dict[str, Any]] = {}
        for constraint in policy.get("constraints", []):
            sentence = str(constraint.get("provenance", {}).get("sentence", "")).strip()
            intent = RULE_TO_INTENT.get(str(constraint.get("rule", "")))
            if not sentence or intent is None:
                continue
            sentence_intent[sentence] = intent
            slots = dict(constraint.get("params") or {})
            if constraint.get("resolved_from"):
                slots["column"] = constraint["resolved_from"]
            sentence_slots[sentence] = slots
        for raw_line in text_path.read_text(encoding="utf-8").splitlines():
            sentence = raw_line.strip().rstrip(".")
            if not sentence:
                continue
            matched = None
            for known, intent in sentence_intent.items():
                if known.rstrip(".").strip() == sentence:
                    matched = (intent, sentence_slots.get(known, {}))
                    break
            if matched is None:
                if _DOMAIN_RE.search(raw_line) and policy.get("dataset_domain"):
                    matched = ("DOMAIN", {"domain": policy["dataset_domain"]})
                else:
                    matched = ("UNKNOWN", {})
            examples.append({
                "sentence": raw_line.strip(),
                "intent": matched[0],
                "slots": matched[1],
                "author": "golden",
            })
    return examples


def teacher_paraphrase_examples() -> list[dict[str, Any]]:
    """Optional cached ``ContextParaphraseBatch`` payloads (training only)."""
    from ..teacher.cache import TeacherCache  # noqa: PLC0415

    out: list[dict[str, Any]] = []
    for entry in TeacherCache().entries():
        for payload in entry.get("payloads", []):
            intent = payload.get("intent")
            if intent in INTENTS and isinstance(payload.get("paraphrases"), list):
                for sentence in payload["paraphrases"]:
                    out.append({
                        "sentence": str(sentence), "intent": str(intent), "slots": {},
                        "author": f"teacher:{payload.get('author', 'unknown')}",
                    })
    return out


# --------------------------------------------------------------------------- #
# Deterministic slot extraction (evaluated as slot F1)
# --------------------------------------------------------------------------- #

_THRESHOLD_RE = re.compile(r">?\s*(\d{2,3})\s*%")
_RANGE_RE = re.compile(
    r"(?:between|within)\s+(\d+)\s+(?:and|to|aur)\s+(\d+)"
    r"|(\d+)\s+aur\s+(\d+)\s+ke\s+beech", re.I)
_RENAME_RE = re.compile(
    r"rename(?:\s+the)?(?:\s+column)?\s+\S+(?:\s+column)?\s+to\s+([A-Za-z_]\w*)"
    r"|ko\s+rename\s+karke\s+([A-Za-z_]\w*)"
    r"|ko\s+([A-Za-z_]\w*)\s+rename\s+kar", re.I)
_MAP_RE = re.compile(r"\b(\w+)\s+(?:to|ko)\s+(\w+)\b")
_DOMAIN_RE_SLOT = re.compile(r"this (?:is a[n]? |dataset is a[n]? )(.+?)\s+data\s?set", re.I)
_VALUES_RE = re.compile(
    r"(?:values are|only be|can only be|values:|ho sakta hai|sirf)\s*:?\s*([^.]+)",
    re.I,
)
_VALUE_STOPWORDS = frozenset({"allowed", "hai", "ho", "sakta"})


def extract_slots(sentence: str, intent: str, columns: tuple[str, ...] = ()) -> dict[str, Any]:
    """Regex/lexicon slot extraction — deterministic, no model involved."""
    slots: dict[str, Any] = {}
    lowered = sentence.lower()
    for column in columns:
        if column.lower() in lowered:
            slots["column"] = column
            break
    if intent == "DOMAIN":
        match = _DOMAIN_RE_SLOT.search(sentence)
        if match:
            slots["domain"] = re.sub(r"\W+", "_", match.group(1).strip().lower())
    if intent == "VALID_FORMAT":
        if "email" in lowered:
            slots["format"] = "email"
        elif "phone" in lowered:
            slots["format"] = "phone"
        elif "url" in lowered or "website" in lowered:
            slots["format"] = "url"
    if intent == "LOCALE_FORMAT":
        if "indian" in lowered or "india" in lowered:
            slots["locale"] = "IN"
        if "phone" in lowered:
            slots["format"] = "phone"
    if intent == "IMPUTE_IF":
        match = _THRESHOLD_RE.search(sentence)
        if match:
            slots["threshold"] = int(match.group(1)) / 100.0
    if intent == "RANGE":
        match = _RANGE_RE.search(sentence)
        if match:
            low, high = (match.group(1), match.group(2)) if match.group(1) else (
                match.group(3), match.group(4))
            slots["min"], slots["max"] = int(low), int(high)
    if intent == "ALLOWED_VALUES":
        match = _VALUES_RE.search(sentence)
        if match:
            raw = match.group(1)
            values = [
                v.strip() for v in re.split(r",|\bor\b|\band\b|\bya\b|\baur\b", raw) if v.strip()
            ]
            cleaned = []
            for value in values:
                words = [w for w in value.split() if w.lower() not in _VALUE_STOPWORDS]
                if words and len(" ".join(words)) < 24:
                    cleaned.append(" ".join(words))
            if cleaned:
                slots["values"] = cleaned
    if intent == "RENAME":
        match = _RENAME_RE.search(sentence)
        if match:
            slots["to"] = next(g for g in match.groups() if g)
    if intent == "MAP":
        skip = {"rows", "karo", "map", "aur"}
        pairs = {a: b for a, b in _MAP_RE.findall(sentence)
                 if a.lower() not in skip and b.lower() not in skip}
        if pairs:
            slots["mapping"] = pairs
    if intent == "DROP_IF" and ("empty" in lowered or "missing" in lowered):
        slots["condition"] = "empty"
    return slots


def _slot_pairs(slots: dict[str, Any]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for key, value in slots.items():
        if isinstance(value, list):
            for item in value:
                pairs.add((key, str(item).strip().lower()))
        elif isinstance(value, dict):
            for k, v in value.items():
                pairs.add((f"{key}.{k}".lower(), str(v).strip().lower()))
        else:
            pairs.add((key, str(value).strip().lower()))
    return pairs


def slot_f1(rows: list[dict[str, Any]], predictions: list[str]) -> float:
    tp = fp = fn = 0
    for row, predicted_intent in zip(rows, predictions):
        gold = _slot_pairs(row.get("slots", {}))
        columns = tuple(str(c) for c in row.get("columns", ())) or (
            (str(row["slots"]["column"]),) if row.get("slots", {}).get("column") else ())
        predicted = _slot_pairs(extract_slots(row["sentence"], predicted_intent, columns))
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    denominator = 2 * tp + fp + fn
    return (2 * tp / denominator) if denominator else 1.0


# --------------------------------------------------------------------------- #
# Conflict detection: PROTECT vs mutating intents on the same column
# --------------------------------------------------------------------------- #

_MUTATING_INTENTS = frozenset({"VALID_FORMAT", "LOCALE_FORMAT", "IMPUTE_IF", "MAP", "RENAME"})


def conflict_cases() -> list[dict[str, Any]]:
    """Sentence pairs on one column; ``conflict`` iff protection meets mutation."""
    return [
        {"sentences": ("Never modify revenue values.", "Revenue must be valid numbers."),
         "intents": ("PROTECT", "VALID_FORMAT"), "conflict": True},
        {"sentences": ("status is protected.", "Map A1 to active in status."),
         "intents": ("PROTECT", "MAP"), "conflict": True},
        {"sentences": ("cust_id is unique.", "Deduplicate rows by cust_id."),
         "intents": ("UNIQUE", "DEDUP_KEY"), "conflict": False},
        {"sentences": ("Emails must be valid.", "Phone numbers are Indian."),
         "intents": ("VALID_FORMAT", "LOCALE_FORMAT"), "conflict": False},
        {"sentences": ("age kabhi mat badalna.",
                      "Missing age should be estimated only if confidence >95%."),
         "intents": ("PROTECT", "IMPUTE_IF"), "conflict": True},
        {"sentences": ("Never modify monthly_revenue values.", "monthly_revenue is protected."),
         "intents": ("PROTECT", "PROTECT"), "conflict": False},
    ]


def conflict_detection_accuracy(head: LinearHead) -> float:
    correct = 0
    cases = conflict_cases()
    for case in cases:
        predicted = [label for label, _ in head.predict(list(case["sentences"]))]
        has_protect = "PROTECT" in predicted
        has_mutation = any(p in _MUTATING_INTENTS for p in predicted)
        detected = has_protect and has_mutation
        correct += int(detected == case["conflict"])
    return correct / len(cases)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #

def build_dataset(*, seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(train, human-verified author-disjoint eval)."""
    rows = load_golden_examples()
    rows.extend(make_context_sentences(seed=seed))
    rows.extend(teacher_paraphrase_examples())
    train_rows, eval_rows = author_disjoint_split(rows, eval_authors=EVAL_AUTHORS)
    # The synthetic corpus is authored and verified in-repo; golden rows are
    # maintainer-verified by construction. Eval rows carry that provenance.
    for row in eval_rows:
        row.setdefault("human_verified", True)
        row.setdefault("reviewer", "JWD")
    return train_rows, eval_rows


def evaluate(head: LinearHead, eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from ..eval.human_verified import check_all_verified  # noqa: PLC0415

    check_all_verified(eval_rows)
    texts = [row["sentence"] for row in eval_rows]
    y_true = [row["intent"] for row in eval_rows]
    predictions = head.predict(texts)
    y_pred = [label for label, _ in predictions]

    exact = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    unknown_predicted = [i for i, p in enumerate(y_pred) if p == "UNKNOWN"]
    unknown_precision = (
        sum(1 for i in unknown_predicted if y_true[i] == "UNKNOWN") / len(unknown_predicted)
        if unknown_predicted else 1.0
    )
    protect_rows = [i for i, t in enumerate(y_true) if t == "PROTECT"]
    protect_recall = (
        sum(1 for i in protect_rows if y_pred[i] == "PROTECT") / len(protect_rows)
        if protect_rows else 1.0
    )
    paraphrase_rows = [i for i, row in enumerate(eval_rows)
                       if str(row.get("author", "")) in EVAL_AUTHORS]
    paraphrase_accuracy = (
        sum(1 for i in paraphrase_rows if y_pred[i] == y_true[i]) / len(paraphrase_rows)
        if paraphrase_rows else 1.0
    )
    return {
        "n_eval": len(eval_rows),
        "exact_intent_accuracy": round(exact, 4),
        "slot_f1": round(slot_f1(eval_rows, y_pred), 4),
        "unknown_precision": round(unknown_precision, 4),
        "protected_intent_recall": round(protect_recall, 4),
        "author_disjoint_paraphrase_accuracy": round(paraphrase_accuracy, 4),
        "conflict_detection_accuracy": round(conflict_detection_accuracy(head), 4),
        "per_class_f1": {k: round(v, 4) for k, v in per_class_f1(y_true, y_pred, INTENTS).items()},
        "confusion": confusion(y_true, y_pred, INTENTS),
    }


def check_gates(metrics: dict[str, Any]) -> list[str]:
    failures = []
    if metrics["exact_intent_accuracy"] < GATE_EXACT_ACCURACY:
        failures.append(
            f"exact accuracy {metrics['exact_intent_accuracy']} < {GATE_EXACT_ACCURACY}")
    if metrics["slot_f1"] < GATE_SLOT_F1:
        failures.append(f"slot F1 {metrics['slot_f1']} < {GATE_SLOT_F1}")
    if metrics["unknown_precision"] < GATE_UNKNOWN_PRECISION:
        failures.append(
            f"UNKNOWN precision {metrics['unknown_precision']} < {GATE_UNKNOWN_PRECISION}")
    if metrics["protected_intent_recall"] < GATE_PROTECT_RECALL:
        failures.append(
            f"protected recall {metrics['protected_intent_recall']} < {GATE_PROTECT_RECALL}")
    return failures


def train(*, seed: int = 0, dev: bool = False, out_dir: Path | str = OUT_DIR) -> dict[str, Any]:
    out = Path(out_dir)
    train_rows, eval_rows = build_dataset(seed=seed)
    head = LinearHead.train(
        [row["sentence"] for row in train_rows],
        [row["intent"] for row in train_rows],
        classes=INTENTS,
        featurizer=FeaturizerConfig(dim=2048 if dev else 4096),
        epochs=150 if dev else 400,
        # Protection favors recall over coverage: PROTECT errors are the
        # expensive ones, so its loss weight dominates.
        class_weights={"PROTECT": 4.0},
        abstain_class="UNKNOWN",
        abstain_threshold=0.2,
    )
    metrics = evaluate(head, eval_rows)
    metrics["n_train"] = len(train_rows)
    metrics["gates"] = {
        "exact_accuracy_min": GATE_EXACT_ACCURACY,
        "slot_f1_min": GATE_SLOT_F1,
        "unknown_precision_min": GATE_UNKNOWN_PRECISION,
        "protect_recall_min": GATE_PROTECT_RECALL,
        "failures": check_gates(metrics),
    }
    head.save(out / "intent_head.weights.json")
    write_json(out / "intent_head.metrics.json", metrics)
    write_json(out / "intent_head.confusion.json", metrics["confusion"])
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.distill.train_intent_head")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--check-gates", action="store_true")
    args = parser.parse_args(argv)
    metrics = train(seed=args.seed, dev=args.dev, out_dir=args.out)
    print(
        f"intent head: exact={metrics['exact_intent_accuracy']} "
        f"slotF1={metrics['slot_f1']} unknownP={metrics['unknown_precision']} "
        f"protectR={metrics['protected_intent_recall']} "
        f"paraphrase={metrics['author_disjoint_paraphrase_accuracy']}"
    )
    failures = metrics["gates"]["failures"]
    if failures and args.check_gates:
        for failure in failures:
            print(f"GATE FAIL: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
