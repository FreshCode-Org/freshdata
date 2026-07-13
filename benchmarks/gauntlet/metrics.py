"""Score gauntlet observations against gold dispositions."""

from __future__ import annotations

from typing import Any

from .runner import CellObservation, FixtureRun, _values_equal

#: validate_fields actions that satisfy an ``expect="review"`` label.
REVIEW_ACTIONS = frozenset({"quarantine", "manual_review", "reject"})


def _detected(o: CellObservation) -> bool:
    """The cell was surfaced somewhere a user would see it."""
    return bool(
        o.issue is not None
        or o.semantic is not None
        or o.pii_types
        or o.lint_hit
        or o.quarantined
        or (o.cell.expect == "repair" and (o.changed_by_clean or o.textclean_changed
                                           or o.normalized is not None))
    )


def _repair_source(o: CellObservation) -> str | None:
    """Which surface produced the gold value, or ``None``."""
    gold = o.cell.repaired
    if _values_equal(o.cell.dirty, gold):
        # representational repair ('402.10' -> 402.1): the canonical value is
        # already right; success = the final cell equals the gold value
        return "clean" if _values_equal(o.clean_value, gold) else None
    if _values_equal(o.clean_value, gold):
        return "clean"
    if o.cell.accept_impute and o.changed_by_clean and o.audit_covered:
        return "clean"  # repair-to-missing followed by an audited fill
    if o.normalized is not None and _values_equal(o.normalized, gold):
        return "validate_fields"
    if o.textclean_changed and _values_equal(o.textclean_value, gold):
        return "clean_text"
    if o.semantic is not None and _values_equal(o.semantic_value, gold):
        return "semantic_auto"
    if o.aggressive_value is not None and _values_equal(o.aggressive_value, gold) \
            and not _values_equal(o.cell.dirty, gold):
        return "clean_text_opt_in"
    return None


def score_cell(o: CellObservation) -> dict[str, Any]:
    """One labelled cell -> outcome dict with ``verdict`` and failure detail."""
    expect = o.cell.expect
    detected = _detected(o)
    outcome: dict[str, Any] = {
        "row": int(o.cell.row), "column": o.cell.column, "kind": o.cell.kind,
        "expect": expect, "detected": detected,
        "dirty": repr(o.cell.dirty),
    }

    if expect == "preserve":
        corrupted = o.changed_by_clean or o.textclean_changed
        false_alarm = o.issue is not None and o.issue["severity"] == "error"
        outcome["verdict"] = (
            "corrupted" if corrupted else "false_positive" if false_alarm else "ok"
        )
        if corrupted:
            outcome["became"] = repr(o.textclean_value if o.textclean_changed
                                     else o.clean_value)
        return outcome

    if expect == "repair":
        source = _repair_source(o)
        if source is not None:
            outcome["verdict"] = "repaired"
            outcome["source"] = source
        elif o.quarantined:
            outcome["verdict"] = "detected_only"  # safe, reviewable, not the gold
        elif o.changed_by_clean:
            outcome["verdict"] = "misrepaired"
            outcome["became"] = repr(o.clean_value)
        elif detected:
            outcome["verdict"] = "detected_only"
        else:
            outcome["verdict"] = "escaped"
        return outcome

    if expect == "flag":
        if o.changed_by_clean and not o.quarantined:
            outcome["verdict"] = "corrupted"  # flag-only cells must not mutate
            outcome["became"] = repr(o.clean_value)
        elif detected:
            outcome["verdict"] = "flagged"
        else:
            outcome["verdict"] = "escaped"
        return outcome

    # review: the cell must land in a human-review pathway
    routed = (o.issue is not None and o.issue["action"] in REVIEW_ACTIONS) \
        or o.quarantined
    if o.changed_by_clean and not o.quarantined:
        outcome["verdict"] = "corrupted"
        outcome["became"] = repr(o.clean_value)
    elif routed:
        outcome["verdict"] = "reviewed"
    elif detected:
        outcome["verdict"] = "detected_only"
    else:
        outcome["verdict"] = "escaped"
    return outcome


def compute_metrics(run: FixtureRun) -> dict[str, Any]:
    outcomes = [score_cell(o) for o in run.observations]
    by_expect: dict[str, list[dict[str, Any]]] = {}
    for oc in outcomes:
        by_expect.setdefault(oc["expect"], []).append(oc)

    problems = [oc for oc in outcomes if oc["expect"] != "preserve"]
    preserves = by_expect.get("preserve", [])
    repairs = by_expect.get("repair", [])
    reviews = by_expect.get("review", [])

    tp = sum(1 for oc in problems
             if oc["detected"] or oc["verdict"] in ("repaired", "flagged", "reviewed"))
    fn = len(problems) - tp
    fp = len(run.false_positive_cells) + sum(
        1 for oc in preserves if oc["verdict"] == "false_positive")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    corrupted = [oc for oc in outcomes if oc["verdict"] == "corrupted"]
    misrepaired = [oc for oc in outcomes if oc["verdict"] == "misrepaired"]
    escaped = [oc for oc in outcomes if oc["verdict"] == "escaped"]
    repaired = [oc for oc in repairs if oc["verdict"] == "repaired"]
    reviewed = [oc for oc in reviews if oc["verdict"] == "reviewed"]

    dup_ok = run.duplicates_removed == run.fixture.dup_row_count

    return {
        "fixture": run.fixture.name,
        "n_rows": len(run.fixture.df),
        "labelled_cells": len(outcomes),
        "detection": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
        },
        "repair_accuracy": round(len(repaired) / len(repairs), 4) if repairs else None,
        "repair_sources": {
            src: sum(1 for oc in repaired if oc.get("source") == src)
            for src in sorted({oc.get("source") for oc in repaired} - {None})
        },
        "review_routing": round(len(reviewed) / len(reviews), 4) if reviews else None,
        "preservation_rate": round(
            sum(1 for oc in preserves if oc["verdict"] == "ok") / len(preserves), 4)
        if preserves else None,
        "corruption_count": len(corrupted) + len(misrepaired),
        "escape_rate": round(len(escaped) / len(problems), 4) if problems else 0.0,
        "false_positive_rate": round(
            fp / (run.n_clean_cells_checked + len(preserves)), 6),
        "duplicates": {"expected": run.fixture.dup_row_count,
                       "removed": run.duplicates_removed, "ok": dup_ok},
        "audit_completeness": round(run.audit_recorded / run.audit_mutations, 4)
        if run.audit_mutations else 1.0,
        "deterministic": run.deterministic,
        "trust": {
            "pristine_frame": round(run.trust_pristine, 3),
            "dirty_frame": round(run.trust_dirty, 3),
            "monotonic": run.trust_dirty <= run.trust_pristine,
        },
        "performance": {
            "clean_seconds": run.clean_seconds,
            "validate_seconds": run.validate_seconds,
            "peak_memory_mb": run.peak_memory_mb,
        },
        "failures": [oc for oc in outcomes
                     if oc["verdict"] in ("corrupted", "misrepaired", "escaped",
                                          "false_positive")],
        "detected_only": [oc for oc in outcomes if oc["verdict"] == "detected_only"],
    }
