"""Render gauntlet results as JSON and Markdown, and gate against a baseline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Regression gates: a PR fails when any fixture drops below these.
GATES = {
    "preservation_rate": 1.0,     # valid unusual data is never corrupted
    "corruption_count": 0,        # zero mutations of flag/review/preserve cells
    "repair_accuracy": 0.95,
    "detection_f1": 0.85,
    "escape_rate_max": 0.10,
    "audit_completeness": 1.0,
}


def results_payload(metrics: dict[str, dict[str, Any]], *, n_rows: int,
                    seed: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "seed": seed,
        "fixtures": metrics,
    }


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# FreshData Validation Gauntlet",
        "",
        f"Generated {payload['generated_at']} · {payload['n_rows']} rows per "
        f"fixture · seed {payload['seed']}",
        "",
        "Gold-labelled dispositions: every injected defect carries the outcome "
        "FreshData should choose (preserve / repair / flag / review). "
        "`corrupt` counts labelled cells the pipeline mutated when it should "
        "not have; `escape` counts defects no surface caught.",
        "",
        "| fixture | cells | P | R | F1 | repair | review | preserve "
        "| corrupt | escape | FPR | audit | determinism | trust mono | clean s |",
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|:--:|--:|",
    ]
    for name, m in sorted(payload["fixtures"].items()):
        d = m["detection"]
        fmt = lambda v: "—" if v is None else f"{v:g}"  # noqa: E731
        lines.append(
            f"| {name} | {m['labelled_cells']} | {d['precision']:g} | "
            f"{d['recall']:g} | {d['f1']:g} | {fmt(m['repair_accuracy'])} | "
            f"{fmt(m['review_routing'])} | {fmt(m['preservation_rate'])} | "
            f"{m['corruption_count']} | {m['escape_rate']:g} | "
            f"{m['false_positive_rate']:g} | {m['audit_completeness']:g} | "
            f"{'✅' if m['deterministic'] else '❌'} | "
            f"{'✅' if m['trust']['monotonic'] else '❌'} | "
            f"{m['performance']['clean_seconds']:g} |"
        )

    lines += ["", "## Failure catalogue", ""]
    any_fail = False
    for name, m in sorted(payload["fixtures"].items()):
        for f in m["failures"]:
            any_fail = True
            lines.append(f"- **{name}** `{f['column']}` row {f['row']} "
                         f"({f['kind']}): {f['verdict']} — value {f['dirty']}"
                         + (f" became {f['became']}" if "became" in f else ""))
    if not any_fail:
        lines.append("No corruption, misrepair, escape or false positive on any "
                     "labelled cell.")

    partial = [(name, f) for name, m in sorted(payload["fixtures"].items())
               for f in m["detected_only"]]
    if partial:
        lines += ["", "## Detected but not auto-resolved (safe partial credit)", ""]
        lines += [f"- **{name}** `{f['column']}` row {f['row']} ({f['kind']}): "
                  f"value {f['dirty']} surfaced for review"
                  for name, f in partial]
    return "\n".join(lines) + "\n"


def check_gates(payload: dict[str, Any],
                baseline: dict[str, Any] | None) -> list[str]:
    """Absolute gates plus no-regression against the stored baseline."""
    problems: list[str] = []
    for name, m in sorted(payload["fixtures"].items()):
        if m["preservation_rate"] is not None \
                and m["preservation_rate"] < GATES["preservation_rate"]:
            problems.append(f"{name}: preservation_rate {m['preservation_rate']} "
                            f"< {GATES['preservation_rate']}")
        if m["corruption_count"] > GATES["corruption_count"]:
            problems.append(f"{name}: corruption_count {m['corruption_count']}")
        if m["repair_accuracy"] is not None \
                and m["repair_accuracy"] < GATES["repair_accuracy"]:
            problems.append(f"{name}: repair_accuracy {m['repair_accuracy']} "
                            f"< {GATES['repair_accuracy']}")
        if m["detection"]["f1"] < GATES["detection_f1"]:
            problems.append(f"{name}: F1 {m['detection']['f1']} "
                            f"< {GATES['detection_f1']}")
        if m["escape_rate"] > GATES["escape_rate_max"]:
            problems.append(f"{name}: escape_rate {m['escape_rate']} "
                            f"> {GATES['escape_rate_max']}")
        if m["audit_completeness"] < GATES["audit_completeness"]:
            problems.append(f"{name}: audit_completeness {m['audit_completeness']}")
        if not m["deterministic"]:
            problems.append(f"{name}: non-deterministic run")
        if not m["trust"]["monotonic"]:
            problems.append(f"{name}: trust score not monotonic")

    if baseline:
        for name, m in sorted(payload["fixtures"].items()):
            base = baseline.get("fixtures", {}).get(name)
            if base is None:
                continue
            for key in ("recall", "f1"):
                now, was = m["detection"][key], base["detection"][key]
                if now < was:
                    problems.append(
                        f"{name}: detection {key} regressed {was} -> {now}")
    return problems
