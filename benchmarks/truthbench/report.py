"""Versioned, atomic TruthBench result artifacts.

``latest.json`` is the machine-readable run (validated against the result
schema before a byte is written), ``latest.md`` the human summary, and
``failures/`` one sanitized JSON file per minimized failure.  Writes go to a
temporary file in the target directory followed by ``os.replace`` so a
crashed run can never leave a half-written artifact claiming success.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .minimize import FailureCase
from .models import RunResult
from .schema import validate_run

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def render_markdown(run: RunResult, failures: Sequence[FailureCase] = ()) -> str:
    payload = run.to_dict()
    gates = payload["gates"]
    passed = sum(1 for gate in gates if gate["passed"])
    lines = [
        "# TruthBench run",
        "",
        f"- run id: `{run.run_id}`",
        f"- profile: `{run.profile}`",
        f"- records: {len(run.records)}",
        f"- required backends: {', '.join(run.required_backends)}",
        f"- gates: {passed}/{len(gates)} passed",
        f"- overall: {'PASS' if payload['summary'].get('overall_passed') else 'FAIL'}",
        "",
        "## Environment",
        "",
    ]
    for key, value in sorted(payload["environment"].items()):
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Gates", "", "| gate | status | failures |", "|---|---|---|"]
    for gate in gates:
        status = "✅ pass" if gate["passed"] else "❌ FAIL"
        lines.append(f"| {gate['name']} | {status} | {gate['failure_count']} |")
    failing = [gate for gate in gates if not gate["passed"]]
    if failing:
        lines += ["", "## Failure detail", ""]
        for gate in failing:
            lines.append(f"### {gate['name']}")
            lines.append("")
            for item in gate["failures"][:25]:
                lines.append(f"- {item}")
            if gate["failure_count"] > 25:
                lines.append(f"- … and {gate['failure_count'] - 25} more")
            lines.append("")
    if failures:
        lines += ["", "## Minimized failures", ""]
        for case in failures:
            lines.append(
                f"- `{case.failure_id}` {case.gate} @ {case.cell_id} "
                f"({len(case.frame_records)} rows) — `{case.reproduce_command}`"
            )
    lines.append("")
    return "\n".join(lines)


def compare_to_baseline(run: RunResult, baseline_path: Path) -> list[str]:
    """Regression evidence only: a baseline never excuses an absolute failure."""

    notes: list[str] = []
    if not baseline_path.is_file():
        return ["no baseline recorded yet"]
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"baseline unreadable: {exc}"]
    previous = {
        gate["name"]: gate["passed"] for gate in baseline.get("gates", ())
    }
    for gate in run.gates:
        before = previous.get(gate.name)
        if before is True and not gate.passed:
            notes.append(f"regression: gate {gate.name} passed in baseline, fails now")
        elif before is False and gate.passed:
            notes.append(f"improvement: gate {gate.name} failed in baseline, passes now")
    return notes or ["no gate-level changes vs baseline"]


def write_artifacts(
    run: RunResult,
    failures: Sequence[FailureCase] = (),
    *,
    results_dir: Path | str = RESULTS_DIR,
) -> dict[str, Path]:
    """Validate and atomically persist all artifacts for one run."""

    results_dir = Path(results_dir)
    payload = run.to_dict()
    validate_run(payload)

    latest_json = results_dir / "latest.json"
    latest_md = results_dir / "latest.md"
    _atomic_write(latest_json, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _atomic_write(latest_md, render_markdown(run, failures))
    written = {"latest.json": latest_json, "latest.md": latest_md}
    failures_dir = results_dir / "failures"
    for case in failures:
        path = failures_dir / f"{case.failure_id}.json"
        _atomic_write(path, json.dumps(case.to_dict(), indent=2, sort_keys=True) + "\n")
        written[f"failures/{case.failure_id}.json"] = path
    return written


__all__ = [
    "RESULTS_DIR",
    "compare_to_baseline",
    "render_markdown",
    "write_artifacts",
]
