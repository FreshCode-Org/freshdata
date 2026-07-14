from __future__ import annotations

from typing import Any

from .analysis import case_id_for_result, case_label
from .schema import validate_finite_numbers


def _number(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _ratio(value: object) -> str:
    return "∞" if value is None else f"{float(value):.3f}x"


def _case_sort_key(result: dict[str, Any]) -> tuple[object, ...]:
    case = result.get("case", {})
    return (
        case.get("rows", 0),
        case.get("width", ""),
        case.get("dataset_type", ""),
        case.get("config_name", ""),
        case.get("return_report", False),
        case.get("backend", ""),
        case.get("output_format", ""),
        case.get("seed", 0),
        result.get("baseline_name") or "",
        case_id_for_result(result),
    )


def _table(results: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Rows | Width | Config / operation | Report | Backend | Format | Median s | "
        "Min s | Max s | CV | Rows/s | Peak RSS/input | Peak Python/input | Comparison |",
        "| ---: | :--- | :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | :--- |",
    ]
    for result in sorted(results, key=_case_sort_key):
        case = result["case"]
        input_bytes = result.get("input_bytes") or 0
        python_ratio = result.get("peak_python_bytes", 0) / input_bytes if input_bytes else None
        comparisons = result.get("comparisons", [])
        comparison = (
            "; ".join(
                f"{item['baseline_name']} {_ratio(item['ratio'])} ({item['classification']})"
                for item in comparisons
            )
            or "—"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(case["rows"]),
                    str(case["width"]),
                    str(result.get("baseline_name") or case["config_name"]),
                    str(case["return_report"]).lower(),
                    str(case["backend"]),
                    str(case["output_format"]),
                    _number(result.get("median_seconds")),
                    _number(result.get("min_seconds")),
                    _number(result.get("max_seconds")),
                    _number(result.get("coefficient_of_variation")),
                    _number(result.get("throughput_rows_per_second"), 1),
                    _number(result.get("input_to_peak_ratio")),
                    _number(python_ratio),
                    comparison,
                )
            )
            + " |"
        )
    if len(lines) == 2:
        lines.append(
            "| — | — | — | — | — | — | — | — | — | — | — | — | — | No completed results |"
        )
    return lines


def render_report(payload: dict[str, Any]) -> str:  # noqa: PLR0915
    validate_finite_numbers(payload, "report payload")
    environment = payload.get("environment", {})
    results = list(payload.get("results", []))
    baselines = list(payload.get("component_baselines", []))
    all_results = results + baselines
    completed = [item for item in all_results if item.get("status") == "completed"]
    failures = [item for item in all_results if item.get("status") != "completed"]
    hypotheses = payload.get("hypotheses", {})
    decisions = [
        (case_id, case_record["label"], name, decision)
        for case_id, case_record in sorted(hypotheses.items())
        for name, decision in sorted(case_record["decisions"].items())
    ]
    lines = [
        "# FreshData performance evidence",
        "",
        "## Environment",
        "",
    ]
    if environment:
        lines.extend(f"- {key}: `{environment[key]}`" for key in sorted(environment))
    else:
        lines.append("- No environment metadata was recorded.")
    lines.extend(
        [
            "",
            "## Architecture and execution flow",
            "",
            "FreshData cases and named pandas component operations run in isolated "
            "worker processes. Timing samples and the PeakRSS/tracemalloc measurement "
            "are collected separately.",
            "",
            "## Reproduction commands",
            "",
        ]
    )
    commands = sorted(set(payload.get("reproduction_commands", [])))
    lines.extend(f"- `{command}`" for command in commands)
    if not commands:
        lines.append("- No command was recorded.")
    lines.extend(["", "## Baseline benchmark table", ""])
    lines.extend(_table(completed))
    lines.extend(["", "## Profiling findings", ""])
    profile_found = False
    for result in sorted(results, key=_case_sort_key):
        profile = result.get("profile")
        if not isinstance(profile, dict):
            continue
        profile_found = True
        lines.append(f"### {case_label(result)}")
        lines.append("")
        total = float(profile.get("stages", {}).get("total", 0.0))
        for stage, seconds in sorted(profile.get("stages", {}).items()):
            if stage == "total":
                continue
            fraction = float(seconds) / total if total else 0.0
            lines.append(f"- {stage}: {fraction:.1%}")
        functions = sorted(
            profile.get("functions", []),
            key=lambda item: (
                -float(item["cumulative_seconds"]),
                -float(item["self_seconds"]),
                str(item["file"]).replace("\\", "/"),
                int(item["line"]),
                str(item["function"]),
                int(item["calls"]),
            ),
        )
        for function in functions[:10]:
            lines.append(
                f"- `{function['file']}:{function['line']} {function['function']}` — "
                f"self {_number(function['self_seconds'])} s, cumulative "
                f"{_number(function['cumulative_seconds'])} s, calls "
                f"{function['calls']}"
            )
        lines.append("")
    if not profile_found:
        lines.append("No profiling records were supplied.")
    lines.extend(["", "## Confirmed root causes", ""])
    candidates = [item for item in decisions if item[3].get("status") == "candidate"]
    if candidates:
        lines.append(
            "No cause is confirmed by profiling alone; these exact-evidence items are "
            "performance candidates:"
        )
        lines.append("")
    for case_id, label, name, decision in candidates:
        lines.append(
            f"- `{name}` in `{case_id}` ({label}): candidate; stage "
            f"{float(decision['stage_fraction']):.1%}, observed calls "
            f"{decision['observed_calls']}."
        )
    if not candidates:
        lines.append("No root cause met the exact-evidence and significance thresholds.")
    lines.extend(["", "## Rejected hypotheses", ""])
    rejected = [item for item in decisions if item[3].get("status") != "candidate"]
    for case_id, label, name, decision in rejected:
        lines.append(
            f"- `{name}` in `{case_id}` ({label}): {decision['status']}; stage "
            f"{float(decision['stage_fraction']):.1%}, observed calls "
            f"{decision['observed_calls']}."
        )
    if not rejected:
        lines.append("No rejected or evidence-limited hypotheses were recorded.")
    lines.extend(["", "## Failures, timeouts, and OOMs", ""])
    for result in sorted(failures, key=_case_sort_key):
        case = result["case"]
        lines.append(
            f"- {case['rows']} rows / {case['width']} / {case['config_name']}: "
            f"{result['status']} ({result.get('error_type') or 'unknown'}: "
            f"{result.get('error_message') or 'no message'})."
        )
    if not failures:
        lines.append("No failures, timeouts, or OOMs were recorded.")
    lines.extend(["", "## Limitations", ""])
    limitations = payload.get("limitations", [])
    lines.extend(f"- {limitation}" for limitation in limitations)
    if not limitations:
        lines.append("- No limitations were supplied.")
    return "\n".join(lines).rstrip() + "\n"
