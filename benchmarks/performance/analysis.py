from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schema import validate_result

_HYPOTHESES = {
    "unnecessary_correlation": {
        "stages": ("correlation",),
        "operations": ("dataframe.corr", "dataframe.corrwith"),
        "functions": ("numeric_corr_matrix", "corr", "corrwith"),
        "paths": ("/engine/context.py",),
    },
    "repeated_null_scans": {
        "stages": ("missing",),
        "operations": ("series.isna", "series.notna"),
        "functions": ("isna", "notna"),
        "paths": ("/engine/missing.py", "/steps/missing.py"),
    },
    "repeated_uniqueness_scans": {
        "stages": ("role_inference",),
        "operations": ("series.nunique", "series.value_counts"),
        "functions": ("infer_role", "build_context", "nunique", "value_counts"),
        "paths": (),
    },
    "copy_pressure": {
        "stages": ("context", "engine_cache"),
        "operations": ("dataframe.copy", "series.copy"),
        "functions": ("copy",),
        "paths": ("/engine/context.py", "/engine/cache.py"),
    },
    "dtype_conversion_pressure": {
        "stages": ("dtype_repair",),
        "operations": ("series.astype", "dataframe.astype"),
        "functions": ("astype",),
        "paths": ("/steps/dtypes.py",),
    },
    "report_finalization_overhead": {
        "stages": ("report_finalization", "audit_events"),
        "operations": (),
        "functions": ("memory_bytes", "cleanreport.add"),
        "paths": ("/report.py", "/cleaner.py"),
    },
    "optional_ml_overhead": {
        "stages": ("semantic_ml",),
        "operations": (),
        "functions": (),
        "paths": ("/semantic/", "/imputation/missforest.py", "/sklearn/"),
    },
    "backend_conversion_overhead": {
        "stages": ("backend_conversion",),
        "operations": (),
        "functions": (),
        "paths": ("/adapters/", "/execution/backends/"),
    },
}


def classify_change(
    baseline: float,
    candidate: float,
    baseline_cv: float,
    candidate_cv: float,
) -> str:
    if baseline <= 0:
        raise ValueError("baseline must be positive")
    if candidate < 0:
        raise ValueError("candidate must be non-negative")
    if baseline_cv < 0 or candidate_cv < 0:
        raise ValueError("variation must be non-negative")
    change = (baseline - candidate) / baseline
    threshold = max(0.10, 2 * max(baseline_cv, candidate_cv))
    if abs(change) < threshold and not math.isclose(
        abs(change), threshold, rel_tol=1e-12, abs_tol=1e-12
    ):
        return "noise"
    return "improved" if change > 0 else "regressed"


def _matches(
    record: dict[str, object],
    functions: tuple[str, ...],
    paths: tuple[str, ...],
) -> bool:
    path = str(record.get("file", "")).replace("\\", "/").lower()
    function = str(record.get("function", "")).lower()
    return function in functions or any(term in path for term in paths)


def classify_hypotheses(
    profile: dict[str, object], *, traced_peak_bytes: int | None = None
) -> dict[str, dict[str, object]]:
    stages = profile.get("stages", {})
    operations = profile.get("operations", {})
    functions = profile.get("functions", [])
    allocations = profile.get("allocations", [])
    if not isinstance(stages, dict) or not isinstance(operations, dict):
        raise TypeError("profile stages and operations must be objects")
    if not isinstance(functions, list) or not isinstance(allocations, list):
        raise TypeError("profile functions and allocations must be arrays")
    total = float(stages.get("total", 0.0))
    decisions: dict[str, dict[str, object]] = {}
    for name, rule in _HYPOTHESES.items():
        stage_seconds = sum(float(stages.get(stage, 0.0)) for stage in rule["stages"])
        stage_fraction = stage_seconds / total if total > 0 else 0.0
        function_evidence = [
            record
            for record in functions
            if isinstance(record, dict) and _matches(record, rule["functions"], rule["paths"])
        ]
        allocation_evidence = [
            record
            for record in allocations
            if isinstance(record, dict) and _matches(record, (), rule["paths"])
        ]
        operation_calls = sum(int(operations.get(key, 0)) for key in rule["operations"])
        exact_function_calls = sum(int(record.get("calls", 0)) for record in function_evidence)
        observed_calls = operation_calls or exact_function_calls
        significant_allocations = (
            traced_peak_bytes is not None
            and traced_peak_bytes > 0
            and sum(int(record.get("bytes", 0)) for record in allocation_evidence)
            / traced_peak_bytes
            >= 0.10
        )
        evidence: list[dict[str, object]] = []
        evidence.extend(function_evidence)
        evidence.extend(allocation_evidence)
        if not evidence or observed_calls == 0:
            status = "insufficient_evidence"
        elif stage_fraction >= 0.10 or significant_allocations:
            status = "candidate"
        else:
            status = "rejected"
        decisions[name] = {
            "status": status,
            "stage_fraction": stage_fraction,
            "observed_calls": observed_calls,
            "evidence": evidence,
        }
    return decisions


def _sort_key(result: dict[str, Any]) -> tuple[object, ...]:
    case = result.get("case", {})
    return (
        case.get("rows", 0),
        case.get("width", ""),
        case.get("config_name", ""),
        case.get("return_report", False),
        case.get("backend", ""),
        case.get("output_format", ""),
        result.get("baseline_name") or "",
    )


def analyze_results(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted((dict(result) for result in results), key=_sort_key)
    for result in ordered:
        validate_result(result)
    component_baselines = [
        result
        for result in ordered
        if result.get("case", {}).get("backend") == "pandas-component-baseline"
    ]
    freshdata_results = [result for result in ordered if result not in component_baselines]
    baselines_by_semantics = {
        (
            result.get("baseline_name"),
            result["case"]["rows"],
            result["case"]["width"],
            result["case"]["dataset_type"],
            result["case"]["seed"],
        ): result
        for result in component_baselines
        if result.get("status") == "completed"
    }
    hypotheses: dict[str, dict[str, object]] = {}
    for result in freshdata_results:
        result["comparisons"] = []
        options = result["case"].get("options", {})
        comparable_name = options.get("comparable_baseline") if isinstance(options, dict) else None
        baseline = baselines_by_semantics.get(
            (
                comparable_name,
                result["case"]["rows"],
                result["case"]["width"],
                result["case"]["dataset_type"],
                result["case"]["seed"],
            )
        )
        if (
            baseline is not None
            and result.get("status") == "completed"
            and baseline.get("median_seconds") is not None
            and result.get("median_seconds") is not None
        ):
            baseline_seconds = float(baseline["median_seconds"])
            candidate_seconds = float(result["median_seconds"])
            result["comparisons"].append(
                {
                    "baseline_name": comparable_name,
                    "baseline_seconds": baseline_seconds,
                    "candidate_seconds": candidate_seconds,
                    "ratio": baseline_seconds / candidate_seconds if candidate_seconds else None,
                    "classification": classify_change(
                        baseline_seconds,
                        candidate_seconds,
                        float(baseline.get("coefficient_of_variation") or 0.0),
                        float(result.get("coefficient_of_variation") or 0.0),
                    ),
                }
            )
        profile = result.get("profile")
        if isinstance(profile, dict):
            decisions = classify_hypotheses(
                profile, traced_peak_bytes=result.get("peak_python_bytes")
            )
            hypotheses[_case_label(result)] = decisions
    environment = ordered[0]["environment"] if ordered else {}
    commands = sorted(
        {str(result.get("command", "")) for result in ordered if result.get("command")}
    )
    return {
        "schema_version": 1,
        "environment": environment,
        "results": freshdata_results,
        "component_baselines": component_baselines,
        "hypotheses": hypotheses,
        "reproduction_commands": commands,
        "limitations": [
            "Component baselines cover only their named pandas operation; no full "
            "balanced FreshData pipeline equivalence is claimed.",
            "Timing classifications require both a 10% effect and twice the larger "
            "observed coefficient of variation.",
        ],
    }


def _case_label(result: dict[str, Any]) -> str:
    case = result["case"]
    return "/".join(
        str(case[key])
        for key in ("rows", "width", "config_name", "return_report", "backend", "output_format")
    )


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"result must be an object: {path}")
        validate_result(payload)
        payloads.append(payload)
    return payloads
