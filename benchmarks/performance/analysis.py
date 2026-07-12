from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import BenchmarkCase
from .schema import validate_result

_HYPOTHESES = {
    "unnecessary_correlation": {
        "stages": ("correlation",),
        "operations": ("dataframe.corr", "dataframe.corrwith"),
        "evidence": (("/freshdata/engine/context.py", ("numeric_corr_matrix",)),),
    },
    "repeated_null_scans": {
        "stages": ("missing",),
        "operations": ("series.isna", "series.notna"),
        "evidence": (
            (
                "/freshdata/engine/missing.py",
                (
                    "auto_missing",
                    "_handle_column",
                    "_fill_low",
                    "_fill_medium",
                    "_handle_high",
                    "_handle_extreme",
                    "_fill_datetime",
                    "_fill",
                    "_assign_filled",
                    "_preserve",
                    "_drop",
                    "_maybe_indicator",
                    "_knn_fill",
                    "_non_collinear_partners",
                ),
            ),
            (
                "/freshdata/steps/missing.py",
                ("impute_missing", "_fill_value", "_strategy_for_column"),
            ),
        ),
    },
    "repeated_uniqueness_scans": {
        "stages": ("role_inference",),
        "operations": ("series.nunique", "series.value_counts"),
        "evidence": (
            (
                "/freshdata/engine/context.py",
                (
                    "infer_role",
                    "_safe_nunique",
                    "_mode_ratio",
                    "_looks_like_text",
                    "build_context",
                    "build_contexts",
                ),
            ),
        ),
    },
    "copy_pressure": {
        "stages": ("context", "engine_cache"),
        "operations": ("dataframe.copy", "series.copy"),
        "evidence": (
            ("/freshdata/engine/context.py", ("build_context", "build_contexts")),
            ("/freshdata/engine/cache.py", ("build_engine_cache",)),
        ),
    },
    "dtype_conversion_pressure": {
        "stages": ("dtype_repair",),
        "operations": ("series.astype", "dataframe.astype"),
        "evidence": (
            (
                "/freshdata/steps/dtypes.py",
                (
                    "fix_dtypes",
                    "suggest_conversion",
                    "_try_boolean",
                    "_try_numeric",
                    "_try_datetime",
                    "_finalize_numeric",
                    "_to_numeric_or_none",
                    "_parse_datetime",
                ),
            ),
        ),
    },
    "report_finalization_overhead": {
        "stages": ("report_finalization", "audit_events"),
        "operations": (),
        "evidence": (
            (
                "/freshdata/report.py",
                ("add", "to_dict", "summary", "brief", "cells_changed", "to_frame"),
            ),
            ("/freshdata/cleaner.py", ("run_pipeline",)),
        ),
    },
    "optional_ml_overhead": {
        "stages": ("semantic_ml",),
        "operations": (),
        "evidence": (
            (
                "/freshdata/imputation/missforest.py",
                ("impute", "_fit_predict_column", "_initial_filled_frame", "_features"),
            ),
            ("/freshdata/semantic/apply.py", ("run_semantic", "resolve_replacements")),
            (
                "/freshdata/semantic/profiler.py",
                ("profile_proposals", "profile_semantic_issues", "plan_semantic"),
            ),
        ),
    },
    "backend_conversion_overhead": {
        "stages": ("backend_conversion",),
        "operations": (),
        "evidence": (
            (
                "/freshdata/adapters/polars.py",
                ("to_pandas", "from_pandas", "is_polars_frame"),
            ),
            (
                "/freshdata/execution/backends/_pandas.py",
                ("materialize_to_pandas", "execute"),
            ),
            (
                "/freshdata/execution/backends/_polars.py",
                ("execute", "_fallback", "_to_lazy", "_collect"),
            ),
            (
                "/freshdata/execution/backends/_duckdb.py",
                ("execute", "_fallback", "_as_arrow"),
            ),
            (
                "/freshdata/execution/backends/_spark.py",
                ("execute", "_fallback", "_to_spark"),
            ),
            (
                "/freshdata/execution/backends/_freshcore.py",
                ("execute", "_fallback", "_frame_from_native"),
            ),
        ),
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
    relationships: tuple[tuple[str, tuple[str, ...]], ...],
) -> bool:
    path = str(record.get("file", "")).replace("\\", "/").lower()
    function = str(record.get("function", "")).lower()
    return any(
        path.endswith(path_suffix) and function in functions
        for path_suffix, functions in relationships
    )


def _matches_allocation(
    record: dict[str, object],
    relationships: tuple[tuple[str, tuple[str, ...]], ...],
) -> bool:
    path = str(record.get("file", "")).replace("\\", "/").lower()
    return any(path.endswith(path_suffix) for path_suffix, _functions in relationships)


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
            if isinstance(record, dict) and _matches(record, rule["evidence"])
        ]
        allocation_evidence = [
            record
            for record in allocations
            if isinstance(record, dict) and _matches_allocation(record, rule["evidence"])
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
        case.get("dataset_type", ""),
        case.get("config_name", ""),
        case.get("return_report", False),
        case.get("backend", ""),
        case.get("output_format", ""),
        case.get("seed", 0),
        result.get("baseline_name") or "",
        case_id_for_result(result),
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
            case_id = case_id_for_result(result)
            hypotheses[case_id] = {
                "label": case_label(result),
                "decisions": decisions,
            }
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


def case_label(result: dict[str, Any]) -> str:
    case = result["case"]
    return " ".join(
        (
            f"case_id={case_id_for_result(result)}",
            f"rows={case['rows']}",
            f"width={case['width']}",
            f"dataset_type={case['dataset_type']}",
            f"config={case['config_name']}",
            f"options={json.dumps(case['options'], sort_keys=True, separators=(',', ':'))}",
            f"report={str(case['return_report']).lower()}",
            f"backend={case['backend']}",
            f"output={case['output_format']}",
            f"seed={case['seed']}",
            f"warmups={case['warmups']}",
            f"repetitions={case['repetitions']}",
        )
    )


def case_id_for_result(result: dict[str, Any]) -> str:
    return BenchmarkCase(**result["case"]).case_id


def _reject_non_standard_json_constant(value: str) -> None:
    raise ValueError(f"result JSON contains non-standard constant: {value}")


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_standard_json_constant,
        )
        if not isinstance(payload, dict):
            raise TypeError(f"result must be an object: {path}")
        validate_result(payload)
        payloads.append(payload)
    return payloads
