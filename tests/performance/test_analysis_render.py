from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks.performance.analysis import (
    analyze_results,
    classify_change,
    classify_hypotheses,
)
from benchmarks.performance.baselines import BASELINES, measure_pandas_baseline
from benchmarks.performance.cli import main
from benchmarks.performance.models import BenchmarkCase
from benchmarks.performance.render import render_report


def _case(**overrides: object) -> BenchmarkCase:
    arguments = {
        "rows": 30,
        "width": "narrow",
        "config_name": "component_baseline",
        "options": {},
        "warmups": 0,
        "repetitions": 1,
    }
    arguments.update(overrides)
    return BenchmarkCase(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "baseline,candidate,baseline_cv,candidate_cv,expected",
    [
        (1.0, 0.85, 0.02, 0.02, "improved"),
        (1.0, 0.94, 0.01, 0.01, "noise"),
        (1.0, 1.12, 0.02, 0.02, "regressed"),
        (1.0, 0.89, 0.06, 0.01, "noise"),
        (1.0, 0.90, 0.01, 0.01, "improved"),
    ],
)
def test_change_requires_ten_percent_and_twice_variability(
    baseline: float,
    candidate: float,
    baseline_cv: float,
    candidate_cv: float,
    expected: str,
) -> None:
    assert classify_change(baseline, candidate, baseline_cv, candidate_cv) == expected


def test_change_rejects_non_positive_or_negative_inputs() -> None:
    with pytest.raises(ValueError, match="baseline must be positive"):
        classify_change(0.0, 0.8, 0.0, 0.0)
    with pytest.raises(ValueError, match="candidate must be non-negative"):
        classify_change(1.0, -0.1, 0.0, 0.0)
    with pytest.raises(ValueError, match="variation must be non-negative"):
        classify_change(1.0, 0.8, -0.1, 0.0)


def _empty_profile() -> dict[str, object]:
    return {
        "stages": {
            "context": 0.0,
            "engine_cache": 0.0,
            "correlation": 0.0,
            "missing": 0.0,
            "outliers": 0.0,
            "role_inference": 0.0,
            "dtype_repair": 0.0,
            "duplicates": 0.0,
            "audit_events": 0.0,
            "report_finalization": 0.0,
            "semantic_ml": 0.0,
            "backend_conversion": 0.0,
            "total": 1.0,
        },
        "operations": {
            "dataframe.copy": 0,
            "series.copy": 0,
            "series.isna": 0,
            "series.notna": 0,
            "series.nunique": 0,
            "series.value_counts": 0,
            "series.astype": 0,
            "dataframe.astype": 0,
            "dataframe.corr": 0,
            "dataframe.corrwith": 0,
        },
        "functions": [],
        "allocations": [],
    }


def test_hypothesis_classifier_requires_exact_profile_evidence() -> None:
    profile = _empty_profile()
    profile["stages"]["correlation"] = 0.40  # type: ignore[index]
    profile["operations"]["dataframe.corr"] = 1  # type: ignore[index]
    profile["functions"] = [
        {
            "file": "src/freshdata/engine/context.py",
            "line": 207,
            "function": "numeric_corr_matrix",
            "self_seconds": 0.4,
            "cumulative_seconds": 0.4,
            "calls": 1,
        }
    ]

    decisions = classify_hypotheses(profile)

    decision = decisions["unnecessary_correlation"]
    assert decision["status"] == "candidate"
    assert decision["stage_fraction"] == pytest.approx(0.4)
    assert decision["observed_calls"] == 1
    assert decision["evidence"] == profile["functions"]


def test_hypothesis_classifier_does_not_substitute_unrelated_evidence() -> None:
    profile = _empty_profile()
    profile["stages"]["correlation"] = 0.40  # type: ignore[index]
    profile["operations"]["dataframe.corr"] = 1  # type: ignore[index]
    profile["functions"] = [
        {
            "file": "src/freshdata/report.py",
            "line": 12,
            "function": "finalize",
            "self_seconds": 0.4,
            "cumulative_seconds": 0.4,
            "calls": 1,
        }
    ]

    decision = classify_hypotheses(profile)["unnecessary_correlation"]

    assert decision["status"] == "insufficient_evidence"
    assert decision["evidence"] == []


def test_hypothesis_classifier_does_not_accept_partial_function_name_match() -> None:
    profile = _empty_profile()
    profile["stages"]["correlation"] = 0.40  # type: ignore[index]
    profile["operations"]["dataframe.corr"] = 1  # type: ignore[index]
    profile["functions"] = [
        {
            "file": "src/freshdata/unrelated.py",
            "line": 12,
            "function": "correlation_label",
            "self_seconds": 0.4,
            "cumulative_seconds": 0.4,
            "calls": 1,
        }
    ]

    decision = classify_hypotheses(profile)["unnecessary_correlation"]

    assert decision["status"] == "insufficient_evidence"
    assert decision["evidence"] == []


def test_hypothesis_classifier_rejects_observed_but_small_work() -> None:
    profile = _empty_profile()
    profile["stages"]["missing"] = 0.09  # type: ignore[index]
    profile["operations"]["series.isna"] = 4  # type: ignore[index]
    profile["functions"] = [
        {
            "file": "src/freshdata/engine/missing.py",
            "line": 8,
            "function": "scan_missing",
            "self_seconds": 0.09,
            "cumulative_seconds": 0.09,
            "calls": 4,
        }
    ]

    decision = classify_hypotheses(profile)["repeated_null_scans"]

    assert decision["status"] == "rejected"
    assert decision["observed_calls"] == 4


def test_hypothesis_classifier_can_use_exact_peak_allocation_evidence() -> None:
    profile = _empty_profile()
    profile["operations"]["dataframe.copy"] = 2  # type: ignore[index]
    profile["allocations"] = [
        {
            "file": "src/freshdata/engine/context.py",
            "line": 44,
            "bytes": 120,
            "count": 2,
        },
        {
            "file": "src/freshdata/other.py",
            "line": 9,
            "bytes": 880,
            "count": 1,
        },
    ]

    decision = classify_hypotheses(profile, traced_peak_bytes=1_000)["copy_pressure"]

    assert decision["status"] == "candidate"
    assert decision["evidence"] == [profile["allocations"][0]]  # type: ignore[index]


def test_all_required_hypotheses_are_classified() -> None:
    assert set(classify_hypotheses(_empty_profile())) == {
        "unnecessary_correlation",
        "repeated_null_scans",
        "repeated_uniqueness_scans",
        "copy_pressure",
        "dtype_conversion_pressure",
        "report_finalization_overhead",
        "optional_ml_overhead",
        "backend_conversion_overhead",
    }


def test_component_baseline_names_and_semantics() -> None:
    assert tuple(BASELINES) == (
        "shallow_copy",
        "numeric_median_fill",
        "duplicates",
        "null_counts",
    )


def test_measure_component_baseline_uses_baseline_backend_and_name() -> None:
    result = measure_pandas_baseline(_case(), "null_counts", command="baseline-test")

    assert result.status == "completed"
    assert result.case.backend == "pandas-component-baseline"
    assert result.baseline_name == "null_counts"
    assert result.result_type == "Series"
    assert len(result.samples_seconds) == 1


def test_measure_component_baseline_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown pandas baseline"):
        measure_pandas_baseline(_case(), "balanced", command="baseline-test")


def _result_payload(
    *,
    rows: int = 100,
    backend: str = "pandas",
    baseline_name: str | None = None,
    median: float = 1.0,
    cv: float = 0.01,
    status: str = "completed",
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "case": {
            "rows": rows,
            "width": "narrow",
            "config_name": "default",
            "options": {},
            "dataset_type": "mixed",
            "return_report": False,
            "backend": backend,
            "output_format": "pandas",
            "seed": 42,
            "warmups": 1,
            "repetitions": 5,
        },
        "environment": {
            "git_commit": "abc123",
            "git_dirty": False,
            "python_version": "3.12",
            "pandas_version": "2.3",
            "numpy_version": "2.0",
            "freshdata_version": "1.1.1",
            "optional_versions": {},
            "platform": "test",
            "processor": "test",
            "cpu_count_logical": 2,
            "cpu_count_physical": 1,
            "total_ram_bytes": 1000,
        },
        "samples_seconds": [median] if status == "completed" else [],
        "median_seconds": median if status == "completed" else None,
        "min_seconds": median if status == "completed" else None,
        "max_seconds": median if status == "completed" else None,
        "stdev_seconds": 0.0 if status == "completed" else None,
        "coefficient_of_variation": cv if status == "completed" else None,
        "throughput_rows_per_second": rows / median if status == "completed" else None,
        "peak_rss_bytes": 200 if status == "completed" else None,
        "peak_python_bytes": 100 if status == "completed" else None,
        "input_bytes": 50 if status == "completed" else None,
        "input_to_peak_ratio": 4.0 if status == "completed" else None,
        "command": "python -m benchmarks.performance run",
        "error_type": None,
        "error_message": None,
        "output_fingerprint": None,
        "report_fingerprint": None,
        "result_type": None,
        "profile": None,
        "baseline_name": baseline_name,
    }


def test_analysis_matches_only_semantically_scoped_component_baselines() -> None:
    freshdata = _result_payload(median=0.94)
    baseline = _result_payload(
        backend="pandas-component-baseline",
        baseline_name="null_counts",
        median=1.0,
    )

    summary = analyze_results([freshdata, baseline])

    assert summary["results"][0]["comparisons"] == []
    assert summary["component_baselines"][0]["baseline_name"] == "null_counts"
    assert "full balanced" in summary["limitations"][0]


def test_analysis_compares_only_an_explicit_matching_component_operation() -> None:
    freshdata = _result_payload(median=0.8)
    freshdata["case"]["options"] = {"comparable_baseline": "null_counts"}  # type: ignore[index]
    baseline = _result_payload(
        backend="pandas-component-baseline",
        baseline_name="null_counts",
        median=1.0,
    )

    comparison = analyze_results([freshdata, baseline])["results"][0]["comparisons"][0]

    assert comparison == {
        "baseline_name": "null_counts",
        "baseline_seconds": 1.0,
        "candidate_seconds": 0.8,
        "ratio": 1.25,
        "classification": "improved",
    }


def test_renderer_is_deterministic_and_contains_required_sections() -> None:
    payload = {
        "schema_version": 1,
        "environment": {"python_version": "3.12", "pandas_version": "2.3"},
        "results": [],
        "component_baselines": [],
        "hypotheses": {},
        "reproduction_commands": [],
        "limitations": ["No full-pipeline pandas equivalence is claimed."],
    }
    first = render_report(payload)
    assert first == render_report(payload)
    for heading in (
        "Architecture and execution flow",
        "Reproduction commands",
        "Baseline benchmark table",
        "Profiling findings",
        "Confirmed root causes",
        "Rejected hypotheses",
        "Failures, timeouts, and OOMs",
        "Limitations",
    ):
        assert f"## {heading}" in first


def test_renderer_labels_noise_without_an_improvement_claim() -> None:
    payload = {
        "schema_version": 1,
        "environment": {},
        "results": [
            {
                **_result_payload(),
                "comparisons": [
                    {
                        "baseline_name": "null_counts",
                        "ratio": 1.06,
                        "classification": "noise",
                    }
                ],
            }
        ],
        "component_baselines": [],
        "hypotheses": {},
        "reproduction_commands": [],
        "limitations": [],
    }

    report = render_report(payload)

    assert "noise" in report
    assert "improved" not in report.lower()


def test_analyze_and_render_cli_write_deterministic_outputs(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "case.json").write_text(json.dumps(_result_payload()), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    assert main(["analyze", "--input", str(input_dir), "--output", str(summary_path)]) == 0
    assert main(["render", "--input", str(summary_path), "--output", str(report_path)]) == 0
    first = report_path.read_text(encoding="utf-8")
    assert main(["render", "--input", str(summary_path), "--output", str(report_path)]) == 0
    assert report_path.read_text(encoding="utf-8") == first


def test_baseline_cli_writes_strict_component_result(tmp_path: Path) -> None:
    assert (
        main(
            [
                "baseline",
                "--rows",
                "30",
                "--widths",
                "narrow",
                "--dataset-types",
                "mixed",
                "--baselines",
                "null_counts",
                "--warmups",
                "0",
                "--repetitions",
                "1",
                "--timeout",
                "60",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    payloads = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(payloads) == 1
    assert payloads[0]["baseline_name"] == "null_counts"
    assert payloads[0]["case"]["backend"] == "pandas-component-baseline"
