from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.truthbench.fixtures import build_fixture
from benchmarks.truthbench.surfaces import (
    SurfaceAdapter,
    SurfaceObservation,
    adapter_for,
    register_adapter,
)
from benchmarks.truthbench.surfaces.cleaning import CleaningAdapter
from benchmarks.truthbench.surfaces.copilot import CopilotAdapter
from benchmarks.truthbench.surfaces.privacy import PrivacyAdapter
from benchmarks.truthbench.surfaces.reporting import ReportingAdapter
from benchmarks.truthbench.surfaces.validation import ValidationAdapter

import freshdata as fd


def test_observation_carries_all_release_evidence() -> None:
    observation = SurfaceObservation(
        output_frame={"value": 1},
        raw_decisions=("repair",),
        audit_sinks=("audit.json",),
        trust={"before": 0.8, "after": 0.9},
        backend_disclosure={"requested": "pandas", "actual": "pandas"},
        generated_code="print('ok')",
        captured_stdout="ok\n",
        captured_stderr="",
        unexpected_exception=None,
    )
    assert observation.output_frame == {"value": 1}
    assert observation.backend_disclosure["actual"] == "pandas"
    assert observation.unexpected_exception is None


def test_adapter_protocol_is_abstract_and_registry_is_explicit() -> None:
    class DemoAdapter(SurfaceAdapter):
        name = "demo"

        def observe(self, fixture, context):
            return SurfaceObservation(output_frame=fixture)

    register_adapter(DemoAdapter)
    adapter = adapter_for("demo")
    assert isinstance(adapter, DemoAdapter)
    assert adapter.observe({"x": 1}, {}).output_frame == {"x": 1}

    with pytest.raises(TypeError):
        SurfaceAdapter()  # type: ignore[abstract]


def test_cleaning_adapter_observes_public_clean_and_report() -> None:
    frame = pd.DataFrame({"x": [" 1 ", "2"]}, index=["r1", "r2"])
    observation = CleaningAdapter().observe(
        frame,
        {"strip_whitespace": True, "fix_dtypes": False, "strategy": "conservative"},
    )
    assert observation.unexpected_exception is None
    assert observation.output_frame["x"].tolist() == ["1", "2"]
    assert "input_snapshot" in observation.audit_sinks
    assert "report_actions" in observation.raw_decisions


def test_validation_adapter_is_read_only_and_captures_findings() -> None:
    frame = pd.DataFrame({"age": [1, -1]}, index=["r1", "r2"])
    original = frame.copy(deep=True)
    observation = ValidationAdapter().observe(
        frame,
        {"schema": {"age": fd.FieldSpec(semantic_type="integer", min_value=0)}},
    )
    assert observation.unexpected_exception is None
    pd.testing.assert_frame_equal(frame, original)
    assert observation.output_frame.equals(frame)
    assert "findings" in observation.raw_decisions


@pytest.mark.parametrize("adapter", [CleaningAdapter(), ValidationAdapter()])
def test_adapter_setup_exception_redacts_fixture_canary(adapter) -> None:
    canary = "tb.person+setup@example.invalid"

    class BrokenFixture:
        pii_canaries = {"email": canary}

        @property
        def frame(self):
            raise RuntimeError(canary)

    observation = adapter.observe(BrokenFixture(), {})
    assert observation.unexpected_exception is not None
    assert observation.unexpected_exception.type_name == "RuntimeError"
    assert canary not in observation.unexpected_exception.message


def test_domain_validator_does_not_duplicate_domain_keyword() -> None:
    fixture = build_fixture("finance")
    observation = ValidationAdapter().observe(
        fixture, {"operation": "domain_validator", "domain": "finance"}
    )
    assert not (
        observation.unexpected_exception
        and observation.unexpected_exception.type_name == "TypeError"
        and "multiple values for keyword argument 'domain'"
        in observation.unexpected_exception.message
    )


def test_privacy_adapter_observes_detection_anonymization_policy_and_k_anonymity() -> None:
    fixture = build_fixture("crm")
    adapter = PrivacyAdapter()
    for operation in (
        "detect_pii",
        "anonymize",
        "anonymize_default_random",
        "privacy_policy",
        "k_anonymity",
    ):
        observation = adapter.observe(
            fixture,
            {"operation": operation, "quasi_identifiers": ["country"]},
        )
        assert observation.unexpected_exception is None, operation
        assert observation.audit_sinks
        assert observation.backend_disclosure == {"requested": "pandas", "actual": "pandas"}
        assert not adapter.scanner_for(fixture).scan(observation.audit_sinks)
        if operation == "anonymize_default_random":
            assert observation.audit_sinks["randomness"]["default_salt_generated"]


def test_reporting_adapter_captures_reports_rendering_exports_and_trust_controls() -> None:
    fixture = build_fixture("retail")
    adapter = ReportingAdapter()
    observation = adapter.observe(fixture, {"operation": "reports"})
    assert observation.unexpected_exception is None
    assert {"quality", "debt", "insight", "compliance", "stakeholder"} <= set(
        observation.raw_decisions
    )
    assert {"to_dict", "to_frame", "to_findings", "json", "markdown", "html", "plain"} <= set(
        observation.audit_sinks["rendered"]
    )
    assert {"pristine", "adversarial", "cleaned", "destructive"} <= set(observation.trust)
    assert observation.trust["destructive"] <= observation.trust["pristine"]
    assert not adapter.scanner_for(fixture).scan(observation.audit_sinks)


def test_reporting_adapter_captures_cli_and_quality_ops_export_sinks() -> None:
    fixture = build_fixture("finance")
    observation = ReportingAdapter().observe(fixture, {"operation": "exports"})
    assert observation.unexpected_exception is None
    assert {"quality_ops", "dbt", "great_expectations", "exceptions", "cli"} <= set(
        observation.audit_sinks["exports"]
    )


def test_copilot_adapter_is_provider_free_and_collects_all_public_report_sinks() -> None:
    fixture = build_fixture("crm")
    observation = CopilotAdapter().observe(fixture, {})
    assert observation.unexpected_exception is None
    assert observation.generated_code
    expected_sinks = {
        "prompt",
        "model_context",
        "recommended_code",
        "audit",
        "narrative",
        "rendered",
    }
    assert expected_sinks <= set(observation.audit_sinks)
    assert observation.audit_sinks["narrative"] is None
    assert not CopilotAdapter().scanner_for(fixture).scan(observation.audit_sinks)
