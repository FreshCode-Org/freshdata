from __future__ import annotations

import benchmarks.truthbench.surfaces.copilot as copilot_module
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
import freshdata.experimental.ai_copilot as ai_copilot_module


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


@pytest.mark.parametrize(
    ("adapter", "context"),
    [
        (PrivacyAdapter(), {"operation": "detect_pii"}),
        (ReportingAdapter(), {"operation": "reports"}),
        (CopilotAdapter(), {}),
    ],
)
def test_task9_adapters_redact_every_public_observation_sink(adapter, context) -> None:
    fixture = build_fixture("crm")
    observation = adapter.observe(fixture, context)
    scanner = adapter.scanner_for(fixture)
    assert observation.unexpected_exception is None
    assert not scanner.scan(observation.output_frame)
    assert not scanner.scan(observation.raw_decisions)
    assert not scanner.scan(observation.audit_sinks)
    assert not scanner.scan(observation.generated_code)
    assert not scanner.scan(observation.captured_stdout)
    assert not scanner.scan(observation.captured_stderr)


@pytest.mark.parametrize(
    ("adapter", "context", "module_name", "attribute"),
    [
        (PrivacyAdapter(), {"operation": "detect_pii"}, "freshdata", "detect_pii"),
        (ReportingAdapter(), {"operation": "reports"}, "freshdata", "clean"),
        (
            CopilotAdapter(),
            {},
            "benchmarks.truthbench.surfaces.copilot",
            "analyze_dataset",
        ),
    ],
)
def test_task9_adapters_redact_captured_stream_canaries(
    adapter, context, module_name, attribute, monkeypatch
) -> None:
    fixture = build_fixture("crm")
    canary = next(iter(fixture.pii_canaries.values()))
    module = __import__(module_name, fromlist=[attribute])
    original = getattr(module, attribute)

    def noisy(*args, **kwargs):
        print(canary)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, attribute, noisy)
    observation = adapter.observe(fixture, context)
    assert observation.unexpected_exception is None
    assert canary not in observation.captured_stdout
    assert not adapter.scanner_for(fixture).scan(observation.captured_stdout)


def test_reporting_trust_uses_distinct_pristine_and_adversarial_fixture_frames(
    monkeypatch,
) -> None:
    # The two frames can legitimately tie on the overall score, so prove the
    # wiring directly: capture exactly which frames the adapter scores.
    fixture = build_fixture("finance")
    scored: list[object] = []
    original = fd.compute_trust_score

    def recording(frame, **kwargs):
        scored.append(frame)
        return original(frame, **kwargs)

    monkeypatch.setattr(fd, "compute_trust_score", recording)
    observation = ReportingAdapter().observe(fixture, {"operation": "reports"})
    assert observation.unexpected_exception is None
    assert any(frame.equals(fixture.pristine) for frame in scored)
    assert any(frame.equals(fixture.frame) for frame in scored)
    assert observation.trust["pristine"] == original(fixture.pristine).overall
    assert observation.trust["adversarial"] == original(fixture.frame).overall


def test_privacy_fixed_secret_is_deterministic_and_default_masking_is_random_and_safe() -> None:
    fixture = build_fixture("crm")
    adapter = PrivacyAdapter()
    fixed_a = adapter.observe(fixture, {"operation": "anonymize"})
    fixed_b = adapter.observe(fixture, {"operation": "anonymize"})
    random_a = adapter.observe(fixture, {"operation": "anonymize_default_random"})
    random_b = adapter.observe(fixture, {"operation": "anonymize_default_random"})
    assert fixed_a.output_frame.equals(fixed_b.output_frame)
    assert not random_a.output_frame.equals(random_b.output_frame)
    for observation in (fixed_a, fixed_b, random_a, random_b):
        assert not adapter.scanner_for(fixture).scan(observation.output_frame)
    assert random_a.audit_sinks["randomness"]["default_salt_generated"]


def test_copilot_adapter_passes_none_to_provider_and_records_actual_prompt(monkeypatch) -> None:
    fixture = build_fixture("crm")
    original = copilot_module.analyze_dataset
    observed: list[object] = []

    def sentinel_network(*args, **kwargs):
        observed.append(kwargs["provider"])
        return original(*args, **kwargs)

    monkeypatch.setattr(copilot_module, "analyze_dataset", sentinel_network)
    observation = CopilotAdapter().observe(fixture, {})
    assert observed == [None]
    assert observation.audit_sinks["prompt_source"].endswith("._build_prompt")
    assert observation.audit_sinks["prompt_digest"]
    assert not CopilotAdapter().scanner_for(fixture).scan(observation.audit_sinks["prompt"])


def test_copilot_adapter_provider_none_never_enters_the_provider_prompt_path(monkeypatch) -> None:
    fixture = build_fixture("crm")
    original_analyze = copilot_module.analyze_dataset
    provider_values: list[object] = []

    def sentinel_provider_prompt(*args, **kwargs):
        raise AssertionError("provider/network prompt path must not run when provider=None")

    def observe_provider_argument(*args, **kwargs):
        provider_values.append(kwargs["provider"])
        return original_analyze(*args, **kwargs)

    # `_build_prompt` is reached inside `analyze_dataset` only for the optional
    # provider hook. The adapter retains its imported prompt builder solely to
    # audit the exact prompt after the provider-free call has completed.
    monkeypatch.setattr(ai_copilot_module, "_build_prompt", sentinel_provider_prompt)
    monkeypatch.setattr(copilot_module, "analyze_dataset", observe_provider_argument)
    observation = CopilotAdapter().observe(fixture, {})
    assert observation.unexpected_exception is None
    assert provider_values == [None]
