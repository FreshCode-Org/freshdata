from __future__ import annotations

import pytest
from benchmarks.truthbench.surfaces import (
    SurfaceAdapter,
    SurfaceObservation,
    adapter_for,
    register_adapter,
)


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
