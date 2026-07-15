from __future__ import annotations

from dataclasses import replace

from benchmarks.truthbench.inventory import (
    SurfaceClass,
    build_manifest,
    cli_commands,
    validate_manifest,
)
from benchmarks.truthbench.surfaces import GenericSurfaceAdapter, adapter_for

import freshdata as fd
from freshdata.domains.registry import available as available_domains
from freshdata.experimental import ai_copilot


def test_manifest_covers_public_exports_and_registered_domains() -> None:
    manifest = build_manifest()
    names = {spec.name for spec in manifest}
    assert set(fd.__dir__()) <= names
    assert set(ai_copilot.__all__) <= names
    assert {f"domain:{name}" for name in available_domains()} <= names
    assert {f"cli:{name}" for name in cli_commands()} <= names
    validate_manifest(manifest)


def test_decision_and_sink_surfaces_have_adapters() -> None:
    for spec in build_manifest():
        if spec.classification in {SurfaceClass.DECISION, SurfaceClass.SINK}:
            assert spec.adapter


def test_no_decision_or_sink_surface_uses_the_placeholder_adapter() -> None:
    """Release credit must come from a concrete behavioral adapter: the
    echoing GenericSurfaceAdapter can never back a decision or sink surface."""
    offenders = []
    for spec in build_manifest():
        if spec.classification not in {SurfaceClass.DECISION, SurfaceClass.SINK}:
            continue
        adapter = adapter_for(spec.adapter)
        if isinstance(adapter, GenericSurfaceAdapter):
            offenders.append(spec.name)
    assert not offenders, (
        "decision/sink surfaces still mapped to the placeholder adapter: "
        f"{sorted(offenders)}"
    )


def test_caller_directed_primitives_are_explicit_transforms() -> None:
    manifest = {spec.name: spec for spec in build_manifest()}
    for name in (
        "fill_missing",
        "remove_outliers",
        "resolve_duplicates",
        "group_aggregate",
        "set_display",
        "get_display",
        "reset_display",
        "tokenize_value",
        "detokenize_value",
    ):
        assert manifest[name].classification is SurfaceClass.EXPLICIT_TRANSFORM
        assert "caller-directed" in manifest[name].rationale


def test_manifest_rejects_unclassified_or_adapterless_specs() -> None:
    original = build_manifest()[0]
    try:
        validate_manifest((replace(original, classification="unknown"),))
    except ValueError as exc:
        assert "classification" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("invalid classification was accepted")
