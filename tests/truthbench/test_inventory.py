from __future__ import annotations

from dataclasses import replace

from benchmarks.truthbench.inventory import (
    SurfaceClass,
    build_manifest,
    cli_commands,
    validate_manifest,
)

import freshdata as fd
from freshdata.domains.registry import available as available_domains


def test_manifest_covers_public_exports_and_registered_domains() -> None:
    manifest = build_manifest()
    names = {spec.name for spec in manifest}
    assert set(fd.__dir__()) <= names
    assert {f"domain:{name}" for name in available_domains()} <= names
    assert {f"cli:{name}" for name in cli_commands()} <= names
    validate_manifest(manifest)


def test_decision_and_sink_surfaces_have_adapters() -> None:
    for spec in build_manifest():
        if spec.classification in {SurfaceClass.DECISION, SurfaceClass.SINK}:
            assert spec.adapter


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
