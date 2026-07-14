"""Manifest of FreshData's public decision, sink, and caller-directed surfaces."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import freshdata as fd
from freshdata.domains.registry import available as available_domains


class SurfaceClass(str, Enum):
    DECISION = "decision"
    SINK = "sink"
    EXPLICIT_TRANSFORM = "explicit-transform"
    DATA_MODEL = "data-model"
    CONFIGURATION = "configuration"
    REGISTRATION = "registration"
    OUT_OF_SCOPE = "out-of-scope-with-reason"


@dataclass(frozen=True)
class SurfaceSpec:
    name: str
    version: int
    classification: SurfaceClass
    adapter: str | None
    mutates: bool
    deterministic: bool
    backend_parity: bool
    rationale: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("surface name must not be empty")
        if self.version != 1:
            raise ValueError("unsupported surface spec version")
        try:
            classification = SurfaceClass(self.classification)
        except ValueError as exc:
            raise ValueError(f"unknown surface classification: {self.classification!r}") from exc
        object.__setattr__(self, "classification", classification)
        if classification in {SurfaceClass.DECISION, SurfaceClass.SINK} and not self.adapter:
            raise ValueError(f"{classification.value} surface {self.name!r} requires an adapter")
        if not self.rationale:
            raise ValueError(f"surface {self.name!r} requires a rationale")


_CALLER_DIRECTED = {
    "fill_missing",
    "remove_outliers",
    "resolve_duplicates",
    "group_aggregate",
    "set_display",
    "get_display",
    "reset_display",
    "tokenize_value",
    "detokenize_value",
    "detokenize_series",
    "make_vault",
    "vault_metadata",
    "anonymize",
    "clean_text",
    "clean_text_value",
    "lint_text_encoding",
    "mask_dataframe",
}
_REGISTRATION = {
    "register_backend",
    "register_comparator",
    "register_expert",
    "register_exporter",
    "register_validator",
    "registered_plugins",
    "available_packs",
}
_SINKS = {
    "build_exception_table",
    "build_quality_report",
    "export",
    "export_dbt_tests",
    "export_gx_suite",
    "export_quality_ops",
    "export_review_queue",
    "generate_compliance_report",
    "insight_report",
    "stakeholder_summary",
    "trust_gate_report",
    "save_baseline",
}
_DECISIONS = {
    "apply",
    "apply_field_policy",
    "apply_plan",
    "apply_privacy_policy",
    "apply_review_decisions",
    "build_review_queue",
    "cdc_profile",
    "clean",
    "clean_csv",
    "clean_domain_file",
    "clean_enterprise",
    "clean_timeseries",
    "cluster_column",
    "compare_clean",
    "compare_plans",
    "compile_context",
    "compute_trust_score",
    "detect_outliers",
    "detect_pii",
    "diff_schema",
    "enforce_contract",
    "evaluate_quality_debt",
    "explain_clean",
    "infer_roles",
    "learn",
    "learn_cleaning_memory",
    "link",
    "link_entities",
    "monitor_contract",
    "parse_domain",
    "pipeline",
    "pipeline_clean",
    "plan",
    "profile",
    "recalibrate_weights",
    "redaction_columns",
    "resolve_entities",
    "run_semantic_validation",
    "run_suite",
    "suggest_join_keys",
    "suggest_plan",
    "validate",
    "validate_fields",
    "analyze_dataset",
    "check_k_anonymity",
    "classify_columns",
    "compare_to_baseline",
    "merge_clusters",
    "merge_entities",
    "schema_of",
    "load_review_decisions",
}
_CONFIG_FUNCTIONS = {
    "education_template",
    "build_baseline",
    "get_template",
    "healthcare_template",
    "load_baseline",
    "load_cleaning_memory",
    "load_compliance_pack",
    "load_privacy_policy",
    "load_profile",
    "media_template",
    "retail_template",
    "save_profile",
}
_OUT_OF_SCOPE = {"enterprise", "models", "testing", "__version__"}


def cli_commands() -> tuple[str, ...]:
    """Return top-level and nested enterprise CLI command paths."""

    from freshdata.enterprise.cli import build_parser

    parser = build_parser()
    result: set[str] = set()

    def visit(current: argparse.ArgumentParser, prefix: str = "") -> None:
        for action in current._actions:
            if not isinstance(action, argparse._SubParsersAction):
                continue
            choices = action.choices
            for command, child in choices.items():
                path = f"{prefix} {command}".strip()
                result.add(path)
                if isinstance(child, argparse.ArgumentParser):
                    visit(child, path)

    visit(parser)
    return tuple(sorted(result))


def discover_public_names() -> tuple[str, ...]:
    """Discover all names and registered command/domain surfaces in this checkout."""

    names = set(fd.__dir__())
    from freshdata.experimental import ai_copilot

    names.update(ai_copilot.__all__)
    names.update(f"domain:{name}" for name in available_domains())
    names.update(f"cli:{name}" for name in cli_commands())
    return tuple(sorted(names))


def _classify(name: str) -> tuple[SurfaceClass, str | None, bool, bool, bool, str]:
    if name.startswith("domain:"):
        return (
            SurfaceClass.DECISION,
            "generic",
            False,
            True,
            True,
            "bundled domain validator decision surface",
        )
    if name.startswith("cli:"):
        return (
            SurfaceClass.DECISION,
            "generic",
            False,
            True,
            False,
            "enterprise CLI decision/report command",
        )
    if name in _CALLER_DIRECTED:
        return (
            SurfaceClass.EXPLICIT_TRANSFORM,
            None,
            name not in {"get_display", "reset_display", "vault_metadata"},
            True,
            False,
            "caller-directed explicit transform; no inferred disposition credit",
        )
    if name in _REGISTRATION:
        return (
            SurfaceClass.REGISTRATION,
            None,
            True,
            True,
            False,
            "plugin registration or registry inspection",
        )
    if name in _CONFIG_FUNCTIONS:
        return (
            SurfaceClass.CONFIGURATION,
            None,
            False,
            True,
            False,
            "configuration/template loading surface",
        )
    if name in _SINKS:
        return (
            SurfaceClass.SINK,
            "generic",
            False,
            True,
            False,
            "report, audit, or export sink",
        )
    if name in _DECISIONS:
        return (
            SurfaceClass.DECISION,
            "generic",
            name.startswith(("clean", "apply", "run_")),
            True,
            True,
            "public FreshData decision surface",
        )
    if name in _OUT_OF_SCOPE:
        return (
            SurfaceClass.OUT_OF_SCOPE,
            None,
            False,
            True,
            False,
            "namespace or metadata surface; no cell disposition",
        )
    if name.startswith("__"):
        return SurfaceClass.CONFIGURATION, None, False, True, False, "package metadata"
    if name.endswith(("Config", "Policy", "Rule", "Spec", "Weights")) or name in {
        "Action",
        "Jurisdiction",
        "CompliancePack",
    }:
        return (
            SurfaceClass.CONFIGURATION,
            None,
            False,
            True,
            False,
            "configuration or policy model",
        )
    if name[:1].isupper():
        return SurfaceClass.DATA_MODEL, None, False, True, False, "public data/report model"
    raise ValueError(f"unclassified FreshData public surface: {name}")


def build_manifest(names: Iterable[str] | None = None) -> tuple[SurfaceSpec, ...]:
    """Build an immutable manifest from the currently discovered public surfaces."""

    selected = discover_public_names() if names is None else tuple(sorted(set(names)))
    specs: list[SurfaceSpec] = []
    for name in selected:
        classification, adapter, mutates, deterministic, parity, rationale = _classify(name)
        specs.append(
            SurfaceSpec(
                name=name,
                version=1,
                classification=classification,
                adapter=adapter,
                mutates=mutates,
                deterministic=deterministic,
                backend_parity=parity,
                rationale=rationale,
            )
        )
    validate_manifest(specs)
    return tuple(specs)


def validate_manifest(manifest: Iterable[SurfaceSpec]) -> None:
    """Validate uniqueness, classifications, and adapter requirements."""

    specs = tuple(manifest)
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("surface manifest contains duplicate names")
    from .surfaces import adapters

    registered = adapters()
    for spec in specs:
        if not isinstance(spec, SurfaceSpec):
            raise TypeError("surface manifest entries must be SurfaceSpec instances")
        # Re-run constructor checks for objects supplied by callers or tests.
        SurfaceSpec(**{field: getattr(spec, field) for field in SurfaceSpec.__dataclass_fields__})
        if (
            spec.classification in {SurfaceClass.DECISION, SurfaceClass.SINK}
            and spec.adapter not in registered
        ):
            raise ValueError(f"surface {spec.name!r} references unknown adapter {spec.adapter!r}")


SURFACE_MANIFEST = build_manifest()
PUBLIC_SURFACES = SURFACE_MANIFEST

__all__ = [
    "PUBLIC_SURFACES",
    "SURFACE_MANIFEST",
    "SurfaceClass",
    "SurfaceSpec",
    "build_manifest",
    "cli_commands",
    "discover_public_names",
    "validate_manifest",
]
