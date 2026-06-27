"""The unified enterprise API and the pandas/polars hybrid layer.

:func:`clean_enterprise` is the one call that ties the whole pipeline together:

    core cleaning → value clustering → semantic validation → PII masking

It accepts a pandas *or* polars DataFrame and returns the **same type** (core cleaning runs
in pandas; clustering and masking run natively on whichever type was given). Along the way
it computes a Data Trust Score before and after, records OpenLineage events per stage, and
packages everything into an :class:`EnterpriseResult` with a quality gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..adapters.polars import from_pandas, to_pandas
from ..cleaner import run_pipeline
from ..config import CleanConfig, merge_options
from ..report import CleanReport
from .cleaner import (
    ClusterResult,
    MaskReport,
    ValidationReport,
    mask_dataframe,
    merge_clusters,
    run_semantic_validation,
)
from .config import EnterpriseConfig
from .contracts import (
    DataContract,
    DatasetBaseline,
    DriftReport,
    build_baseline,
    compare_to_baseline,
)
from .entity_resolution import EntityResolutionReport, resolve_entities
from .lineage import LineageTracker
from .metrics import QualityReport, TrustScore, compute_trust_score
from .privacy import KAnonymityReport, PrivacyReport, anonymize, check_k_anonymity

_PRIVACY_STRATEGIES = ("tokenize", "fpe", "surrogate")


@dataclass
class EnterpriseResult:
    """Everything one :func:`clean_enterprise` run produced.

    ``data`` is the cleaned frame in the *same type* as the input. The rest is the audit
    surface: trust scores, the core clean report, clustering/masking/validation reports,
    the lineage tracker, and the combined quality report.
    """

    data: Any
    trust_before: TrustScore
    trust_after: TrustScore
    clean_report: CleanReport
    quality: QualityReport
    lineage: LineageTracker
    cluster_results: list[ClusterResult] = field(default_factory=list)
    mask_report: MaskReport | None = None
    validation_report: ValidationReport | None = None
    fail_under_trust: float | None = None
    #: New enterprise reports (populated only when the matching feature is enabled).
    drift_report: DriftReport | None = None
    privacy_report: PrivacyReport | None = None
    k_anonymity_report: KAnonymityReport | None = None
    entity_resolution_report: EntityResolutionReport | None = None

    @property
    def passed_gate(self) -> bool:
        """True if the trust gate and any contract/drift gate both pass."""
        trust_ok = (
            self.fail_under_trust is None
            or self.trust_after.overall >= self.fail_under_trust
        )
        drift_ok = self.drift_report is None or self.drift_report.passed
        return trust_ok and drift_ok

    @property
    def cells_merged(self) -> int:
        return sum(r.n_cells_merged for r in self.cluster_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed_gate": self.passed_gate,
            "fail_under_trust": self.fail_under_trust,
            "trust_before": self.trust_before.to_dict(),
            "trust_after": self.trust_after.to_dict(),
            "clean_report": self.clean_report.to_dict(),
            "clusters": [r.to_dict() for r in self.cluster_results],
            "masking": self.mask_report.to_dict() if self.mask_report else None,
            "validation": self.validation_report.to_dict() if self.validation_report else None,
            "drift": self.drift_report.to_dict() if self.drift_report else None,
            "privacy": self.privacy_report.to_dict() if self.privacy_report else None,
            "k_anonymity": (
                self.k_anonymity_report.to_dict() if self.k_anonymity_report else None
            ),
            "entity_resolution": (
                self.entity_resolution_report.to_dict()
                if self.entity_resolution_report
                else None
            ),
            "lineage": self.lineage.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        lines = [
            f"freshdata enterprise — trust {self.trust_before.overall:.1f} → "
            f"{self.trust_after.overall:.1f} (grade {self.trust_after.grade})"
        ]
        if self.cluster_results:
            n_clusters = sum(r.n_clusters for r in self.cluster_results)
            lines.append(
                f"  clustered: {self.cells_merged} cell(s) merged in {n_clusters} group(s)"
            )
        if self.mask_report:
            lines.append(
                f"  masked: {self.mask_report.total_cells_masked} cell(s) across "
                f"{len(self.mask_report.columns)} column(s)"
            )
        if self.validation_report:
            lines.append(f"  validation: {self.validation_report.n_invalid_total} invalid cell(s)")
        if self.privacy_report:
            lines.append(
                f"  privacy: {self.privacy_report.cells_changed} cell(s) anonymized, "
                f"{self.privacy_report.entities_found} entit(y/ies) detected"
            )
        if self.k_anonymity_report:
            lines.append(f"  {self.k_anonymity_report.summary()}")
        if self.entity_resolution_report:
            lines.append(f"  {self.entity_resolution_report.summary()}")
        if self.drift_report:
            verdict = "PASS" if self.drift_report.passed else "FAIL"
            lines.append(
                f"  drift: {verdict} ({self.drift_report.n_errors} error(s), "
                f"{self.drift_report.n_warnings} warning(s))"
            )
        if self.fail_under_trust is not None:
            verdict = "PASS" if self.passed_gate else "FAIL"
            lines.append(f"  gate: {verdict} (threshold {self.fail_under_trust:.1f})")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        parts = [self.quality.to_markdown()]
        if self.cluster_results:
            parts.append("\n## Clustering\n")
            for result in self.cluster_results:
                parts.append(
                    f"- `{result.column}` ({result.method}): {result.n_clusters} cluster(s), "
                    f"{result.n_cells_merged} cell(s) merged"
                )
        if self.mask_report and self.mask_report.columns:
            cols = ", ".join(f"`{c}`→{s}" for c, s in self.mask_report.columns.items())
            parts.append(
                f"\n## PII masking\n\n- {self.mask_report.total_cells_masked} cell(s): {cols}"
            )
        if self.validation_report and self.validation_report.columns:
            parts.append("\n## Semantic validation\n")
            for name, cv in self.validation_report.columns.items():
                parts.append(
                    f"- `{name}` ({cv.validator}): {cv.n_invalid} invalid / {cv.n_checked}"
                )
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"<EnterpriseResult trust={self.trust_after.overall:.1f} "
            f"grade={self.trust_after.grade} gate={'pass' if self.passed_gate else 'fail'}>"
        )


def clean_enterprise(
    df: Any,
    *,
    clean_config: CleanConfig | None = None,
    enterprise: EnterpriseConfig | None = None,
    actor: str | None = None,
    baseline: DatasetBaseline | None = None,
    contract: DataContract | None = None,
    **clean_options: object,
) -> EnterpriseResult:
    """Run the full enterprise pipeline on *df* (pandas or polars).

    Stages: core cleaning (pandas engine) → value clustering → semantic validation →
    PII masking. The returned :class:`EnterpriseResult` carries the cleaned frame (same
    type as the input) plus the complete audit trail and a trust-score gate.

    ``**clean_options`` are forwarded to :class:`~freshdata.CleanConfig` (e.g.
    ``strategy="aggressive"``); unknown names raise :class:`TypeError`.
    """
    ec = enterprise or EnterpriseConfig()
    cc = merge_options(clean_config, **clean_options)
    who = actor or ec.actor or ec.lineage.actor
    tracker = LineageTracker(ec.lineage)

    def track(rule: str, before: Any, after: Any, count: int, description: str) -> None:
        if ec.enable_lineage:
            tracker.record(rule, before, after, who=who, count=count, description=description)

    trust_before = compute_trust_score(df, weights=ec.trust_weights, config=cc)

    frame = to_pandas(df)
    cleaned, clean_report = run_pipeline(frame, cc)
    track("core_clean", frame, cleaned, clean_report.cells_changed,
          "representation repair + decision engine")

    # Hand back to the input's native type; clustering/masking run natively on it.
    work = from_pandas(cleaned, df)

    cluster_results: list[ClusterResult] = []
    if ec.enable_clustering and ec.clustering is not None:
        before = work
        work, cluster_results = merge_clusters(work, config=ec.clustering)
        merged = sum(r.n_cells_merged for r in cluster_results)
        track("cluster_merge", before, work, merged, f"merged {merged} variant cell(s)")

    validation_report: ValidationReport | None = None
    if ec.enable_validation and ec.semantic:
        validation_report = run_semantic_validation(work, ec.semantic)

    mask_report: MaskReport | None = None
    privacy_report: PrivacyReport | None = None
    use_privacy = ec.enable_privacy_detection and ec.privacy is not None
    new_strategies = any(r.strategy in _PRIVACY_STRATEGIES for r in ec.masking)
    if ec.enable_masking and (ec.masking or use_privacy):
        before = work
        if use_privacy or new_strategies:
            # Route through the privacy engine for tokenize/fpe/surrogate or
            # detection-driven anonymization; produces a PrivacyReport.
            work, privacy_report = anonymize(
                work,
                rules=ec.masking,
                detection_config=ec.privacy if use_privacy else None,
            )
            track("pii_mask", before, work, privacy_report.cells_changed,
                  f"anonymized {privacy_report.cells_changed} cell(s)")
        elif ec.masking:
            work, mask_report = mask_dataframe(work, ec.masking)
            track("pii_mask", before, work, mask_report.total_cells_masked,
                  f"masked {mask_report.total_cells_masked} cell(s)")

    k_anonymity_report: KAnonymityReport | None = None
    if ec.k_anonymity is not None and ec.k_anonymity.enabled and ec.k_anonymity.quasi_identifiers:
        k_anonymity_report = check_k_anonymity(
            work,
            list(ec.k_anonymity.quasi_identifiers),
            k=ec.k_anonymity.k,
            max_report_groups=ec.k_anonymity.max_report_groups,
        )

    # Entity resolution runs after cleaning/masking but before final trust scoring.
    # It reports clusters without mutating ``work`` (schema/row count stay stable).
    entity_resolution_report: EntityResolutionReport | None = None
    if ec.enable_entity_resolution and ec.entity_resolution is not None:
        _resolved, entity_resolution_report = resolve_entities(
            work, config=ec.entity_resolution, return_report=True
        )

    trust_after = compute_trust_score(work, weights=ec.trust_weights, config=cc)

    drift_report: DriftReport | None = None
    if ec.enable_contracts and (baseline is not None or contract is not None):
        base = baseline if baseline is not None else build_baseline(work, name="_inline")
        drift_report = compare_to_baseline(
            work,
            base,
            contract=contract,
            drift_config=ec.drift,
            trust_score=trust_after.overall,
        )
    quality = QualityReport(
        trust_before=trust_before,
        trust_after=trust_after,
        clean_report=clean_report,
        actor=who or "unknown",
    )
    return EnterpriseResult(
        data=work,
        trust_before=trust_before,
        trust_after=trust_after,
        clean_report=clean_report,
        quality=quality,
        lineage=tracker,
        cluster_results=cluster_results,
        mask_report=mask_report,
        validation_report=validation_report,
        fail_under_trust=ec.fail_under_trust,
        drift_report=drift_report,
        privacy_report=privacy_report,
        k_anonymity_report=k_anonymity_report,
        entity_resolution_report=entity_resolution_report,
    )
