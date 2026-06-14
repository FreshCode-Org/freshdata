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
from .lineage import LineageTracker
from .metrics import QualityReport, TrustScore, compute_trust_score


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

    @property
    def passed_gate(self) -> bool:
        """True if no gate is set or the post-clean trust score clears it."""
        return self.fail_under_trust is None or self.trust_after.overall >= self.fail_under_trust

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
    if ec.enable_masking and ec.masking:
        before = work
        work, mask_report = mask_dataframe(work, ec.masking)
        track("pii_mask", before, work, mask_report.total_cells_masked,
              f"masked {mask_report.total_cells_masked} cell(s)")

    trust_after = compute_trust_score(work, weights=ec.trust_weights, config=cc)
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
    )


class FreshDataEnterprise:
    """A reusable, configured enterprise pipeline (mirrors :class:`freshdata.Cleaner`).

    >>> pipe = FreshDataEnterprise(enterprise=ec, strategy="balanced")
    >>> for path in paths:
    ...     result = pipe.run(pd.read_csv(path))
    ...     print(result.summary())
    """

    def __init__(
        self,
        *,
        clean_config: CleanConfig | None = None,
        enterprise: EnterpriseConfig | None = None,
        **clean_options: object,
    ) -> None:
        self.enterprise = enterprise or EnterpriseConfig()
        self._clean_config = clean_config
        self._clean_options = clean_options
        self.result_: EnterpriseResult | None = None

    def run(self, df: Any, *, actor: str | None = None) -> EnterpriseResult:
        result = clean_enterprise(
            df,
            clean_config=self._clean_config,
            enterprise=self.enterprise,
            actor=actor,
            **self._clean_options,
        )
        self.result_ = result
        return result

    def __repr__(self) -> str:
        return f"<FreshDataEnterprise masking={len(self.enterprise.masking)} rules>"
