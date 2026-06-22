"""Normalize a real :class:`freshdata.CleanReport` into compliance concepts.

The framework generators are written against an idealized action schema
(``action_type``, ``record_action_type``, ``original_value_class``, masked
columns, per-column roles, a 0–100 trust score). The real :class:`Action`
carries only ``step``/``column``/``description``/``count``/``rationale``/
``risk``/``confidence``, and PII masking + the 0–100 trust score live in the
optional enterprise layer. :func:`build_context` bridges that gap *additively*:
it reads what a core report offers and enriches it when a source DataFrame
and/or an ``EnterpriseResult`` are supplied, degrading gracefully otherwise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from freshdata.api import infer_roles

from ._base import ComplianceConfig, new_session_id, utc_now

logger = logging.getLogger("freshdata.compliance")

# --------------------------------------------------------------------------- #
# Mapping tables (real engine `step` vocabulary -> spec concepts)              #
# --------------------------------------------------------------------------- #
#: Real ``Action.step`` values emitted by the engine, mapped to the spec's
#: normalized ``action_type`` vocabulary.
STEP_TO_ACTION_TYPE: dict[str, str] = {
    "impute": "impute",
    "mean": "impute",
    "median": "impute",
    "mode": "impute",
    "knn": "impute",
    "linear": "impute",
    "time_fill": "impute",
    "partner_median": "impute",
    "drop": "drop_col",
    "drop_constant_columns": "drop_col",
    "drop_empty_columns": "drop_col",
    "drop_empty_rows": "drop_row",
    "drop_duplicates": "drop_row",
    "fix_dtypes": "type_repair",
    "optimize_memory": "type_repair",
    "normalize_sentinels": "sentinel_normalize",
    "sentinel": "sentinel_normalize",
    "strip_whitespace": "strip_whitespace",
    "outliers": "outlier_cap",
    "column_names": "rename",
    "preserve": "flag",
}

#: Normalized ``action_type`` -> 21 CFR record action class.
ACTION_TYPE_TO_RECORD: dict[str, str] = {
    "impute": "MODIFY",
    "drop_row": "DELETE",
    "drop_col": "DELETE",
    "flag": "MODIFY",
    "type_repair": "MODIFY",
    "rename": "MODIFY",
    "strip_whitespace": "MODIFY",
    "sentinel_normalize": "MODIFY",
    "pii_mask": "MODIFY",
    "fuzzy_cluster": "MODIFY",
    "outlier_cap": "MODIFY",
}

#: Real ``step`` -> the *basis* on which prior information is (or is not)
#: preserved. ``not_captured`` means the engine overwrote an existing value
#: without retaining a pre-image (the only class that fails the CFR gate).
STEP_TO_ORIGINAL_BASIS: dict[str, str] = {
    "impute": "was_missing",
    "mean": "was_missing",
    "median": "was_missing",
    "mode": "was_missing",
    "knn": "was_missing",
    "linear": "was_missing",
    "time_fill": "was_missing",
    "partner_median": "was_missing",
    "drop": "removed_record",
    "drop_constant_columns": "removed_record",
    "drop_empty_columns": "removed_record",
    "drop_empty_rows": "removed_record",
    "drop_duplicates": "removed_record",
    "column_names": "structure_only",
    "optimize_memory": "structure_only",
    "fix_dtypes": "structure_only",  # lossless representation repair (e.g. "25" -> 25)
    "normalize_sentinels": "normalized",  # junk placeholder ("N/A", "-") -> canonical NaN
    "sentinel": "normalized",
    "strip_whitespace": "normalized",  # cosmetic trim of surrounding whitespace
    "outliers": "not_captured",  # genuinely overwrites a real value (capping/winsorize)
    "preserve": "unchanged",
}

#: Bases that do NOT obscure previously recorded information (CFR-safe). Anything
#: not listed here (i.e. ``not_captured``) lossily overwrote a recorded value.
NON_OBSCURING_BASES = frozenset(
    {
        "was_missing",
        "removed_record",
        "structure_only",
        "normalized",
        "intentional_mask",
        "unchanged",
    }
)


def classify_step(step: str, count: int) -> tuple[str, str]:
    """Return ``(action_type, original_basis)`` for a real engine ``step``.

    ``step == "missing"`` is the decision engine's missing-value handler: it
    either fills absent cells (``count > 0`` → imputation, prior value was
    absent) or deliberately leaves them in place (``count == 0`` → preserve).
    """
    if step == "missing":
        if count > 0:
            return "impute", "was_missing"
        return "flag", "unchanged"
    action_type = STEP_TO_ACTION_TYPE.get(step, step or "unknown")
    basis = STEP_TO_ORIGINAL_BASIS.get(step, "not_captured")
    return action_type, basis


@dataclass(frozen=True)
class NormalizedAction:
    """One transformation, expressed in the schema the generators expect."""

    action_type: str
    record_action_type: str  # CREATE | MODIFY | DELETE
    column: str | None
    column_role: str
    rationale: str
    description: str
    risk: str
    confidence: float
    row_count: int
    #: One of: was_missing, removed_record, structure_only, normalized,
    #: intentional_mask, unchanged, not_captured (see STEP_TO_ORIGINAL_BASIS).
    original_basis: str
    source: str  # core | mask | cluster

    @property
    def original_captured(self) -> bool:
        """Whether prior information is preserved (vs. silently overwritten)."""
        return self.original_basis != "not_captured"

    @property
    def original_value_class(self) -> str:
        """21 CFR coarse class: ``preserved`` unless a pre-image was lost."""
        return "preserved" if self.original_captured else "not_captured"


@dataclass
class ComplianceContext:
    """Everything the framework generators read, built once per call."""

    session_id: str
    timestamp: str
    actions: list[NormalizedAction]
    trust_score: float | None
    trust_score_available: bool
    masked_columns: set[str]
    roles: dict[str, str]
    missing_ratio: dict[str, float]
    domain_sensitive_columns: set[str]
    all_columns: list[str]
    core_action_count: int
    clean_report: Any
    input_dataframe_hash: str | None = None


# --------------------------------------------------------------------------- #
# Builder                                                                      #
# --------------------------------------------------------------------------- #
def _looks_like_enterprise_result(obj: Any) -> bool:
    return obj is not None and hasattr(obj, "clean_report") and hasattr(obj, "trust_after")


def _roles_from_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, float], set[str]]:
    """Return (role, missing_ratio, domain_sensitive) maps via ``infer_roles``."""
    try:
        rdf = infer_roles(dataframe)
        roles = {str(c): str(r) for c, r in zip(rdf["column"], rdf["role"])}
        missing = {str(c): float(p) / 100.0 for c, p in zip(rdf["column"], rdf["missing_pct"])}
        sensitive = {str(c) for c in rdf.loc[rdf["domain_sensitive"].astype(bool), "column"]}
        return roles, missing, sensitive
    except Exception:  # pragma: no cover - defensive; infer_roles is read-only
        logger.debug("infer_roles failed; proceeding without role enrichment", exc_info=True)
        return {}, {}, set()


def _mask_columns_meta(enterprise_result: Any) -> dict[str, Any]:
    """Return ``{column: strategy}`` from an EnterpriseResult's mask report."""
    if enterprise_result is None:
        return {}
    mask_report = getattr(enterprise_result, "mask_report", None)
    if mask_report is None:
        return {}
    cols = getattr(mask_report, "columns", None)
    if isinstance(cols, dict):
        return {str(k): v for k, v in cols.items()}
    if cols:
        return {str(c): None for c in cols}
    return {}


def build_context(
    report: Any,
    config: ComplianceConfig,
    dataframe: pd.DataFrame | None = None,
    enterprise_result: Any = None,
) -> ComplianceContext:
    """Normalize ``report`` (+ optional context) into a :class:`ComplianceContext`.

    ``report`` may be a :class:`freshdata.CleanReport` or an enterprise result
    (anything exposing ``clean_report`` + ``trust_after``); in the latter case
    its embedded clean report and trust/mask data are used automatically.
    """
    # Resolve the core report and any enterprise context.
    enterprise = enterprise_result
    clean_report = report
    if _looks_like_enterprise_result(report):
        enterprise = report
        clean_report = report.clean_report

    core_actions = list(getattr(clean_report, "actions", []) or [])

    # Per-column enrichment from a source DataFrame (roles + missing ratios).
    roles: dict[str, str] = {}
    missing_ratio: dict[str, float] = {}
    domain_sensitive: set[str] = set()
    if dataframe is not None:
        roles, missing_ratio, domain_sensitive = _roles_from_dataframe(dataframe)

    # Masked columns: caller-declared + anything an EnterpriseResult masked.
    mask_meta = _mask_columns_meta(enterprise)
    masked_columns: set[str] = set(config.masked_columns or []) | set(mask_meta)

    actions: list[NormalizedAction] = []

    # 1) Core engine actions.
    for action in core_actions:
        step = getattr(action, "step", "") or ""
        column = getattr(action, "column", None)
        count = int(getattr(action, "count", 0) or 0)
        action_type, basis = classify_step(step, count)
        actions.append(
            NormalizedAction(
                action_type=action_type,
                record_action_type=ACTION_TYPE_TO_RECORD.get(action_type, "MODIFY"),
                column=column,
                column_role=roles.get(column, "unknown") if column else "table",
                rationale=getattr(action, "rationale", "") or "",
                description=getattr(action, "description", "") or "",
                risk=getattr(action, "risk", "low") or "low",
                confidence=float(getattr(action, "confidence", 1.0)),
                row_count=count,
                original_basis=basis,
                source="core",
            )
        )

    # 2) Synthesized PII-mask actions (enterprise mask report or config).
    for column in sorted(masked_columns):
        strategy = mask_meta.get(column)
        detail = f" using {strategy}" if strategy else ""
        actions.append(
            NormalizedAction(
                action_type="pii_mask",
                record_action_type="MODIFY",
                column=column,
                column_role=roles.get(column, "pii"),
                rationale=f"PII masking applied{detail}",
                description=f"Masked column {column!r}{detail}",
                risk="low",
                confidence=1.0,
                row_count=0,
                original_basis="intentional_mask",
                source="mask",
            )
        )

    # 3) Synthesized fuzzy-cluster actions (enterprise cluster results).
    for cluster in getattr(enterprise, "cluster_results", None) or []:
        column = getattr(cluster, "column", None)
        merged = int(getattr(cluster, "n_cells_merged", 0) or 0)
        n_clusters = getattr(cluster, "n_clusters", None)
        into = f" into {n_clusters} clusters" if n_clusters is not None else ""
        actions.append(
            NormalizedAction(
                action_type="fuzzy_cluster",
                record_action_type="MODIFY",
                column=column,
                column_role=roles.get(column, "unknown") if column else "table",
                rationale=f"Fuzzy clustering merged near-duplicate values{into}",
                description=f"Merged {merged} cell(s) in {column!r}",
                risk="medium",
                confidence=1.0,
                row_count=merged,
                original_basis="not_captured",
                source="cluster",
            )
        )

    # Trust score: EnterpriseResult -> domain score (0–1, scaled) -> config.
    trust_score = _resolve_trust_score(enterprise, clean_report, config)

    return ComplianceContext(
        session_id=new_session_id(),
        timestamp=utc_now(),
        actions=actions,
        trust_score=trust_score,
        trust_score_available=trust_score is not None,
        masked_columns=masked_columns,
        roles=roles,
        missing_ratio=missing_ratio,
        domain_sensitive_columns=domain_sensitive,
        all_columns=_resolve_all_columns(clean_report, dataframe, core_actions, masked_columns),
        core_action_count=len(core_actions),
        clean_report=clean_report,
        input_dataframe_hash=config.input_dataframe_hash,
    )


def _resolve_trust_score(
    enterprise: Any, clean_report: Any, config: ComplianceConfig
) -> float | None:
    trust_after = getattr(enterprise, "trust_after", None)
    overall = getattr(trust_after, "overall", None)
    if overall is not None:
        return float(overall)
    domain_score = getattr(clean_report, "domain_trust_score", None)
    if domain_score is not None:
        return float(domain_score) * 100.0
    if config.trust_score is not None:
        return float(config.trust_score)
    return None


def _resolve_all_columns(
    clean_report: Any,
    dataframe: pd.DataFrame | None,
    core_actions: list[Any],
    masked_columns: set[str],
) -> list[str]:
    if dataframe is not None:
        return [str(c) for c in dataframe.columns]
    columns: set[str] = set(masked_columns)
    for action in core_actions:
        column = getattr(action, "column", None)
        if column:
            columns.add(column)
    for attr in ("columns_dropped", "columns_imputed", "columns_preserved"):
        columns.update(getattr(clean_report, attr, []) or [])
    return sorted(columns)
