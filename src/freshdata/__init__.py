"""freshdata — fast, safe, automatic data cleaning for real-world tabular data.

>>> import freshdata as fd
>>> cleaned = fd.clean(df)
>>> cleaned, report = fd.clean(df, return_report=True)
>>> print(fd.profile(df))

Design principles
-----------------
- **Real cleaning, real rules.** ``strategy="balanced"`` (default) runs an
  accuracy-first decision engine: every column is profiled (missing ratio, skewness,
  cardinality, inferred role) and threshold rules decide whether to impute,
  preserve, flag, or deliberately leave untouched. Use ``strategy="aggressive"``
  for zero-NaN scrubbing (KNN, column drops, capping). ``strategy="auto"`` is
  deprecated (alias for ``aggressive``).
- **Everything is reported.** Each decision is recorded with the column, the
  affected count, a rationale, a risk level, and a confidence score; the
  report also carries warnings and manual-review recommendations.
- **Never mutates input** (unless ``preserve_original=False``). ``clean``
  returns a new frame; profiling is read-only.
- **Fast by construction.** Vectorized pandas operations only, with
  sample-based pre-screening so type inference stays cheap on large frames.
"""

from .api import (
    apply_plan,
    clean,
    clean_csv,
    clean_domain_file,
    clean_timeseries,
    compile_context,
    infer_roles,
    parse_domain,
    profile,
    suggest_plan,
    validate,
)
from .cdc import CDCDefect, CDCReport, cdc_profile
from .cleaner import Cleaner

# Compliance report generators (additive — Phase 1 roadmap). Light import:
# only stdlib + pandas at load; the enterprise layer is touched lazily at call time.
from .compliance import ComplianceBundle, ComplianceConfig, generate_compliance_report
from .config import CleanConfig
from .context import ColumnConstraint, ContextPolicy, PolicyError
from .execution import EngineConfig
from .explain import ExplainReport, explain_clean
from .findings import FindingList, QualityFinding
from .guard import ProtectedColumnError
from .memory import (
    CleaningMemory,
    learn_cleaning_memory,
    load_cleaning_memory,
)
from .plan import CleanPlan, ColumnPlan, compare_clean, compare_plans
from .profile import ColumnProfile, Profile
from .quality import QualityDebtGate, evaluate_quality_debt
from .repairplan import (
    FrameSignature,
    PlanDriftError,
    PlannedAction,
    RepairPlan,
    RepairProposal,
)
from .report import Action, CleanReport
from .stakeholder import StakeholderSummary, stakeholder_summary
from .streaming import (
    StreamingCleanConfig,
    StreamingCleaner,
    StreamingState,
    TimeSeriesCleanConfig,
)
from .textlint import TextIssue, TextLintReport, lint_text_encoding

__version__ = "1.0.0"

__all__ = [
    "Action",
    "CDCDefect",
    "CDCReport",
    "CleanConfig",
    "CleanPlan",
    "CleanReport",
    "Cleaner",
    "CleaningMemory",
    "ColumnConstraint",
    "ColumnPlan",
    "ColumnProfile",
    "ContextPolicy",
    "FrameSignature",
    "PlanDriftError",
    "PlannedAction",
    "ProtectedColumnError",
    "RepairPlan",
    "RepairProposal",
    "FindingList",
    "PolicyError",
    "ComplianceBundle",
    "ComplianceConfig",
    "EngineConfig",
    "ExplainReport",
    "Profile",
    "QualityDebtGate",
    "QualityFinding",
    "StakeholderSummary",
    "StreamingCleanConfig",
    "StreamingCleaner",
    "StreamingState",
    "TextIssue",
    "TextLintReport",
    "TimeSeriesCleanConfig",
    "__version__",
    "cdc_profile",
    "apply_plan",
    "clean",
    "clean_csv",
    "clean_domain_file",
    "clean_timeseries",
    "compare_clean",
    "compare_plans",
    "compile_context",
    "evaluate_quality_debt",
    "explain_clean",
    "generate_compliance_report",
    "infer_roles",
    "learn_cleaning_memory",
    "lint_text_encoding",
    "load_cleaning_memory",
    "parse_domain",
    "profile",
    "stakeholder_summary",
    "suggest_plan",
    "validate",
]

#: Names served lazily from :mod:`freshdata.enterprise` via PEP 562, so the optional
#: enterprise layer (and its optional deps) is only imported when actually used. These are
#: deliberately *not* in ``__all__`` to keep ``import freshdata`` and ``import *`` light.
_ENTERPRISE_EXPORTS = frozenset(
    {
        "clean_enterprise",
        "EnterpriseResult",
        "EnterpriseConfig",
        "MaskingRule",
        "ClusterConfig",
        "TrustScoreWeights",
        "LineageConfig",
        "SemanticValidatorConfig",
        "TrustScore",
        "QualityReport",
        "compute_trust_score",
        "build_quality_report",
        "LineageTracker",
        "schema_of",
        "merge_clusters",
        "cluster_column",
        "mask_dataframe",
        "run_semantic_validation",
        # contracts / drift monitoring
        "DriftConfig",
        "ColumnContract",
        "DataContract",
        "ColumnBaseline",
        "DatasetBaseline",
        "DriftFinding",
        "DriftReport",
        "build_baseline",
        "save_baseline",
        "load_baseline",
        "compare_to_baseline",
        "monitor_contract",
        "diff_schema",
        "ContractViolation",
        # privacy / anonymization
        "detect_pii",
        "anonymize",
        "check_k_anonymity",
        "tokenize_value",
        "detokenize_value",
        "PIIEntity",
        "PIIScanReport",
        "MaskingEvent",
        "PrivacyReport",
        "KAnonymityReport",
        "InMemoryTokenVault",
        "JsonTokenVault",
        "SqliteTokenVault",
        "make_vault",
        "vault_metadata",
        # privacy policy engine
        "PrivacyPolicy",
        "PrivacyRule",
        "CompliancePack",
        "Jurisdiction",
        "Action",
        "apply_privacy_policy",
        "load_privacy_policy",
        "load_compliance_pack",
        "available_packs",
        "classify_columns",
        "detokenize_series",
        # dirty-join assistant
        "suggest_join_keys",
        "JoinKeyReport",
        "JoinCandidate",
        # entity resolution
        "resolve_entities",
        "link_entities",
        "link",
        "MatchPair",
        "EntityCluster",
        "EntityResolutionReport",
        "EntityResolutionConfig",
        "ComparisonLevel",
        "BlockingRule",
        "FieldExplanation",
        "redaction_columns",
        "ReviewQueueConfig",
        "ReviewItem",
        "ReviewDecision",
        "ReviewQueueReport",
        "build_review_queue",
        "export_review_queue",
        "load_review_decisions",
        "apply_review_decisions",
        "recalibrate_weights",
        "GoldenRecordPolicy",
        "merge_entities",
        "DomainTemplate",
        "get_template",
        "education_template",
        "healthcare_template",
        "retail_template",
        "media_template",
    }
)

#: Quality-ops exporters served lazily from :mod:`freshdata.integrations`. Like the
#: enterprise exports they are kept out of ``__all__`` so ``import freshdata`` stays
#: light — importing the integrations layer pulls in the enterprise layer and its
#: optional deps, which should only happen when an exporter is actually used.
_INTEGRATION_EXPORTS = {
    "export_quality_ops": "freshdata.integrations.quality_ops",
    "QualityOpsResult": "freshdata.integrations.quality_ops",
    "export_dbt_tests": "freshdata.integrations.dbt",
    "export_gx_suite": "freshdata.integrations.great_expectations",
    "build_exception_table": "freshdata.integrations.exceptions",
}


def __getattr__(name: str) -> object:
    """Lazily resolve the ``enterprise`` submodule and its key exports (PEP 562)."""
    if name == "enterprise":
        import importlib

        return importlib.import_module("freshdata.enterprise")
    if name == "models":
        # Local model registry/runtime (`fd.models.status()` etc). Stdlib-only
        # at import; the [semantic] extra is resolved lazily at encode time.
        import importlib

        return importlib.import_module("freshdata.models")
    if name in _ENTERPRISE_EXPORTS:
        import importlib

        return getattr(importlib.import_module("freshdata.enterprise"), name)
    if name in _INTEGRATION_EXPORTS:
        import importlib

        return getattr(importlib.import_module(_INTEGRATION_EXPORTS[name]), name)
    raise AttributeError(f"module 'freshdata' has no attribute {name!r}")


def __dir__() -> list:
    return sorted([*__all__, "enterprise", "models", *_ENTERPRISE_EXPORTS, *_INTEGRATION_EXPORTS])
