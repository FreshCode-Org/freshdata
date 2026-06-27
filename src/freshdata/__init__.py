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
    clean,
    clean_csv,
    clean_domain_file,
    infer_roles,
    parse_domain,
    profile,
    suggest_plan,
)
from .cleaner import Cleaner

# Compliance report generators (additive — Phase 1 roadmap). Light import:
# only stdlib + pandas at load; the enterprise layer is touched lazily at call time.
from .compliance import ComplianceBundle, ComplianceConfig, generate_compliance_report
from .config import CleanConfig
from .execution import EngineConfig
from .explain import ExplainReport, explain_clean
from .plan import CleanPlan, ColumnPlan, compare_clean, compare_plans
from .profile import ColumnProfile, Profile
from .report import Action, CleanReport
from .streaming import StreamingCleanConfig, StreamingCleaner, StreamingState

__version__ = "1.0.0"

__all__ = [
    "Action",
    "CleanConfig",
    "CleanPlan",
    "CleanReport",
    "Cleaner",
    "ColumnPlan",
    "ColumnProfile",
    "ComplianceBundle",
    "ComplianceConfig",
    "EngineConfig",
    "ExplainReport",
    "Profile",
    "StreamingCleanConfig",
    "StreamingCleaner",
    "StreamingState",
    "__version__",
    "clean",
    "clean_csv",
    "clean_domain_file",
    "compare_clean",
    "compare_plans",
    "explain_clean",
    "generate_compliance_report",
    "infer_roles",
    "parse_domain",
    "profile",
    "suggest_plan",
]

#: Names served lazily from :mod:`freshdata.enterprise` via PEP 562, so the optional
#: enterprise layer (and its optional deps) is only imported when actually used. These are
#: deliberately *not* in ``__all__`` to keep ``import freshdata`` and ``import *`` light.
_ENTERPRISE_EXPORTS = frozenset({
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
    # entity resolution
    "resolve_entities",
    "link_entities",
    "MatchPair",
    "EntityCluster",
    "EntityResolutionReport",
    "EntityResolutionConfig",
    "ComparisonLevel",
    "BlockingRule",
})


def __getattr__(name: str) -> object:
    """Lazily resolve the ``enterprise`` submodule and its key exports (PEP 562)."""
    if name == "enterprise":
        import importlib

        return importlib.import_module("freshdata.enterprise")
    if name in _ENTERPRISE_EXPORTS:
        import importlib

        return getattr(importlib.import_module("freshdata.enterprise"), name)
    raise AttributeError(f"module 'freshdata' has no attribute {name!r}")


def __dir__() -> list:
    return sorted([*__all__, "enterprise", *_ENTERPRISE_EXPORTS])
