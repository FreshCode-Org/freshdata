"""freshdata enterprise layer — clustering, trust scoring, lineage, and PII masking.

The headline entry point is :func:`clean_enterprise`, which runs core cleaning, value
clustering, semantic validation, and PII masking in one call and returns an
:class:`EnterpriseResult` with a Data Trust Score, a quality report, and OpenLineage JSON.

>>> import freshdata as fd
>>> from freshdata.enterprise import clean_enterprise, EnterpriseConfig, MaskingRule
>>> result = clean_enterprise(df, enterprise=EnterpriseConfig(
...     masking=(MaskingRule(name="pii", columns=("email",), strategy="hash"),)))
>>> print(result.summary())

Optional dependencies are imported lazily, so ``import freshdata`` stays cheap and
pandas-only installs keep working; the Polars-native fast paths activate automatically when
polars is installed.
"""

from .cleaner import (
    PII_PATTERNS,
    CallableValidator,
    Cluster,
    ClusterResult,
    ColumnValidation,
    MaskReport,
    ReferenceSetValidator,
    RegexValidator,
    SemanticValidator,
    ValidationReport,
    build_validator,
    cluster_column,
    detect_label_issues,
    detect_outliers,
    mask_dataframe,
    merge_clusters,
    run_semantic_validation,
    validate_columns,
)
from .config import (
    BUILTIN_SCRUB_PATTERNS,
    AnonymizationConfig,
    BlockingRule,
    ClusterConfig,
    ComparisonLevel,
    DriftConfig,
    EnterpriseConfig,
    EntityResolutionConfig,
    KAnonymityConfig,
    LineageConfig,
    MaskingRule,
    PIIDetectionConfig,
    SemanticValidatorConfig,
    TrustScoreWeights,
)
from .contracts import (
    ColumnBaseline,
    ColumnContract,
    DataContract,
    DatasetBaseline,
    DriftFinding,
    DriftReport,
    build_baseline,
    compare_to_baseline,
    load_baseline,
    monitor_contract,
    save_baseline,
)
from .entity_resolution import (
    EntityCluster,
    EntityResolutionReport,
    MatchPair,
    link_entities,
    resolve_entities,
)
from .interface import EnterpriseResult, clean_enterprise
from .lineage import LineageEvent, LineageTracker, schema_of
from .metrics import (
    ColumnTrust,
    QualityReport,
    TrustScore,
    build_quality_report,
    compute_trust_score,
)
from .privacy import (
    InMemoryTokenVault,
    JsonTokenVault,
    KAnonymityReport,
    MaskingEvent,
    PIIEntity,
    PIIScanReport,
    PrivacyReport,
    TokenVault,
    anonymize,
    check_k_anonymity,
    detect_pii,
    detokenize_value,
    tokenize_value,
)

__all__ = [
    # interface
    "clean_enterprise",
    "EnterpriseResult",
    # config
    "EnterpriseConfig",
    "MaskingRule",
    "ClusterConfig",
    "TrustScoreWeights",
    "LineageConfig",
    "SemanticValidatorConfig",
    "BUILTIN_SCRUB_PATTERNS",
    "DriftConfig",
    "PIIDetectionConfig",
    "AnonymizationConfig",
    "KAnonymityConfig",
    "ComparisonLevel",
    "BlockingRule",
    "EntityResolutionConfig",
    # contracts / drift
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
    "TokenVault",
    "InMemoryTokenVault",
    "JsonTokenVault",
    # entity resolution
    "resolve_entities",
    "link_entities",
    "MatchPair",
    "EntityCluster",
    "EntityResolutionReport",
    # metrics
    "TrustScore",
    "ColumnTrust",
    "QualityReport",
    "compute_trust_score",
    "build_quality_report",
    # lineage
    "LineageTracker",
    "LineageEvent",
    "schema_of",
    # clustering
    "merge_clusters",
    "cluster_column",
    "Cluster",
    "ClusterResult",
    # masking
    "mask_dataframe",
    "MaskReport",
    "PII_PATTERNS",
    # semantic validation
    "SemanticValidator",
    "ReferenceSetValidator",
    "RegexValidator",
    "CallableValidator",
    "build_validator",
    "run_semantic_validation",
    "validate_columns",
    "ValidationReport",
    "ColumnValidation",
    # cleanlab
    "detect_label_issues",
    "detect_outliers",
]
