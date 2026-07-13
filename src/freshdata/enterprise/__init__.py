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
    ContractViolation,
    DataContract,
    DatasetBaseline,
    DriftFinding,
    DriftReport,
    build_baseline,
    compare_to_baseline,
    diff_schema,
    enforce_contract,
    load_baseline,
    monitor_contract,
    save_baseline,
)
from .entity_resolution import (
    EntityCluster,
    EntityResolutionReport,
    FieldExplanation,
    GoldenRecordPolicy,
    MatchPair,
    ReviewDecision,
    ReviewItem,
    ReviewQueueConfig,
    ReviewQueueReport,
    apply_review_decisions,
    build_review_queue,
    export_review_queue,
    link,
    link_entities,
    load_review_decisions,
    merge_entities,
    recalibrate_weights,
    redaction_columns,
    resolve_entities,
)
from .entity_resolution_templates import (
    DomainTemplate,
    education_template,
    get_template,
    healthcare_template,
    media_template,
    retail_template,
)
from .interface import EnterpriseResult, clean_enterprise
from .join_assist import (
    JoinCandidate,
    JoinKeyReport,
    suggest_join_keys,
)
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
    SqliteTokenVault,
    TokenVault,
    anonymize,
    check_k_anonymity,
    detect_pii,
    detokenize_value,
    make_vault,
    tokenize_value,
    vault_metadata,
)
from .privacy_policy import (
    Action,
    CompliancePack,
    Jurisdiction,
    PrivacyPolicy,
    PrivacyRule,
    apply_privacy_policy,
    available_packs,
    classify_columns,
    detokenize_series,
    load_compliance_pack,
    load_privacy_policy,
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
    "ContractViolation",
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
    "enforce_contract",
    # dirty-join assistant
    "suggest_join_keys",
    "JoinKeyReport",
    "JoinCandidate",
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
    "link",
    "link_entities",
    "MatchPair",
    "EntityCluster",
    "EntityResolutionReport",
    "FieldExplanation",
    "redaction_columns",
    # entity resolution — review queue & feedback
    "ReviewQueueConfig",
    "ReviewItem",
    "ReviewDecision",
    "ReviewQueueReport",
    "build_review_queue",
    "export_review_queue",
    "load_review_decisions",
    "apply_review_decisions",
    "recalibrate_weights",
    # entity resolution — golden records & templates
    "GoldenRecordPolicy",
    "merge_entities",
    "DomainTemplate",
    "get_template",
    "education_template",
    "healthcare_template",
    "retail_template",
    "media_template",
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
