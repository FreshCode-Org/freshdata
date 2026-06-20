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
    ClusterConfig,
    EnterpriseConfig,
    LineageConfig,
    MaskingRule,
    SemanticValidatorConfig,
    TrustScoreWeights,
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
