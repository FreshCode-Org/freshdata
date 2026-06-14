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

from .api import clean, infer_roles, profile, suggest_plan
from .cleaner import Cleaner
from .config import CleanConfig
from .explain import ExplainReport, explain_clean
from .plan import CleanPlan, ColumnPlan, compare_clean, compare_plans
from .profile import ColumnProfile, Profile
from .report import Action, CleanReport

__version__ = "1.0.0"

__all__ = [
    "Action",
    "CleanConfig",
    "CleanPlan",
    "CleanReport",
    "Cleaner",
    "ColumnPlan",
    "ExplainReport",
    "ColumnProfile",
    "Profile",
    "__version__",
    "clean",
    "compare_clean",
    "compare_plans",
    "explain_clean",
    "infer_roles",
    "profile",
    "suggest_plan",
]

#: Names served lazily from :mod:`freshdata.enterprise` via PEP 562, so the optional
#: enterprise layer (and its optional deps) is only imported when actually used. These are
#: deliberately *not* in ``__all__`` to keep ``import freshdata`` and ``import *`` light.
_ENTERPRISE_EXPORTS = frozenset({
    "clean_enterprise",
    "FreshDataEnterprise",
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
