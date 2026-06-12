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

__version__ = "0.3.0"

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
