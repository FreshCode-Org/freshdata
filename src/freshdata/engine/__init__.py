"""The rule-based decision engine behind ``strategy="auto"``.

The engine profiles every column (missing ratio, dtype, skewness,
cardinality, inferred role, whether missingness looks informative) and the
dataset (size, duplicate ratio), then chooses cleaning actions from explicit
threshold rules. Every decision — including the decision to leave a column
untouched — is logged with a rationale, a risk level, and a confidence score.
"""

from .context import ColumnContext, build_contexts, infer_role
from .missing import auto_missing
from .model_select import ModelChoice, rank_missing_models, select_outlier_action
from .outliers import auto_outliers

__all__ = [
    "ColumnContext",
    "ModelChoice",
    "auto_missing",
    "auto_outliers",
    "build_contexts",
    "infer_role",
    "rank_missing_models",
    "select_outlier_action",
]
