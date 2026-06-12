"""freshdata — fast, safe, automatic data cleaning for real-world tabular data.

>>> import freshdata as fd
>>> cleaned = fd.clean(df)
>>> cleaned, report = fd.clean(df, return_report=True)
>>> print(fd.profile(df))

Design principles
-----------------
- **Real cleaning, real rules.** ``strategy="auto"`` (default) runs a
  decision engine: every column is profiled (missing ratio, skewness,
  cardinality, inferred role) and threshold rules decide whether to impute,
  drop, cap, flag, or deliberately preserve. NaNs, duplicates, and outliers
  are never silently ignored — and never silently mangled either: targets are
  untouched, IDs are never imputed, free text is never force-filled.
- **Everything is reported.** Each decision is recorded with the column, the
  affected count, a rationale, a risk level, and a confidence score; the
  report also carries warnings and manual-review recommendations.
- **Never mutates input** (unless ``preserve_original=False``). ``clean``
  returns a new frame; profiling is read-only.
- **Fast by construction.** Vectorized pandas operations only, with
  sample-based pre-screening so type inference stays cheap on large frames.
"""

from .api import clean, profile
from .cleaner import Cleaner
from .config import CleanConfig
from .profile import ColumnProfile, Profile
from .report import Action, CleanReport

__version__ = "0.2.0"

__all__ = [
    "Action",
    "CleanConfig",
    "CleanReport",
    "Cleaner",
    "ColumnProfile",
    "Profile",
    "__version__",
    "clean",
    "profile",
]
