"""Configuration for the cleaning pipeline.

:class:`CleanConfig` is the single source of truth for every option accepted
by :func:`freshdata.clean`, :func:`freshdata.profile`, and
:class:`freshdata.Cleaner`. It is frozen (hashable, safely shareable) and
validates itself on construction so that bad options fail loudly and early.
"""

from __future__ import annotations

import dataclasses
import difflib
import warnings
from dataclasses import dataclass

_STRATEGY_CHOICES = ("conservative", "balanced", "aggressive", "auto")
_AUTO_DEPRECATION_WARNED = False
_IMPUTE_CHOICES = (None, "auto", "mean", "median", "mode")
_OUTLIER_CHOICES = (None, "clip", "flag")
_OUTLIER_METHODS = ("iqr", "zscore", "auto", "isolation_forest")
_OUTLIER_ACTIONS = (None, "cap", "remove", "flag")
_DUPLICATE_KEEP_CHOICES = ("first", "last", "drop", "aggregate")
_TRISTATE_CHOICES = (True, False, "auto")

#: Default outlier factor per method: 1.5×IQR (Tukey) or 3.0 standard deviations.
#: "auto" and "isolation_forest" resolve to a concrete method per column, so
#: their entry here is only the fallback used by representation-level previews.
_DEFAULT_FACTOR = {"iqr": 1.5, "zscore": 3.0, "auto": 1.5, "isolation_forest": 1.5}


@dataclass(frozen=True)
class CleanConfig:
    """Options controlling what :func:`freshdata.clean` does.

    Two layers of cleaning are controlled here:

    - **Representation repair** (whitespace, sentinel strings, wrong dtypes,
      exact duplicate rows, structurally empty rows/columns) — always safe,
      on by default.
    - **The decision engine** (``strategy="balanced"``, the default) — profiles
      every column and applies accuracy-first rules for missing values and
      outliers. Use ``strategy="aggressive"`` for zero-NaN scrubbing (KNN,
      column drops, capping). Set ``strategy="conservative"`` to disable the
      engine and only repair representation; statistical changes are then
      opt-in via ``impute`` / ``outliers``.
    """

    #: Normalize column names to snake_case and deduplicate collisions.
    column_names: bool = True
    #: Drop rows where every cell is missing.
    drop_empty_rows: bool = True
    #: Drop columns where every cell is missing.
    drop_empty_columns: bool = True
    #: Drop columns with a single distinct value (off by default).
    drop_constant_columns: bool = False
    #: Trim leading/trailing whitespace in text cells.
    strip_whitespace: bool = True
    #: Replace sentinel strings ("N/A", "null", "-", …) with missing values.
    normalize_sentinels: bool = True
    #: Additional sentinel strings to treat as missing (case-insensitive).
    extra_sentinels: tuple[str, ...] = ()
    #: Infer better dtypes for text columns (numeric, datetime, boolean).
    fix_dtypes: bool = True
    #: Fraction of non-missing values that must parse for a numeric conversion.
    numeric_threshold: float = 0.95
    #: Fraction of non-missing values that must parse for a datetime conversion.
    datetime_threshold: float = 0.95
    #: Drop exact duplicate rows (keeps the first occurrence).
    drop_duplicates: bool = True
    #: Restrict duplicate detection to these columns (post-rename names).
    duplicate_subset: tuple[str, ...] | None = None
    #: Cleaning strategy: "balanced" (default) accuracy-first engine;
    #: "aggressive" KNN/drops/capping; "conservative" representation only.
    #: "auto" is deprecated (alias for "aggressive").
    strategy: str = "balanced"
    #: Missing ratio at or below which a column is "low missingness".
    missing_threshold_low: float = 0.05
    #: Missing ratio at or below which a column is "medium missingness".
    missing_threshold_medium: float = 0.30
    #: Missing ratio at or below which a column is "high missingness";
    #: above it the column is "extreme" and dropped unless protected.
    missing_threshold_high: float = 0.60
    #: Duplicate-row ratio above which a data-collection warning is raised.
    duplicate_threshold: float = 0.10
    #: Engine action for detected outliers: "cap" (winsorize, default),
    #: "remove" (drop rows), "flag" (boolean column), None (preserve).
    outlier_action: str | None = "cap"
    #: Copy the input (default). With False the input frame may be reused
    #: in place to save memory and is no longer guaranteed unchanged.
    preserve_original: bool = True
    #: Print a one-line cleaning summary (plus warnings) after each clean.
    verbose: bool = True
    #: Columns that must never be dropped by the engine (post-rename names).
    preserve_columns: tuple[str, ...] = ()
    #: The label/target column; never modified by the engine. Columns named
    #: "target", "label", "y", "outcome", or "class" are detected automatically.
    target_column: str | None = None
    #: Columns to treat as identifiers (never imputed; outliers ignored).
    #: ID-like names ("*_id", "uuid", …) and all-unique keys are auto-detected.
    id_columns: tuple[str, ...] = ()
    #: How to resolve duplicates: keep "first"/"last", "drop" every member,
    #: or "aggregate" groups (numeric mean, first otherwise; needs a subset).
    duplicate_keep: str = "first"
    #: Allow duplicate removal on time-indexed frames (off: preserved + warned).
    allow_timeseries_duplicates: bool = False
    #: KNN imputation for medium-missingness numeric columns: "auto" uses it
    #: when scikit-learn is installed and correlated features exist.
    advanced_imputation: bool | str = "auto"
    #: Add ``<col>_was_missing`` indicator columns: "auto" adds them only when
    #: missingness looks informative (correlates with other features).
    missing_indicators: bool | str = "auto"
    #: Missing-value imputation override: None (engine decides under "auto"),
    #: "auto", "mean", "median", "mode" — forces simple per-column filling.
    impute: str | None = None
    #: Outlier handling override: None (engine decides under "auto"),
    #: "clip", "flag" — forces simple handling of every numeric column.
    outliers: str | None = None
    #: Outlier detection method: "iqr", "zscore", "auto" (per-column choice:
    #: z-score for ~normal, IQR for skewed), or "isolation_forest" (needs
    #: scikit-learn and >= 100 rows; falls back to IQR otherwise).
    outlier_method: str = "iqr"
    #: Detection factor; defaults to 1.5 for "iqr" and 3.0 for "zscore".
    outlier_factor: float | None = None
    #: Downcast numerics and convert low-cardinality text to category.
    optimize_memory: bool = False
    #: Max unique/total ratio for object→category conversion.
    category_threshold: float = 0.5
    #: Reset the index to 0..n-1 after cleaning (off: original labels kept).
    reset_index: bool = False
    #: Sample size used to cheaply pre-screen expensive type inference.
    sample_size: int = 10_000
    #: Seed for the (rare) sampling used during inference pre-screening.
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGY_CHOICES:
            raise ValueError(
                f"strategy must be one of {_STRATEGY_CHOICES}, got {self.strategy!r}"
            )
        if self.strategy == "auto":
            global _AUTO_DEPRECATION_WARNED  # noqa: PLW0603
            if not _AUTO_DEPRECATION_WARNED:
                warnings.warn(
                    'strategy="auto" is deprecated; use "aggressive" for full '
                    'engine scrubbing or "balanced" (default) for accuracy-first '
                    "cleaning",
                    DeprecationWarning,
                    stacklevel=3,
                )
                _AUTO_DEPRECATION_WARNED = True
        if self.impute not in _IMPUTE_CHOICES:
            raise ValueError(f"impute must be one of {_IMPUTE_CHOICES}, got {self.impute!r}")
        if self.outliers not in _OUTLIER_CHOICES:
            raise ValueError(f"outliers must be one of {_OUTLIER_CHOICES}, got {self.outliers!r}")
        if self.outlier_method not in _OUTLIER_METHODS:
            raise ValueError(
                f"outlier_method must be one of {_OUTLIER_METHODS}, got {self.outlier_method!r}"
            )
        if self.outlier_action not in _OUTLIER_ACTIONS:
            raise ValueError(
                f"outlier_action must be one of {_OUTLIER_ACTIONS}, got {self.outlier_action!r}"
            )
        if self.duplicate_keep not in _DUPLICATE_KEEP_CHOICES:
            raise ValueError(
                f"duplicate_keep must be one of {_DUPLICATE_KEEP_CHOICES}, "
                f"got {self.duplicate_keep!r}"
            )
        for name in ("advanced_imputation", "missing_indicators"):
            if getattr(self, name) not in _TRISTATE_CHOICES:
                raise ValueError(
                    f"{name} must be True, False, or 'auto', got {getattr(self, name)!r}"
                )
        for name in ("missing_threshold_low", "missing_threshold_medium",
                     "missing_threshold_high", "duplicate_threshold"):
            value = getattr(self, name)
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1), got {value!r}")
        if not (self.missing_threshold_low <= self.missing_threshold_medium
                <= self.missing_threshold_high):
            raise ValueError(
                "missing thresholds must be ordered: low <= medium <= high, got "
                f"{self.missing_threshold_low!r} / {self.missing_threshold_medium!r} / "
                f"{self.missing_threshold_high!r}"
            )
        for name in ("numeric_threshold", "datetime_threshold", "category_threshold"):
            value = getattr(self, name)
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value!r}")
        if self.outlier_factor is not None and not self.outlier_factor > 0:
            raise ValueError(f"outlier_factor must be > 0, got {self.outlier_factor!r}")
        if self.sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {self.sample_size!r}")
        if not all(isinstance(s, str) for s in self.extra_sentinels):
            raise TypeError("extra_sentinels must be strings")
        for name in ("preserve_columns", "id_columns"):
            if not all(isinstance(s, str) for s in getattr(self, name)):
                raise TypeError(f"{name} must be strings")
            object.__setattr__(self, name, tuple(getattr(self, name)))
        # Normalize user-facing conveniences onto the frozen instance.
        object.__setattr__(
            self, "extra_sentinels", tuple(s.casefold().strip() for s in self.extra_sentinels)
        )
        if self.duplicate_subset is not None:
            object.__setattr__(self, "duplicate_subset", tuple(self.duplicate_subset))

    @property
    def resolved_outlier_factor(self) -> float:
        """The outlier factor in effect, applying the per-method default."""
        if self.outlier_factor is not None:
            return self.outlier_factor
        return _DEFAULT_FACTOR[self.outlier_method]

    @property
    def engine_mode(self) -> str | None:
        """``"balanced"``, ``"aggressive"``, or ``None`` when engine is off."""
        if self.strategy == "conservative":
            return None
        if self.strategy in ("aggressive", "auto"):
            return "aggressive"
        return "balanced"


_FIELD_NAMES = frozenset(f.name for f in dataclasses.fields(CleanConfig))


def merge_options(base: CleanConfig | None, **options: object) -> CleanConfig:
    """Build a config from *base* (or defaults) plus keyword overrides.

    Unknown option names raise :class:`TypeError` with a "did you mean"
    suggestion, so typos never silently fall back to defaults.
    """
    unknown = sorted(set(options) - _FIELD_NAMES)
    if unknown:
        hints = []
        for name in unknown:
            match = difflib.get_close_matches(name, _FIELD_NAMES, n=1)
            hints.append(f"{name!r}" + (f" (did you mean {match[0]!r}?)" if match else ""))
        raise TypeError(
            f"unknown option(s): {', '.join(hints)}. "
            f"Valid options: {', '.join(sorted(_FIELD_NAMES))}"
        )
    if base is None:
        return CleanConfig(**options)  # type: ignore[arg-type]
    if not isinstance(base, CleanConfig):
        raise TypeError(f"config must be a CleanConfig, got {type(base).__name__}")
    return dataclasses.replace(base, **options)  # type: ignore[arg-type]
