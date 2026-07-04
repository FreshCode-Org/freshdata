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
_OUTLIER_ACTIONS = (None, "auto", "cap", "remove", "flag")
_DUPLICATE_KEEP_CHOICES = ("first", "last", "drop", "aggregate")
_TRISTATE_CHOICES = (True, False, "auto")
_STRING_CASE_CHOICES = (None, "lower", "upper")
_SEMANTIC_MODE_CHOICES = (None, "off", "assist", "review", "auto")
_SEMANTIC_PRIVACY_CHOICES = (
    "local_only",
    "redact_external",
    "allow_private_endpoint",
    "disabled",
)
#: Semantic modes that actually engage the layer (``None``/``"off"`` are inert).
_SEMANTIC_ACTIVE_MODES = frozenset({"assist", "review", "auto"})

#: Default outlier factor per method: 1.5×IQR (Tukey) or 3.0 standard deviations.
#: "auto" and "isolation_forest" resolve to a concrete method per column, so
#: their entry here is only the fallback used by representation-level previews.
_DEFAULT_FACTOR = {"iqr": 1.5, "zscore": 3.0, "auto": 1.5, "isolation_forest": 1.5}


def _coerce_str_tuple(value: tuple[str, ...] | str) -> tuple[str, ...]:
    """Accept a single column/sentinel name or a tuple of names."""
    if isinstance(value, str):
        return (value,)
    return tuple(value)


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
    #: Optional case normalization for text cells. ``None`` preserves case;
    #: ``"lower"`` and ``"upper"`` normalize string values.
    string_case: str | None = None
    #: Infer better dtypes for text columns (numeric, datetime, boolean).
    fix_dtypes: bool = True
    #: Fraction of non-missing values that must parse for a numeric conversion.
    numeric_threshold: float = 0.95
    #: Fraction of non-missing values that must parse for a datetime conversion.
    datetime_threshold: float = 0.95
    #: Keep zero-padded numeric strings (ZIP codes, zero-padded IDs, phone
    #: codes) as text instead of silently coercing "007" -> 7. Set False to
    #: convert them to numbers anyway.
    preserve_leading_zeros: bool = True
    #: Day/month order for ambiguous numeric dates. "auto" (default) infers the
    #: order from disambiguating values (a field > 12) and falls back to
    #: month-first when a column is genuinely ambiguous. True forces day-first
    #: (DD/MM/YYYY), False forces month-first (MM/DD/YYYY).
    dayfirst: bool | str = "auto"
    #: Decimal separator used when parsing numbers from text (e.g. "," for
    #: many European locales). Must be a single character and differ from
    #: ``thousands``.
    decimal: str = "."
    #: Thousands/grouping separator stripped when parsing numbers from text.
    thousands: str = ","
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
    #: Engine action for detected outliers. "auto" (default) is context-aware:
    #: it flags under strategy="balanced" and caps (winsorizes) under
    #: "aggressive". "cap"/"remove"/"flag" are explicit directives, always
    #: applied to eligible numeric columns — heavy-tailed columns (>15% outlying)
    #: are still acted on, but a warning is raised. None detects and preserves.
    outlier_action: str | None = "auto"
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

    # -- Semantic cleaning layer (off by default) --------------------------
    #: Semantic cleaning mode. ``None``/``"off"`` disable the layer entirely;
    #: ``"assist"`` only reports proposals (never mutates); ``"review"`` applies
    #: deterministic zero-risk repairs and suggests the rest; ``"auto"`` applies
    #: high-confidence, low-risk, policy-approved repairs.
    semantic_mode: str | None = None
    #: Minimum confidence to auto-apply a semantic proposal.
    semantic_auto_threshold: float = 0.95
    #: Minimum confidence below which a proposal is never applied (only skipped).
    semantic_review_threshold: float = 0.70
    #: Skip semantic profiling for columns with more distinct values than this.
    semantic_max_distinct_values: int = 500
    #: Cap on rows sampled when profiling distinct semantic values.
    semantic_sample_size: int = 10_000
    #: Candidate-generation backends, in trust order. ``"deterministic"``
    #: (Phase-2 experts, always safe to keep first), ``"memory"`` (replay of a
    #: supplied CleaningMemory), and ``"embedding"`` (optional local model via
    #: the ``[semantic]`` extra; self-disables with a report note when the
    #: dependency or model files are missing — never downloads anything).
    semantic_backends: tuple[str, ...] = ("deterministic",)
    #: Optional user-supplied semantic hints (e.g. per-column semantic_type,
    #: unit, allowed_values, mutable). See the semantic layer docs.
    semantic_context: object | None = None
    #: Privacy posture for any future external inference; the deterministic v1
    #: never leaves the process regardless of this value.
    semantic_privacy_policy: str = "local_only"
    #: Optional resource budget for model-assisted backends: recognized keys
    #: are ``max_columns``, ``max_distinct_values``, ``max_model_calls`` and
    #: ``max_seconds``. Deterministic and memory backends are never metered;
    #: an exhausted budget stops the embedding backend cleanly with a report
    #: fallback event.
    semantic_budget: dict[str, object] | None = None
    #: In-process LRU capacity (distinct texts) for embedding vectors; 0
    #: disables caching. Embeddings only — decisions and rows are never cached,
    #: and nothing is written to disk.
    semantic_embedding_cache_size: int = 65_536

    # -- Context policy compiler (deterministic, off by default) ------------
    #: Natural-language cleaning rules ("CustomerID is unique. Never modify
    #: revenue."), compiled deterministically into a ContextPolicy when the
    #: config meets a frame. ``None`` (default) changes nothing.
    context: str | None = None
    #: A pre-compiled :class:`freshdata.ContextPolicy`; mutually exclusive
    #: with ``context`` (supplying a policy skips parsing entirely).
    policy: object | None = None
    #: Strict context mode: unresolved column references, unparsed sentences,
    #: and protection conflicts raise :class:`freshdata.PolicyError` instead of
    #: being surfaced as report warnings.
    strict: bool = False

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
        if self.string_case not in _STRING_CASE_CHOICES:
            raise ValueError(
                f"string_case must be one of {_STRING_CASE_CHOICES}, got {self.string_case!r}"
            )
        if self.duplicate_keep not in _DUPLICATE_KEEP_CHOICES:
            raise ValueError(
                f"duplicate_keep must be one of {_DUPLICATE_KEEP_CHOICES}, "
                f"got {self.duplicate_keep!r}"
            )
        for name in ("advanced_imputation", "missing_indicators", "dayfirst"):
            if getattr(self, name) not in _TRISTATE_CHOICES:
                raise ValueError(
                    f"{name} must be True, False, or 'auto', got {getattr(self, name)!r}"
                )
        for name in ("decimal", "thousands"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 1:
                raise ValueError(f"{name} must be a single character, got {value!r}")
        if self.decimal == self.thousands:
            raise ValueError(
                f"decimal and thousands separators must differ, both are {self.decimal!r}"
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
        self._validate_semantic()
        self._validate_context()
        extra = _coerce_str_tuple(self.extra_sentinels)
        if not all(isinstance(s, str) for s in extra):
            raise TypeError("extra_sentinels must be strings")
        for name in ("preserve_columns", "id_columns"):
            raw = _coerce_str_tuple(getattr(self, name))
            if not all(isinstance(s, str) for s in raw):
                raise TypeError(f"{name} must be strings")
            object.__setattr__(self, name, raw)
        # Normalize user-facing conveniences onto the frozen instance.
        object.__setattr__(
            self, "extra_sentinels", tuple(s.casefold().strip() for s in extra)
        )
        if self.duplicate_subset is not None:
            object.__setattr__(
                self, "duplicate_subset", _coerce_str_tuple(self.duplicate_subset)
            )

    def _validate_semantic(self) -> None:
        if self.semantic_mode not in _SEMANTIC_MODE_CHOICES:
            raise ValueError(
                f"semantic_mode must be one of {_SEMANTIC_MODE_CHOICES}, "
                f"got {self.semantic_mode!r}"
            )
        for name in ("semantic_auto_threshold", "semantic_review_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.semantic_review_threshold > self.semantic_auto_threshold:
            raise ValueError(
                "semantic_review_threshold must be <= semantic_auto_threshold, got "
                f"{self.semantic_review_threshold!r} > {self.semantic_auto_threshold!r}"
            )
        if self.semantic_max_distinct_values < 1:
            raise ValueError(
                "semantic_max_distinct_values must be >= 1, got "
                f"{self.semantic_max_distinct_values!r}"
            )
        if self.semantic_sample_size < 1:
            raise ValueError(
                f"semantic_sample_size must be >= 1, got {self.semantic_sample_size!r}"
            )
        if self.semantic_privacy_policy not in _SEMANTIC_PRIVACY_CHOICES:
            raise ValueError(
                f"semantic_privacy_policy must be one of {_SEMANTIC_PRIVACY_CHOICES}, "
                f"got {self.semantic_privacy_policy!r}"
            )

    def _validate_context(self) -> None:
        if self.context is not None and not isinstance(self.context, str):
            raise TypeError(f"context must be a string, got {type(self.context).__name__}")
        if self.policy is not None:
            # Lazy import: the context package is only loaded when a policy is
            # actually supplied, keeping ``import freshdata`` light.
            from .context.types import ContextPolicy  # noqa: PLC0415

            if not isinstance(self.policy, ContextPolicy):
                raise TypeError(
                    "policy must be a freshdata.ContextPolicy "
                    f"(see fd.compile_context), got {type(self.policy).__name__}"
                )
        if self.context is not None and self.policy is not None:
            raise ValueError(
                "context= and policy= are mutually exclusive: pass the raw text "
                "to compile it here, or a pre-compiled ContextPolicy, not both"
            )
        if not isinstance(self.strict, bool):
            raise TypeError(f"strict must be a bool, got {self.strict!r}")

    @property
    def semantic_enabled(self) -> bool:
        """``True`` when ``semantic_mode`` actually engages the semantic layer."""
        return self.semantic_mode in _SEMANTIC_ACTIVE_MODES

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
