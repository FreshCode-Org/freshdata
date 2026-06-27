"""Configuration for freshdata's enterprise layer.

These frozen dataclasses mirror the design of :class:`freshdata.CleanConfig`:
hashable, safely shareable, and self-validating on construction so bad options
fail loudly and early. They configure the *optional* enterprise capabilities —
fuzzy clustering, PII masking, semantic validation, trust scoring, and
OpenLineage emission — without changing the always-on core cleaning surface.

Nothing here imports a heavy dependency; the modules that actually need polars,
requests, or cleanlab import them lazily so ``import freshdata`` stays cheap.
"""

from __future__ import annotations

import dataclasses
import secrets
from dataclasses import dataclass, field
from typing import Any, Literal

_MASK_STRATEGIES = (
    "hash",
    "redact",
    "partial",
    "regex_scrub",
    "drop",
    # New privacy strategies (see freshdata.enterprise.privacy):
    "tokenize",
    "fpe",
    "surrogate",
)
_CLUSTER_METHODS = ("fingerprint", "ngram", "fingerprint_ngram")
_CANONICAL_CHOICES = ("most_frequent", "longest", "shortest", "first")
_SEMANTIC_KINDS = ("reference", "regex")

#: Named PII patterns recognised by the ``regex_scrub`` masking strategy.
#: The concrete regular expressions live in :mod:`freshdata.enterprise.cleaner`.
BUILTIN_SCRUB_PATTERNS = ("email", "phone", "ssn", "credit_card", "ip", "iban")


@dataclass(frozen=True)
class MaskingRule:
    """One PII masking rule applied to a set of columns.

    Columns are selected by exact ``columns`` names (post-clean snake_case) and
    by ``pattern`` (a regex matched against column *names*). At least one
    selector must be given.

    Strategies
    ----------
    ``hash``
        HMAC-SHA256 keyed by ``salt``, hex-truncated to ``hash_length``. Equal
        inputs map to equal tokens *for a given salt*. If ``salt`` is left
        empty a cryptographically random one is generated per rule, so the
        default is non-reversible (an empty salt would otherwise let an
        attacker rainbow-table low-entropy PII like emails or SSNs). Set an
        explicit ``salt`` when you need stable tokens across runs (e.g. joins).
    ``redact``
        Replace every non-null value with ``placeholder``.
    ``partial``
        Keep the last ``visible`` characters, prefix the rest with
        ``placeholder`` (e.g. ``"***6789"`` for a card number).
    ``regex_scrub``
        Replace PII *substrings* inside free text using the named
        ``scrub_patterns`` plus any custom ``regexes``.
    ``drop``
        Remove the column entirely.
    """

    name: str
    columns: tuple[str, ...] = ()
    pattern: str | None = None
    strategy: str = "hash"
    salt: str = ""
    hash_length: int = 16
    visible: int = 4
    placeholder: str = "***"
    scrub_patterns: tuple[str, ...] = ("email", "phone", "ssn", "credit_card")
    regexes: tuple[str, ...] = ()
    # --- privacy extensions (all optional; old rules keep working unchanged) ---
    #: Entity types (see :mod:`freshdata.enterprise.privacy`) this rule targets
    #: when driven by PII detection rather than explicit ``columns``.
    entity_types: tuple[str, ...] = ()
    #: Let nearby context keywords raise detector confidence for this rule.
    use_context: bool = True
    #: Opt-in reversibility (only meaningful for ``tokenize``/``fpe``).
    reversible: bool = False
    #: Secret key material for ``tokenize``/``fpe``. Prefer ``key_env`` so the
    #: literal never lives in source; raw keys are never written to reports.
    key: str | None = None
    #: Name of an environment variable holding the key (takes precedence).
    key_env: str | None = None
    #: Optional on-disk JSON token vault for reversible ``tokenize``.
    token_vault_path: str | None = None
    #: Keep the shape (digit count / separators / domain) of the original value.
    preserve_format: bool = False
    #: Extra HIPAA identifier tags to attach to masking events.
    hipaa_tags: tuple[str, ...] = ()
    #: Extra GDPR personal-data category tags to attach to masking events.
    gdpr_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in _MASK_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_MASK_STRATEGIES}, got {self.strategy!r}"
            )
        if not self.columns and not self.pattern:
            raise ValueError(
                f"masking rule {self.name!r} selects nothing: set columns= or pattern="
            )
        if self.visible < 0:
            raise ValueError(f"visible must be >= 0, got {self.visible!r}")
        if not 4 <= self.hash_length <= 64:
            raise ValueError(f"hash_length must be in [4, 64], got {self.hash_length!r}")
        unknown = sorted(set(self.scrub_patterns) - set(BUILTIN_SCRUB_PATTERNS))
        if unknown:
            raise ValueError(
                f"unknown scrub_patterns {unknown}; known: {list(BUILTIN_SCRUB_PATTERNS)}"
            )
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "scrub_patterns", tuple(self.scrub_patterns))
        object.__setattr__(self, "regexes", tuple(self.regexes))
        object.__setattr__(self, "entity_types", tuple(self.entity_types))
        object.__setattr__(self, "hipaa_tags", tuple(self.hipaa_tags))
        object.__setattr__(self, "gdpr_tags", tuple(self.gdpr_tags))
        # Secure default: an empty salt on a hash rule would make low-entropy
        # PII trivially reversible, so generate a random per-rule salt instead.
        if self.strategy == "hash" and not self.salt:
            object.__setattr__(self, "salt", secrets.token_hex(16))


@dataclass(frozen=True)
class ClusterConfig:
    """Settings for heuristic value clustering (typo / variant merging).

    ``fingerprint`` is the fully Polars-native token key-collision algorithm
    (case, punctuation, whitespace, and word-order insensitive). ``ngram`` adds
    character n-gram keys to catch single-character typos; ``fingerprint_ngram``
    runs both passes.
    """

    columns: tuple[str, ...] = ()
    method: str = "fingerprint"
    ngram_size: int = 2
    min_cluster_size: int = 2
    canonical: str = "most_frequent"

    def __post_init__(self) -> None:
        if self.method not in _CLUSTER_METHODS:
            raise ValueError(f"method must be one of {_CLUSTER_METHODS}, got {self.method!r}")
        if self.canonical not in _CANONICAL_CHOICES:
            raise ValueError(
                f"canonical must be one of {_CANONICAL_CHOICES}, got {self.canonical!r}"
            )
        if self.ngram_size < 1:
            raise ValueError(f"ngram_size must be >= 1, got {self.ngram_size!r}")
        if self.min_cluster_size < 2:
            raise ValueError(f"min_cluster_size must be >= 2, got {self.min_cluster_size!r}")
        object.__setattr__(self, "columns", tuple(self.columns))


@dataclass(frozen=True)
class TrustScoreWeights:
    """Relative weights blending the four trust dimensions into one score.

    Weights need not sum to 1 — :meth:`normalized` rescales them. Each must be
    non-negative and at least one must be positive.
    """

    completeness: float = 0.30
    validity: float = 0.30
    uniqueness: float = 0.20
    consistency: float = 0.20

    def __post_init__(self) -> None:
        for name in ("completeness", "validity", "uniqueness", "consistency"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} weight must be >= 0, got {value!r}")
        if self.completeness + self.validity + self.uniqueness + self.consistency <= 0:
            raise ValueError("at least one trust weight must be positive")

    def normalized(self) -> dict[str, float]:
        """Weights rescaled to sum to 1.0, keyed by dimension name."""
        total = self.completeness + self.validity + self.uniqueness + self.consistency
        return {
            "completeness": self.completeness / total,
            "validity": self.validity / total,
            "uniqueness": self.uniqueness / total,
            "consistency": self.consistency / total,
        }


@dataclass(frozen=True)
class LineageConfig:
    """Identity and addressing for OpenLineage events.

    ``actor`` records *who* ran the pipeline; when ``None`` the OS login name is
    used at run time. ``namespace``/``job_name`` address the job in a catalog;
    ``dataset_namespace`` addresses the input/output datasets.
    """

    namespace: str = "freshdata"
    job_name: str = "freshdata.clean"
    producer: str = "https://github.com/FreshCode-Org/freshdata"
    dataset_namespace: str = "freshdata"
    actor: str | None = None
    emit: bool = True


@dataclass(frozen=True)
class SemanticValidatorConfig:
    """Declarative spec for an external/reference semantic validator.

    The concrete validator object is built by
    :func:`freshdata.enterprise.cleaner.build_validator`. ``kind`` selects the
    backend; ``reference``/``regex`` carry the backend-specific setup.
    ``columns`` lists the columns this validator checks.
    """

    name: str
    kind: str = "reference"
    columns: tuple[str, ...] = ()
    reference: tuple[str, ...] = ()
    regex: str | None = None
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _SEMANTIC_KINDS:
            raise ValueError(f"kind must be one of {_SEMANTIC_KINDS}, got {self.kind!r}")
        if self.kind == "reference" and not self.reference:
            raise ValueError(f"validator {self.name!r}: kind='reference' needs reference=")
        if self.kind == "regex" and not self.regex:
            raise ValueError(f"validator {self.name!r}: kind='regex' needs regex=")
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "reference", tuple(self.reference))


@dataclass(frozen=True)
class DriftConfig:
    """Thresholds for schema-drift & data-contract monitoring.

    Consumed by :func:`freshdata.enterprise.contracts.compare_to_baseline`. All
    deltas are absolute unless named ``*_ratio``. ``*_warn`` thresholds raise a
    warning; ``*_fail`` thresholds raise an error. The ``ks``/``psi`` knobs gate
    distribution drift (Kolmogorov–Smirnov statistic and Population Stability
    Index); both are dependency-free.
    """

    enabled: bool = True
    fail_on_schema_drift: bool = True
    warn_on_distribution_drift: bool = True
    missing_ratio_warn_delta: float = 0.05
    missing_ratio_fail_delta: float = 0.20
    cardinality_warn_delta_ratio: float = 0.50
    numeric_ks_warn: float = 0.10
    numeric_ks_fail: float = 0.25
    psi_warn: float = 0.10
    psi_fail: float = 0.25
    min_samples_for_distribution: int = 30
    max_categories_for_categorical_drift: int = 100
    #: Optional trust-score gate (0-100). If set and the current frame scores
    #: below it, the monitor fails even when only warnings were raised.
    trust_score_min: float | None = None

    def __post_init__(self) -> None:
        if self.trust_score_min is not None and not 0.0 <= self.trust_score_min <= 100.0:
            raise ValueError(
                f"trust_score_min must be in [0, 100], got {self.trust_score_min!r}"
            )


@dataclass(frozen=True)
class PIIDetectionConfig:
    """How to scan free-text / mixed columns for PII.

    The fallback detector (regex + context keywords) needs no extra
    dependencies. When ``use_ner`` is set and the optional
    ``freshdata-cleaner[privacy]`` extra (Presidio) is installed, an NER pass is
    layered on top; otherwise it is skipped silently.
    """

    enabled: bool = True
    use_regex: bool = True
    use_context: bool = True
    use_ner: bool = False
    language: str = "en"
    min_score: float = 0.50
    entities: tuple[str, ...] = (
        "EMAIL",
        "PHONE",
        "SSN",
        "CREDIT_CARD",
        "IP_ADDRESS",
        "IBAN",
        "DATE_OF_BIRTH",
        "ZIP_CODE",
        "MRN",
        "PATIENT_ID",
        "ICD_CODE",
        "INSURANCE_ID",
        "PASSPORT",
        "DRIVER_LICENSE",
        "PERSON",
    )
    context_window: int = 40
    custom_patterns: tuple[dict[str, Any], ...] = ()
    #: Redact raw matched substrings in detection output (safe default).
    redact_samples: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {self.min_score!r}")
        if self.context_window < 0:
            raise ValueError(f"context_window must be >= 0, got {self.context_window!r}")
        object.__setattr__(self, "entities", tuple(self.entities))
        object.__setattr__(self, "custom_patterns", tuple(self.custom_patterns))


@dataclass(frozen=True)
class AnonymizationConfig:
    """A standalone anonymization spec (column-targeted or detection-driven).

    Mirrors the masking knobs on :class:`MaskingRule` for callers that want to
    drive :func:`freshdata.enterprise.privacy.anonymize` without the legacy
    rule object. ``preserve_format`` selects surrogate/FPE shape preservation.
    """

    strategy: Literal[
        "hash", "redact", "partial", "regex_scrub", "drop", "tokenize", "fpe", "surrogate"
    ] = "redact"
    reversible: bool = False
    key: str | None = None
    key_env: str | None = None
    token_prefix: str = "tok"
    preserve_format: bool = False
    placeholder: str = "***"
    visible: int = 4
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.strategy not in _MASK_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_MASK_STRATEGIES}, got {self.strategy!r}"
            )
        if self.visible < 0:
            raise ValueError(f"visible must be >= 0, got {self.visible!r}")
        object.__setattr__(self, "tags", tuple(self.tags))


@dataclass(frozen=True)
class KAnonymityConfig:
    """Settings for a k-anonymity check over quasi-identifier columns."""

    enabled: bool = False
    quasi_identifiers: tuple[str, ...] = ()
    k: int = 5
    fail_under_k: bool = True
    max_report_groups: int = 20

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k!r}")
        if self.max_report_groups < 0:
            raise ValueError(
                f"max_report_groups must be >= 0, got {self.max_report_groups!r}"
            )
        object.__setattr__(self, "quasi_identifiers", tuple(self.quasi_identifiers))


_COMPARISON_KINDS = (
    "exact",
    "jaro_winkler",
    "levenshtein",
    "numeric_distance",
    "date_distance",
    "phonetic",
    "custom_sql",
)


@dataclass(frozen=True)
class ComparisonLevel:
    """One field comparison contributing weighted evidence to a match score.

    ``threshold`` is the agreement cut-off (e.g. a Jaro-Winkler similarity, or
    an absolute numeric/date distance below which the field is considered to
    agree). ``weight`` scales the field's evidence. ``custom_sql`` levels carry
    a boolean SQL expression in ``sql`` (DuckDB backend only).
    """

    column: str
    kind: Literal[
        "exact",
        "jaro_winkler",
        "levenshtein",
        "numeric_distance",
        "date_distance",
        "phonetic",
        "custom_sql",
    ] = "exact"
    threshold: float = 0.0
    weight: float = 1.0
    sql: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _COMPARISON_KINDS:
            raise ValueError(f"kind must be one of {_COMPARISON_KINDS}, got {self.kind!r}")
        if self.weight < 0:
            raise ValueError(f"weight must be >= 0, got {self.weight!r}")
        if self.kind == "custom_sql" and not self.sql:
            raise ValueError("custom_sql comparison requires sql=")


@dataclass(frozen=True)
class BlockingRule:
    """A candidate-pair generation rule (a SQL equi-join predicate).

    ``sql`` is a boolean expression over the ``l``/``r`` aliases, e.g.
    ``"lower(l.email) = lower(r.email)"``. The DuckDB backend uses it verbatim;
    the pandas fallback understands a conjunction of equality predicates.
    """

    sql: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.sql or not self.sql.strip():
            raise ValueError("BlockingRule.sql must be a non-empty SQL predicate")


@dataclass(frozen=True)
class EntityResolutionConfig:
    """Probabilistic entity-resolution settings (Splink-style, DuckDB-backed).

    Disabled by default. Blocking rules cap the candidate space; ``max_pairs``
    is a hard safety gate that aborts before a cartesian explosion. Scoring is
    rule-weighted probabilistic linkage (not full EM-trained Splink parity).
    """

    enabled: bool = False
    backend: Literal["duckdb", "pandas"] = "duckdb"
    unique_id_column: str = "id"
    blocking_rules: tuple[BlockingRule, ...] = ()
    comparisons: tuple[ComparisonLevel, ...] = ()
    match_threshold: float = 0.85
    clerical_review_threshold: float = 0.65
    max_pairs: int | None = 5_000_000
    output_clusters: bool = True
    link_type: Literal["dedupe_only", "link_only", "link_and_dedupe"] = "dedupe_only"
    left_prefix: str = "l"
    right_prefix: str = "r"
    duckdb_path: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in ("duckdb", "pandas"):
            raise ValueError(f"backend must be 'duckdb' or 'pandas', got {self.backend!r}")
        if not 0.0 <= self.clerical_review_threshold <= self.match_threshold <= 1.0:
            raise ValueError(
                "require 0 <= clerical_review_threshold <= match_threshold <= 1, got "
                f"{self.clerical_review_threshold!r} / {self.match_threshold!r}"
            )
        if self.max_pairs is not None and self.max_pairs < 1:
            raise ValueError(f"max_pairs must be >= 1 or None, got {self.max_pairs!r}")
        if not all(isinstance(b, BlockingRule) for b in self.blocking_rules):
            raise TypeError("blocking_rules must be a sequence of BlockingRule")
        if not all(isinstance(c, ComparisonLevel) for c in self.comparisons):
            raise TypeError("comparisons must be a sequence of ComparisonLevel")
        object.__setattr__(self, "blocking_rules", tuple(self.blocking_rules))
        object.__setattr__(self, "comparisons", tuple(self.comparisons))


@dataclass(frozen=True)
class EnterpriseConfig:
    """Top-level switchboard for the enterprise pipeline.

    Bundles the feature toggles and sub-configs consumed by
    :func:`freshdata.enterprise.clean_enterprise`. Frozen and hashable, so a
    single instance can be shared across threads or reused for many frames.
    """

    actor: str | None = None
    enable_masking: bool = True
    enable_clustering: bool = False
    enable_validation: bool = True
    enable_lineage: bool = True
    masking: tuple[MaskingRule, ...] = ()
    clustering: ClusterConfig | None = None
    semantic: tuple[SemanticValidatorConfig, ...] = ()
    trust_weights: TrustScoreWeights = field(default_factory=TrustScoreWeights)
    lineage: LineageConfig = field(default_factory=LineageConfig)
    #: Optional quality gate: fail the run if the post-clean trust score
    #: (0-100) falls below this threshold. ``None`` disables the gate.
    fail_under_trust: float | None = None
    # --- new enterprise capabilities (all opt-in, backward compatible) ---
    drift: DriftConfig | None = None
    privacy: PIIDetectionConfig | None = None
    anonymization: tuple[AnonymizationConfig, ...] = ()
    k_anonymity: KAnonymityConfig | None = None
    entity_resolution: EntityResolutionConfig | None = None
    enable_contracts: bool = False
    enable_privacy_detection: bool = False
    enable_entity_resolution: bool = False

    def __post_init__(self) -> None:
        if not all(isinstance(r, MaskingRule) for r in self.masking):
            raise TypeError("masking must be a sequence of MaskingRule")
        if not all(isinstance(s, SemanticValidatorConfig) for s in self.semantic):
            raise TypeError("semantic must be a sequence of SemanticValidatorConfig")
        if self.clustering is not None and not isinstance(self.clustering, ClusterConfig):
            raise TypeError("clustering must be a ClusterConfig or None")
        if self.fail_under_trust is not None and not 0.0 <= self.fail_under_trust <= 100.0:
            raise ValueError(
                f"fail_under_trust must be in [0, 100], got {self.fail_under_trust!r}"
            )
        if not all(isinstance(a, AnonymizationConfig) for a in self.anonymization):
            raise TypeError("anonymization must be a sequence of AnonymizationConfig")
        if self.drift is not None and not isinstance(self.drift, DriftConfig):
            raise TypeError("drift must be a DriftConfig or None")
        if self.privacy is not None and not isinstance(self.privacy, PIIDetectionConfig):
            raise TypeError("privacy must be a PIIDetectionConfig or None")
        if self.k_anonymity is not None and not isinstance(self.k_anonymity, KAnonymityConfig):
            raise TypeError("k_anonymity must be a KAnonymityConfig or None")
        if self.entity_resolution is not None and not isinstance(
            self.entity_resolution, EntityResolutionConfig
        ):
            raise TypeError("entity_resolution must be an EntityResolutionConfig or None")
        object.__setattr__(self, "masking", tuple(self.masking))
        object.__setattr__(self, "semantic", tuple(self.semantic))
        object.__setattr__(self, "anonymization", tuple(self.anonymization))

    def with_overrides(self, **changes: object) -> EnterpriseConfig:
        """Return a copy with the given top-level fields replaced."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]
