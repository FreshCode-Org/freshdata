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

_MASK_STRATEGIES = ("hash", "redact", "partial", "regex_scrub", "drop")
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
        object.__setattr__(self, "masking", tuple(self.masking))
        object.__setattr__(self, "semantic", tuple(self.semantic))

    def with_overrides(self, **changes: object) -> EnterpriseConfig:
        """Return a copy with the given top-level fields replaced."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]
