"""The enterprise cleaning engine: clustering, PII masking, semantic validation.

Everything here is Polars-native on the hot path *when Polars is installed*, with a
vectorized pandas fallback otherwise — and it accepts and returns whichever frame type it
was given (via :mod:`freshdata.adapters.polars`). Heavy/optional dependencies (polars,
requests, cleanlab) are imported lazily so ``import freshdata`` stays cheap.

Three capabilities:

- **Clustering** — OpenRefine-style key-collision merging of value variants (typos, case,
  punctuation, word-order). Fingerprint keys are built with pure Polars string expressions;
  an n-gram method catches single-character typos.
- **PII masking** — salted SHA-256 hashing, redaction, partial masking, regex scrubbing,
  and column dropping. Regexes are lookaround-free so the *same* pattern runs under Polars'
  Rust regex engine and Python's ``re``.
- **Semantic validation** — an extensible validator interface (reference sets, regex, user
  callables, external APIs) plus an optional :mod:`cleanlab` wrapper for label noise.
"""

from __future__ import annotations

import abc
import hashlib
import hmac
import re
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .._util import _is_stringlike_dtype
from ..adapters.polars import _polars_module, is_polars_frame, to_pandas
from .config import ClusterConfig, MaskingRule, SemanticValidatorConfig


def _all_columns(df: Any) -> list:
    # Both pandas and polars expose ``.columns``; pandas returns an Index, polars a list.
    return list(df.columns)


def _is_string_column(df: Any, column: Any) -> bool:
    if is_polars_frame(df):
        pl = _polars_module()
        dtype = df.schema.get(column)
        return dtype in (pl.Utf8, pl.String, pl.Categorical)
    if column not in df.columns:
        return False
    dtype = df[column].dtype
    return _is_stringlike_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype)


def _string_columns(df: Any) -> list:
    return [c for c in _all_columns(df) if _is_string_column(df, c)]


# =====================================================================
# Clustering (key collision)
# =====================================================================

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _join_sorted_tokens(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    return " ".join(sorted({tok for tok in text.split(" ") if tok}))


def _ngram_str(value: Any, n: int) -> str:
    """Sorted, de-duplicated character n-gram key (both backends)."""
    clean = _ALNUM_RE.sub("", str(value).lower())
    if len(clean) < n:
        return clean
    grams = sorted({clean[i : i + n] for i in range(len(clean) - n + 1)})
    return "".join(grams)


def _fingerprint_expr(pl: Any, column: Any) -> Any:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"[^\w\s]", "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.split(" ")
        .list.unique()
        .list.sort()
        .list.join(" ")
    )


def _ngram_expr(pl: Any, column: Any, n: int) -> Any:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .map_elements(
            lambda v: _ngram_str(v, n) if v is not None else None, return_dtype=pl.Utf8
        )
    )


def _fingerprint_series_pandas(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.strip().str.lower()
    text = text.str.replace(_PUNCT_RE, "", regex=True)
    text = text.str.replace(_WS_RE, " ", regex=True)
    text = text.str.strip()
    return text.map(_join_sorted_tokens)


def _keys_and_counts(df: Any, column: Any, method: str, ngram_size: int) -> list:
    """Return distinct ``(value, key, count)`` triples for one column.

    Polars frames use native string expressions + ``group_by``; pandas frames use
    vectorized string ops + ``groupby``. Only the *distinct* combinations cross into
    Python, so this stays cheap on tall frames.
    """
    if is_polars_frame(df):
        pl = _polars_module()
        key_expr = _fingerprint_expr(pl, column) if method == "fingerprint" else _ngram_expr(
            pl, column, ngram_size
        )
        grouped = (
            df.select(
                [pl.col(column).cast(pl.Utf8).alias("__value"), key_expr.alias("__key")]
            )
            .drop_nulls("__value")
            .group_by(["__key", "__value"])
            .len()
        )
        return [(value, key, int(count)) for key, value, count in grouped.iter_rows()]

    series = df[column]
    nonnull = series[series.notna()]
    if nonnull.empty:
        return []
    if method == "fingerprint":
        keys = _fingerprint_series_pandas(nonnull)
    else:
        keys = nonnull.map(lambda v: _ngram_str(v, ngram_size))
    tmp = pd.DataFrame(
        {"value": nonnull.astype(object).to_numpy(), "key": keys.astype(object).to_numpy()}
    )
    grouped = tmp.groupby(["key", "value"], dropna=True).size().reset_index(name="count")
    return list(
        zip(
            grouped["value"].tolist(),
            grouped["key"].tolist(),
            grouped["count"].astype(int).tolist(),
        )
    )


def _pick_canonical(members: list, policy: str) -> str:
    """Choose the surviving value for a cluster of ``(value, count)`` members."""
    if policy == "longest":
        return max(members, key=lambda vc: (len(vc[0]), vc[1]))[0]
    if policy == "shortest":
        return min(members, key=lambda vc: (len(vc[0]), -vc[1]))[0]
    if policy == "first":
        return min(members, key=lambda vc: vc[0])[0]
    return max(members, key=lambda vc: (vc[1], len(vc[0]), vc[0]))[0]  # most_frequent


@dataclass(frozen=True)
class Cluster:
    """One merged group: a canonical value and the variants that map to it."""

    key: str
    canonical: str
    variants: tuple[str, ...]
    size: int
    n_variants: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "canonical": self.canonical,
            "variants": list(self.variants),
            "size": self.size,
            "n_variants": self.n_variants,
        }


@dataclass(frozen=True)
class ClusterResult:
    """Outcome of clustering one column (pure — holds the mapping, not the data)."""

    column: str
    method: str
    n_clusters: int
    n_cells_merged: int
    mapping: dict[str, str]
    clusters: tuple[Cluster, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "method": self.method,
            "n_clusters": self.n_clusters,
            "n_cells_merged": self.n_cells_merged,
            "clusters": [c.to_dict() for c in self.clusters],
        }

    def __repr__(self) -> str:
        return (
            f"<ClusterResult {self.column!r} method={self.method} "
            f"clusters={self.n_clusters} merged={self.n_cells_merged}>"
        )


def _clusters_from_counts(rows: list, config: ClusterConfig) -> tuple[dict, list, int]:
    groups: dict[str, list] = defaultdict(list)
    for value, key, count in rows:
        if key is None:
            continue
        groups[key].append((str(value), int(count)))

    mapping: dict[str, str] = {}
    clusters: list[Cluster] = []
    n_merged = 0
    for key, members in groups.items():
        counts = dict(members)
        if len(counts) < 2 or sum(counts.values()) < config.min_cluster_size:
            continue
        canonical = _pick_canonical(members, config.canonical)
        for value, count in members:
            if value != canonical:
                mapping[value] = canonical
                n_merged += count
        ordered = tuple(sorted(counts, key=lambda v: (-counts[v], v)))
        clusters.append(
            Cluster(
                key=key,
                canonical=canonical,
                variants=ordered,
                size=sum(counts.values()),
                n_variants=len(counts),
            )
        )
    return mapping, clusters, n_merged


def cluster_column(
    df: Any, column: Any, *, config: ClusterConfig | None = None, method: str | None = None
) -> ClusterResult:
    """Cluster the values of one text column. Pure — returns the merge mapping only.

    *df* may be a pandas or polars DataFrame. ``method`` overrides
    ``config.method``; the combined ``"fingerprint_ngram"`` resolves to a single
    ``"fingerprint"`` pass here (use :func:`merge_clusters` for both passes).
    """
    cfg = config or ClusterConfig()
    resolved = method or cfg.method
    if resolved == "fingerprint_ngram":
        resolved = "fingerprint"
    if not _is_string_column(df, column):
        raise ValueError(f"cluster_column expects a text column; {column!r} is not string-like")
    rows = _keys_and_counts(df, column, resolved, cfg.ngram_size)
    mapping, clusters, n_merged = _clusters_from_counts(rows, cfg)
    return ClusterResult(
        column=str(column),
        method=resolved,
        n_clusters=len(clusters),
        n_cells_merged=n_merged,
        mapping=mapping,
        clusters=tuple(clusters),
    )


def _apply_mapping(df: Any, column: Any, mapping: Mapping[str, str]) -> Any:
    if not mapping:
        return df
    if is_polars_frame(df):
        pl = _polars_module()
        return df.with_columns(pl.col(column).replace(dict(mapping)))
    out = df.copy()
    out[column] = out[column].map(lambda v: mapping.get(v, v))
    return out


def merge_clusters(
    df: Any, columns: Sequence | None = None, config: ClusterConfig | None = None
) -> tuple[Any, list[ClusterResult]]:
    """Cluster and merge variant values in *df*, returning ``(df_same_type, results)``.

    Columns default to ``config.columns`` if set, else *columns*, else every text column.
    ``method="fingerprint_ngram"`` runs a fingerprint pass then an n-gram pass per column.
    """
    cfg = config or ClusterConfig()
    target = list(cfg.columns) or (list(columns) if columns is not None else _string_columns(df))
    methods = ["fingerprint", "ngram"] if cfg.method == "fingerprint_ngram" else [cfg.method]

    out = df
    results: list[ClusterResult] = []
    for column in target:
        if column not in _all_columns(out) or not _is_string_column(out, column):
            continue
        for method in methods:
            result = cluster_column(out, column, config=cfg, method=method)
            out = _apply_mapping(out, column, result.mapping)
            results.append(result)
    return out, results


# =====================================================================
# PII masking
# =====================================================================

#: Lookaround-free PII patterns — valid under both Polars (Rust regex) and Python ``re``.
PII_PATTERNS: dict[str, str] = {
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone": r"\+?\d[\d ()\-.]{7,}\d",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b",
    "ip": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
}


def _hash_value(value: Any, salt: str, length: int) -> str:
    # HMAC keyed by the salt — avoids the salt/value boundary ambiguity of a
    # plain ``sha256(salt + value)`` concatenation and is non-reversible
    # without the (per-rule random by default) salt.
    digest = hmac.new(salt.encode("utf-8"), str(value).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()[:length]


def _partial_value(value: Any, visible: int, placeholder: str) -> str:
    text = str(value)
    if visible <= 0:
        return placeholder
    return placeholder + text[-visible:]


def _scrub_patterns(rule: MaskingRule) -> list[str]:
    return [PII_PATTERNS[name] for name in rule.scrub_patterns] + list(rule.regexes)


def _resolve_columns(rule: MaskingRule, columns: Sequence) -> list:
    selected = [c for c in columns if c in set(rule.columns)]
    if rule.pattern:
        regex = re.compile(rule.pattern)
        selected += [c for c in columns if c not in selected and regex.search(str(c))]
    return selected


@dataclass
class MaskReport:
    """What :func:`mask_dataframe` masked, by column."""

    columns: dict[str, str] = field(default_factory=dict)
    cells_masked: dict[str, int] = field(default_factory=dict)
    rules_applied: list[str] = field(default_factory=list)

    def _record(self, column: str, rule: MaskingRule, n_cells: int) -> None:
        self.columns[column] = rule.strategy
        self.cells_masked[column] = self.cells_masked.get(column, 0) + int(n_cells)
        if rule.name not in self.rules_applied:
            self.rules_applied.append(rule.name)

    @property
    def total_cells_masked(self) -> int:
        return sum(self.cells_masked.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": dict(self.columns),
            "cells_masked": dict(self.cells_masked),
            "total_cells_masked": self.total_cells_masked,
            "rules_applied": list(self.rules_applied),
        }

    def __repr__(self) -> str:
        return (
            f"<MaskReport columns={len(self.columns)} "
            f"cells_masked={self.total_cells_masked}>"
        )


def _apply_mask_polars(df: Any, column: Any, rule: MaskingRule) -> tuple[Any, int]:
    pl = _polars_module()
    if rule.strategy == "drop":
        return df.drop(column), df.height
    series = df.get_column(column)
    nonnull = int(series.is_not_null().sum())
    base = pl.col(column).cast(pl.Utf8)
    if rule.strategy == "hash":
        expr = base.map_elements(
            lambda v: _hash_value(v, rule.salt, rule.hash_length) if v is not None else None,
            return_dtype=pl.Utf8,
        )
        return df.with_columns(expr.alias(column)), nonnull
    if rule.strategy == "redact":
        expr = pl.when(pl.col(column).is_not_null()).then(pl.lit(rule.placeholder)).otherwise(None)
        return df.with_columns(expr.alias(column)), nonnull
    if rule.strategy == "partial":
        expr = base.map_elements(
            lambda v: _partial_value(v, rule.visible, rule.placeholder) if v is not None else None,
            return_dtype=pl.Utf8,
        )
        return df.with_columns(expr.alias(column)), nonnull
    # regex_scrub
    expr = base
    for pattern in _scrub_patterns(rule):
        expr = expr.str.replace_all(pattern, rule.placeholder)
    new_values = df.select(expr.alias("__new")).get_column("__new")
    changed = int(((new_values != series.cast(pl.Utf8)) & series.is_not_null()).sum())
    return df.with_columns(expr.alias(column)), changed


def _apply_mask_pandas(
    df: pd.DataFrame, column: Any, rule: MaskingRule
) -> tuple[pd.DataFrame, int]:
    if rule.strategy == "drop":
        return df.drop(columns=[column]), len(df)
    out = df.copy()
    series = out[column]
    nonnull = int(series.notna().sum())
    if rule.strategy == "hash":
        out[column] = series.map(
            lambda v: _hash_value(v, rule.salt, rule.hash_length) if pd.notna(v) else v
        )
        return out, nonnull
    if rule.strategy == "redact":
        out[column] = series.map(lambda v: rule.placeholder if pd.notna(v) else v)
        return out, nonnull
    if rule.strategy == "partial":
        out[column] = series.map(
            lambda v: _partial_value(v, rule.visible, rule.placeholder) if pd.notna(v) else v
        )
        return out, nonnull
    # regex_scrub
    original = series.astype("string")
    scrubbed = original
    for pattern in _scrub_patterns(rule):
        scrubbed = scrubbed.str.replace(pattern, rule.placeholder, regex=True)
    changed = int(((scrubbed != original) & original.notna()).sum())
    out[column] = scrubbed
    return out, changed


def mask_dataframe(df: Any, rules: Sequence[MaskingRule]) -> tuple[Any, MaskReport]:
    """Apply PII masking *rules* to *df*; returns ``(df_same_type, MaskReport)``.

    Rules run in order; ``drop`` removes the column so later rules see the new schema.
    The input frame is never mutated.
    """
    report = MaskReport()
    out = df
    for rule in rules:
        for column in _resolve_columns(rule, _all_columns(out)):
            if column not in _all_columns(out):
                continue
            if is_polars_frame(out):
                out, n_cells = _apply_mask_polars(out, column, rule)
            else:
                out, n_cells = _apply_mask_pandas(out, column, rule)
            report._record(str(column), rule, n_cells)
    return out, report


# =====================================================================
# Semantic validation
# =====================================================================


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


class SemanticValidator(abc.ABC):
    """Validate a column's values against an external/reference notion of "valid".

    Subclasses implement :meth:`validate`, returning one boolean per input value
    (``True`` = valid). Null/missing values are reported as valid — completeness is a
    separate concern handled by the trust score.
    """

    name: str

    @abc.abstractmethod
    def validate(self, values: Sequence[Any]) -> list[bool]:
        ...


class ReferenceSetValidator(SemanticValidator):
    """Valid iff the value is a member of a reference set (e.g. ISO codes)."""

    def __init__(self, name: str, reference: Sequence[Any], *, case_sensitive: bool = False):
        self.name = name
        self.case_sensitive = case_sensitive
        self._reference = (
            {str(x) for x in reference}
            if case_sensitive
            else {str(x).casefold() for x in reference}
        )

    def validate(self, values: Sequence[Any]) -> list[bool]:
        out = []
        for value in values:
            if _is_null(value):
                out.append(True)
            else:
                key = str(value) if self.case_sensitive else str(value).casefold()
                out.append(key in self._reference)
        return out


class RegexValidator(SemanticValidator):
    """Valid iff the value fully matches a regular expression."""

    def __init__(self, name: str, pattern: str, *, case_sensitive: bool = True):
        self.name = name
        self._regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)

    def validate(self, values: Sequence[Any]) -> list[bool]:
        return [
            True if _is_null(v) else bool(self._regex.fullmatch(str(v))) for v in values
        ]


class CallableValidator(SemanticValidator):
    """Valid iff a user-supplied predicate returns truthy for the value."""

    def __init__(self, name: str, func: Callable[[Any], bool]):
        self.name = name
        self._func = func

    def validate(self, values: Sequence[Any]) -> list[bool]:
        return [True if _is_null(v) else bool(self._func(v)) for v in values]


def build_validator(config: SemanticValidatorConfig) -> SemanticValidator:
    """Construct a :class:`SemanticValidator` from a declarative config."""
    if config.kind == "reference":
        return ReferenceSetValidator(
            config.name, config.reference, case_sensitive=config.case_sensitive
        )
    return RegexValidator(  # kind == "regex" (validated by the config)
        config.name, config.regex or "", case_sensitive=config.case_sensitive
    )


@dataclass(frozen=True)
class ColumnValidation:
    """Validity outcome for one column under one validator."""

    column: str
    validator: str
    n_checked: int
    n_invalid: int
    valid_ratio: float
    invalid_samples: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "validator": self.validator,
            "n_checked": self.n_checked,
            "n_invalid": self.n_invalid,
            "valid_ratio": round(self.valid_ratio, 4),
            "invalid_samples": list(self.invalid_samples),
        }


@dataclass
class ValidationReport:
    """Per-column semantic validation results."""

    columns: dict[str, ColumnValidation] = field(default_factory=dict)

    @property
    def n_invalid_total(self) -> int:
        return sum(c.n_invalid for c in self.columns.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_invalid_total": self.n_invalid_total,
            "columns": {name: c.to_dict() for name, c in self.columns.items()},
        }

    def __repr__(self) -> str:
        return (
            f"<ValidationReport columns={len(self.columns)} "
            f"invalid={self.n_invalid_total}>"
        )


def validate_columns(
    df: Any, validators: Mapping[Any, SemanticValidator]
) -> ValidationReport:
    """Run each ``column -> validator`` over *df* (pandas or polars)."""
    frame = to_pandas(df)
    report = ValidationReport()
    for column, validator in validators.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        distinct = series.dropna().unique().tolist()
        flags = validator.validate(distinct)
        invalid_values = [value for value, ok in zip(distinct, flags) if not ok]
        invalid_set = set(invalid_values)
        n_invalid = int(series.isin(invalid_set).sum()) if invalid_set else 0
        n_checked = int(series.notna().sum())
        report.columns[str(column)] = ColumnValidation(
            column=str(column),
            validator=validator.name,
            n_checked=n_checked,
            n_invalid=n_invalid,
            valid_ratio=(1.0 - n_invalid / n_checked) if n_checked else 1.0,
            invalid_samples=tuple(str(v) for v in invalid_values[:5]),
        )
    return report


def run_semantic_validation(
    df: Any, configs: Sequence[SemanticValidatorConfig]
) -> ValidationReport:
    """Build validators from *configs* and validate each config's columns."""
    validators: dict[Any, SemanticValidator] = {}
    for config in configs:
        validator = build_validator(config)
        for column in config.columns:
            validators[column] = validator
    return validate_columns(df, validators)


# =====================================================================
# Optional Cleanlab integration (label noise / outliers)
# =====================================================================

_CLEANLAB_HINT = (
    "cleanlab is required for label-noise / outlier detection. "
    "Install it with: pip install 'freshdata-cleaner[cleanlab]'"
)


def _require_cleanlab() -> Any:
    try:
        import cleanlab
    except ImportError as exc:
        raise ImportError(_CLEANLAB_HINT) from exc
    return cleanlab


def detect_label_issues(
    labels: Any,
    pred_probs: Any,
    *,
    return_indices_ranked_by: str = "self_confidence",
    **kwargs: Any,
) -> Any:
    """Find likely-mislabeled rows with Cleanlab (optional dependency).

    *labels* are integer class indices; *pred_probs* an ``(n, k)`` array of
    out-of-sample predicted probabilities. Returns indices ranked by
    ``return_indices_ranked_by`` (a valid Cleanlab option). Raises a clear
    :class:`ImportError` if Cleanlab is not installed.
    """
    _require_cleanlab()
    from cleanlab.filter import find_label_issues  # pragma: no cover - needs cleanlab

    return find_label_issues(  # pragma: no cover - needs cleanlab
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=return_indices_ranked_by,
        **kwargs,
    )


def detect_outliers(features: Any, **kwargs: Any) -> Any:
    """Score rows by out-of-distribution-ness with Cleanlab (optional dependency)."""
    _require_cleanlab()
    from cleanlab.outlier import OutOfDistribution  # pragma: no cover - needs cleanlab

    ood = OutOfDistribution()  # pragma: no cover - needs cleanlab
    return ood.fit_score(features=features, **kwargs)  # pragma: no cover - needs cleanlab
