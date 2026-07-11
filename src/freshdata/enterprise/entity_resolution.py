"""Probabilistic entity resolution at scale (Splink-style, DuckDB-backed).

This module adds a probabilistic record-linkage backend that complements the
existing single-column fuzzy clustering (:func:`freshdata.enterprise.cleaner.cluster_column`).
It is opt-in via :class:`~freshdata.enterprise.config.EntityResolutionConfig` and
scales the *blocking* (candidate-pair generation) step through DuckDB while
scoring candidate pairs in Python.

Pipeline:

1. **Blocking** — :class:`~freshdata.enterprise.config.BlockingRule` predicates
   generate candidate pairs (DuckDB self-join, or a pandas hash-join fallback).
   A hard ``max_pairs`` gate aborts before any cartesian explosion.
2. **Comparison** — each :class:`~freshdata.enterprise.config.ComparisonLevel`
   contributes weighted agreement evidence (exact, Jaro–Winkler, Levenshtein,
   numeric/date distance, phonetic Soundex, or custom SQL).
3. **Scoring** — evidence is combined into a 0–1 probability-like score and a
   log-odds-style match weight, with the comparison vector exposed for audit.
4. **Clustering** — connected components of matched pairs become entity
   clusters with a canonical record chosen by completeness.

This is **rule-weighted probabilistic linkage**, not full EM-trained Splink; we
do not claim Splink parity. The string-distance primitives are dependency-free
pure-Python implementations.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from .._util import sanitize_csv_formulas
from ..adapters.polars import from_pandas, to_pandas
from .config import (  # noqa: F401  (configs re-exported for discoverability)
    BlockingRule,
    ComparisonLevel,
    EntityResolutionConfig,
)

_PAIRS_SAMPLE = 50
_CLUSTERS_SAMPLE = 50
_PREVIEW_LEN = 24

#: Column-name substrings that, in the presence of a privacy config, mark a
#: field whose value previews should be redacted in explanations / review items.
_PII_COLUMN_HINTS = (
    "email",
    "phone",
    "ssn",
    "dob",
    "birth",
    "mrn",
    "patient",
    "insurance",
    "address",
    "name",
    "guardian",
    "loyalty",
    "passport",
    "license",
    "tax",
)


def _utcnow_iso() -> str:
    """Timezone-aware UTC timestamp (seconds precision) for audit fields."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class EntityResolutionError(RuntimeError):
    """Raised for ER misconfiguration or safety-gate violations."""


# =====================================================================
# Dependency-free string / numeric comparison primitives
# =====================================================================


def jaro_winkler(s1: str, s2: str, *, prefix_weight: float = 0.1) -> float:
    """Jaro–Winkler similarity in ``[0, 1]`` (pure-Python, no dependencies)."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    match_distance = max(len1, len2) // 2 - 1
    s1_matches = [False] * len1
    s2_matches = [False] * len2
    matches = 0
    for i in range(len1):
        lo = max(0, i - match_distance)
        hi = min(i + match_distance + 1, len2)
        for j in range(lo, hi):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1
    transpositions //= 2
    jaro = (matches / len1 + matches / len2 + (matches - transpositions) / matches) / 3.0
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break
    return jaro + prefix * prefix_weight * (1 - jaro)


def levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance (pure-Python)."""
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        cur = [i]
        for j, c2 in enumerate(s2, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (c1 != c2)))
        prev = cur
    return prev[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    longest = max(len(s1), len(s2))
    if longest == 0:
        return 1.0
    return 1.0 - levenshtein(s1, s2) / longest


def soundex(value: str) -> str:
    """American Soundex code (pure-Python)."""
    s = "".join(c for c in value.upper() if c.isalpha())
    if not s:
        return "0000"
    codes = {
        **dict.fromkeys("BFPV", "1"),
        **dict.fromkeys("CGJKQSXZ", "2"),
        **dict.fromkeys("DT", "3"),
        "L": "4",
        **dict.fromkeys("MN", "5"),
        "R": "6",
    }
    first = s[0]
    result = first
    prev = codes.get(first, "")
    for ch in s[1:]:
        code = codes.get(ch, "")
        if code and code != prev:
            result += code
        if ch not in "HW":
            prev = code
    return (result + "000")[:4]


# =====================================================================
# Result dataclasses
# =====================================================================


@dataclass
class FieldExplanation:
    """Per-field contribution to a pair's match weight (audit / clerical review)."""

    field: str
    left_value: str
    right_value: str
    similarity: float
    threshold: float
    weight: float
    contribution: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "similarity": round(self.similarity, 4),
            "threshold": self.threshold,
            "weight": self.weight,
            "contribution": round(self.contribution, 4),
            "rationale": self.rationale,
        }


@dataclass
class MatchPair:
    """A scored candidate pair."""

    left_id: Any
    right_id: Any
    match_probability: float
    match_weight: float
    comparison_vector: dict[str, float]
    decision: Literal["match", "possible_match", "non_match"]
    explanation: list[FieldExplanation] = field(default_factory=list)
    blocking_rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "match_probability": round(self.match_probability, 4),
            "match_weight": round(self.match_weight, 4),
            "comparison_vector": self.comparison_vector,
            "decision": self.decision,
            "blocking_rule_ids": list(self.blocking_rule_ids),
            "explanation": [e.to_dict() for e in self.explanation],
        }

    def explanation_text(self, *, max_fields: int = 6) -> str:
        """One-line human-readable rationale for the strongest contributing fields."""
        if not self.explanation:
            return (
                f"score {self.match_probability:.3f} "
                f"(weight {self.match_weight:+.2f}); no field-level detail"
            )
        ranked = sorted(self.explanation, key=lambda e: abs(e.contribution), reverse=True)
        parts = [
            f"{e.field}: {e.rationale} (contrib {e.contribution:+.2f})"
            for e in ranked[:max_fields]
        ]
        head = f"score {self.match_probability:.3f}, weight {self.match_weight:+.2f} — "
        return head + "; ".join(parts)


@dataclass
class EntityCluster:
    """A resolved entity (connected component of matched records)."""

    cluster_id: str
    record_ids: tuple[Any, ...]
    size: int
    canonical_record_id: Any
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "record_ids": list(self.record_ids),
            "size": self.size,
            "canonical_record_id": self.canonical_record_id,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class EntityResolutionReport:
    """Summary of a :func:`resolve_entities` / :func:`link_entities` run."""

    n_records: int
    n_candidate_pairs: int
    n_matches: int
    n_possible_matches: int
    n_clusters: int
    backend: str
    pairs: list[MatchPair] = field(default_factory=list)
    clusters: list[EntityCluster] = field(default_factory=list)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    #: Populated by :func:`apply_review_decisions` — counts of clerical decisions
    #: and how many pairs were promoted/demoted as a result.
    feedback_summary: dict[str, Any] = field(default_factory=dict)
    #: Populated by :func:`merge_entities` — per-cluster record of which source
    #: row contributed each output field of the golden record.
    golden_record_lineage: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pairs_sample(self) -> list[MatchPair]:
        return self.pairs[:_PAIRS_SAMPLE]

    @property
    def clusters_sample(self) -> list[EntityCluster]:
        return self.clusters[:_CLUSTERS_SAMPLE]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_records": self.n_records,
            "n_candidate_pairs": self.n_candidate_pairs,
            "n_matches": self.n_matches,
            "n_possible_matches": self.n_possible_matches,
            "n_clusters": self.n_clusters,
            "backend": self.backend,
            "pairs_sample": [p.to_dict() for p in self.pairs_sample],
            "clusters_sample": [c.to_dict() for c in self.clusters_sample],
            "runtime_metadata": self.runtime_metadata,
            "feedback_summary": self.feedback_summary,
            "golden_record_lineage": self.golden_record_lineage,
        }

    def to_frame(self, kind: Literal["pairs", "explanations"] = "pairs") -> pd.DataFrame:
        """Render the report's pairs (``kind="pairs"``) or their per-field
        explanations (``kind="explanations"``, one row per pair-field) as a frame.
        """
        if kind == "pairs":
            rows = []
            for p in self.pairs:
                row: dict[str, Any] = {
                    "left_id": p.left_id,
                    "right_id": p.right_id,
                    "match_probability": round(p.match_probability, 4),
                    "match_weight": round(p.match_weight, 4),
                    "decision": p.decision,
                    "blocking_rule_ids": ",".join(p.blocking_rule_ids),
                }
                for col, sim in p.comparison_vector.items():
                    row[f"cmp_{col}"] = sim
                rows.append(row)
            return pd.DataFrame(rows)
        if kind == "explanations":
            rows = []
            for p in self.pairs:
                for e in p.explanation:
                    d = e.to_dict()
                    d["left_id"] = p.left_id
                    d["right_id"] = p.right_id
                    d["decision"] = p.decision
                    rows.append(d)
            cols = [
                "left_id",
                "right_id",
                "decision",
                "field",
                "left_value",
                "right_value",
                "similarity",
                "threshold",
                "weight",
                "contribution",
                "rationale",
            ]
            return pd.DataFrame(rows, columns=cols if rows else None)
        raise ValueError(f"kind must be 'pairs' or 'explanations', got {kind!r}")

    def to_findings(self, *, lineage_run_id: str | None = None) -> list:
        """Project match / possible-match pairs into :class:`~freshdata.QualityFinding`.

        Each surviving pair is a candidate duplicate: ``match`` maps to ``error``,
        ``possible_match`` to ``warning``; non-matches are dropped.
        """
        from ..findings import QualityFinding

        out: list = []
        for p in self.pairs:
            if p.decision == "non_match":
                continue
            out.append(
                QualityFinding.create(
                    severity=p.decision,
                    step="entity_resolution",
                    column=None,
                    rule_name="duplicate_match",
                    message=(
                        f"records {p.left_id} & {p.right_id} {p.decision} "
                        f"(p={p.match_probability:.3f})"
                    ),
                    row_selector=f"{p.left_id} <-> {p.right_id}",
                    observed_value=p.comparison_vector,
                    expected_condition="distinct entities",
                    action_taken=p.decision,
                    lineage_run_id=lineage_run_id,
                    extra={
                        "match_probability": round(p.match_probability, 4),
                        "match_weight": round(p.match_weight, 4),
                    },
                )
            )
        return out

    def summary(self) -> str:
        return (
            f"entity resolution ({self.backend}): {self.n_records} record(s), "
            f"{self.n_candidate_pairs} candidate pair(s) → {self.n_matches} match(es), "
            f"{self.n_clusters} multi-record cluster(s)"
        )

    def __str__(self) -> str:
        return self.summary()


# =====================================================================
# Comparison evaluation
# =====================================================================


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v))


def _compare(cmp: ComparisonLevel, a: Any, b: Any) -> float | None:
    """Agreement in ``[0, 1]`` for one comparison level (``None`` = skip)."""
    if cmp.kind == "custom_sql":
        return None  # custom_sql is a DuckDB blocking-time construct
    if _is_missing(a) or _is_missing(b):
        return 0.0
    if cmp.kind == "exact":
        return 1.0 if str(a) == str(b) else 0.0
    if cmp.kind == "jaro_winkler":
        return jaro_winkler(str(a), str(b))
    if cmp.kind == "levenshtein":
        return levenshtein_similarity(str(a), str(b))
    if cmp.kind == "phonetic":
        return 1.0 if soundex(str(a)) == soundex(str(b)) else 0.0
    if cmp.kind == "numeric_distance":
        try:
            dist = abs(float(a) - float(b))
        except (TypeError, ValueError):
            return 0.0
        return _grade_distance(dist, cmp.threshold)
    if cmp.kind == "date_distance":
        try:
            dist = abs((pd.Timestamp(a) - pd.Timestamp(b)).days)
        except (TypeError, ValueError):
            return 0.0
        return _grade_distance(dist, cmp.threshold)
    return None  # pragma: no cover - guarded by config validation


def _grade_distance(dist: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0 if dist == 0 else 0.0
    return max(0.0, 1.0 - dist / threshold)


def _preview_value(v: Any, *, redact: bool) -> str:
    """Short, single-line preview of a field value; ``<redacted>`` when masked."""
    if redact:
        return "<redacted>"
    if _is_missing(v):
        return ""
    s = str(v).replace("\n", " ").strip()
    return s if len(s) <= _PREVIEW_LEN else s[: _PREVIEW_LEN - 1] + "…"


def _explain_field(
    cmp: ComparisonLevel, a: Any, b: Any, sim: float, *, redact: bool
) -> FieldExplanation:
    contribution = cmp.weight * (2 * sim - 1)
    if _is_missing(a) or _is_missing(b):
        rationale = f"{cmp.kind}: value missing on one side → no support"
    else:
        verdict = "supports" if contribution > 0 else "opposes"
        rationale = f"{cmp.kind}: sim={sim:.2f} (thr={cmp.threshold:g}) → {verdict} match"
    return FieldExplanation(
        field=cmp.column,
        left_value=_preview_value(a, redact=redact),
        right_value=_preview_value(b, redact=redact),
        similarity=sim,
        threshold=cmp.threshold,
        weight=cmp.weight,
        contribution=contribution,
        rationale=rationale,
    )


def _score_pairs(
    records: list[dict[str, Any]],
    candidate_pairs: list[tuple[int, int]],
    config: EntityResolutionConfig,
    ids: list[Any],
    *,
    redact_columns: frozenset[str] = frozenset(),
    parsed_rules: list[tuple[str, Callable, Callable]] | None = None,
) -> list[MatchPair]:
    comparisons = config.comparisons
    pairs: list[MatchPair] = []
    for i, j in candidate_pairs:
        vec: dict[str, float] = {}
        explanation: list[FieldExplanation] = []
        weighted = 0.0
        wsum = 0.0
        logodds = 0.0
        for cmp in comparisons:
            a, b = records[i].get(cmp.column), records[j].get(cmp.column)
            sim = _compare(cmp, a, b)
            if sim is None:
                continue
            vec[cmp.column] = round(sim, 4)
            weighted += cmp.weight * sim
            wsum += cmp.weight
            logodds += cmp.weight * (2 * sim - 1)
            explanation.append(_explain_field(cmp, a, b, sim, redact=cmp.column in redact_columns))
        prob = weighted / wsum if wsum else 0.0
        if prob >= config.match_threshold:
            decision: Literal["match", "possible_match", "non_match"] = "match"
        elif prob >= config.clerical_review_threshold:
            decision = "possible_match"
        else:
            decision = "non_match"
        rule_ids = (
            _pair_blocking_rule_ids(records[i], records[j], parsed_rules) if parsed_rules else ()
        )
        pairs.append(
            MatchPair(
                ids[i],
                ids[j],
                prob,
                logodds,
                vec,
                decision,
                explanation=explanation,
                blocking_rule_ids=rule_ids,
            )
        )
    return pairs


def blocking_rule_id(index: int) -> str:
    """Stable identifier for the *index*-th blocking rule of a config."""
    return f"block_{index:03d}"


def _parse_blocking_rules(
    config: EntityResolutionConfig,
) -> list[tuple[str, Callable, Callable]]:
    """Parse equi-join blocking rules into (id, left_key_fn, right_key_fn).

    Rules the pandas parser cannot handle (e.g. custom DuckDB SQL) are skipped;
    they simply won't be attributed to pairs in ``blocking_rule_ids``.
    """
    parsed: list[tuple[str, Callable, Callable]] = []
    for idx, rule in enumerate(config.blocking_rules):
        try:
            left_key, right_key = _parse_blocking(rule.sql)
        except (ValueError, KeyError):
            continue
        parsed.append((blocking_rule_id(idx), left_key, right_key))
    return parsed


def _pair_blocking_rule_ids(
    rec_a: dict[str, Any],
    rec_b: dict[str, Any],
    parsed_rules: list[tuple[str, Callable, Callable]],
) -> tuple[str, ...]:
    out: list[str] = []
    for rid, left_key, right_key in parsed_rules:
        ka, kb = left_key(rec_a), right_key(rec_b)
        kb2, ka2 = left_key(rec_b), right_key(rec_a)
        if (ka is not None and ka == kb) or (kb2 is not None and kb2 == ka2):
            out.append(rid)
    return tuple(out)


def redaction_columns(
    columns: Iterable[str],
    *,
    privacy: Any = None,
    extra: Iterable[str] | None = None,
) -> frozenset[str]:
    """Decide which columns should have their value previews redacted.

    When a truthy *privacy* config (e.g. a ``PIIDetectionConfig``) is supplied,
    columns whose names hint at PII (see :data:`_PII_COLUMN_HINTS`) are redacted;
    *extra* always adds explicit column names. With no privacy config and no
    *extra*, nothing is redacted.
    """
    cols = list(columns)
    redacted: set[str] = set(extra or ())
    if privacy:
        for col in cols:
            low = str(col).lower()
            if any(hint in low for hint in _PII_COLUMN_HINTS):
                redacted.add(col)
    return frozenset(c for c in redacted if c in set(cols) or c in (extra or ()))


# =====================================================================
# Blocking — DuckDB backend
# =====================================================================


def _require_duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised when duckdb absent
        raise ImportError(
            "DuckDB entity-resolution backend requires duckdb. "
            "Install with pip install freshdata-cleaner[entity-resolution]"
        ) from exc
    return duckdb


def _candidates_duckdb(
    frame: pd.DataFrame, config: EntityResolutionConfig
) -> list[tuple[int, int]]:
    duckdb = _require_duckdb()
    work = frame.assign(_er_pos=range(len(frame)))
    con = duckdb.connect(config.duckdb_path or ":memory:")
    try:
        con.register("_er_input", work)
        lp, rp = config.left_prefix, config.right_prefix
        on = " OR ".join(f"({b.sql})" for b in config.blocking_rules)
        join = f"FROM _er_input {lp} JOIN _er_input {rp} ON ({on}) AND {lp}._er_pos < {rp}._er_pos"
        count = con.execute(f"SELECT count(*) {join}").fetchone()[0]
        _gate_max_pairs(int(count), config)
        rows = con.execute(f"SELECT {lp}._er_pos, {rp}._er_pos {join}").fetchall()
    finally:
        con.close()
    return sorted({(int(a), int(b)) for a, b in rows})


# =====================================================================
# Blocking — pandas fallback (parses a SQL equi-join subset)
# =====================================================================

_FUNC_RE = re.compile(r"^(\w+)\s*\((.*)\)$", re.DOTALL)


def _make_expr(expr: str) -> Callable[[dict[str, Any]], Any]:
    """Compile a tiny SQL expression subset to a record→value function."""
    expr = expr.strip()
    m = _FUNC_RE.match(expr)
    if m:
        func = m.group(1).lower()
        args = _split_args(m.group(2))
        inner = _make_expr(args[0])
        if func == "lower":
            return lambda rec: _safe_str(inner(rec)).lower()
        if func == "upper":
            return lambda rec: _safe_str(inner(rec)).upper()
        if func == "trim":
            return lambda rec: _safe_str(inner(rec)).strip()
        if func == "left":
            n = int(args[1])
            return lambda rec: _safe_str(inner(rec))[:n]
        if func == "right":
            n = int(args[1])
            return lambda rec: _safe_str(inner(rec))[-n:]
        if func == "substr":
            start = int(args[1])
            length = int(args[2]) if len(args) > 2 else None
            return lambda rec: _substr(_safe_str(inner(rec)), start, length)
        raise EntityResolutionError(
            f"unsupported SQL function {func!r} in pandas blocking; use the duckdb backend"
        )
    # bare column reference, possibly prefixed (l.col / r.col)
    col = expr.split(".", 1)[1] if "." in expr else expr
    col = col.strip().strip('"')
    return lambda rec: rec.get(col)


def _safe_str(v: Any) -> str:
    return "" if _is_missing(v) else str(v)


def _substr(s: str, start: int, length: int | None) -> str:
    # SQL substr is 1-indexed.
    begin = max(0, start - 1)
    return s[begin : begin + length] if length is not None else s[begin:]


def _split_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    current = ""
    for ch in text:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            args.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        args.append(current)
    return [a.strip() for a in args]


def _parse_blocking(
    sql: str,
) -> tuple[Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]]:
    """Parse ``a = b [and c = d ...]`` into (left_key_fn, right_key_fn)."""
    predicates = re.split(r"\band\b", sql, flags=re.IGNORECASE)
    left_fns: list[Callable[[dict[str, Any]], Any]] = []
    right_fns: list[Callable[[dict[str, Any]], Any]] = []
    for pred in predicates:
        if "=" not in pred:
            raise EntityResolutionError(
                f"pandas blocking only supports equality predicates, got {pred!r}; "
                "use the duckdb backend for richer SQL"
            )
        lhs, rhs = pred.split("=", 1)
        left_fns.append(_make_expr(lhs))
        right_fns.append(_make_expr(rhs))

    def left_key(rec: dict[str, Any]) -> tuple[Any, ...] | None:
        vals = tuple(fn(rec) for fn in left_fns)
        return None if any(_is_missing(v) or v == "" for v in vals) else vals

    def right_key(rec: dict[str, Any]) -> tuple[Any, ...] | None:
        vals = tuple(fn(rec) for fn in right_fns)
        return None if any(_is_missing(v) or v == "" for v in vals) else vals

    return left_key, right_key


def _candidates_pandas(
    frame: pd.DataFrame, config: EntityResolutionConfig
) -> list[tuple[int, int]]:
    from collections import defaultdict

    records = frame.to_dict("records")
    n = len(records)
    pairs: set[tuple[int, int]] = set()
    for rule in config.blocking_rules:
        left_key, right_key = _parse_blocking(rule.sql)
        buckets: dict[Any, list[int]] = defaultdict(list)
        right_keys = [right_key(records[j]) for j in range(n)]
        for j, rk in enumerate(right_keys):
            if rk is not None:
                buckets[rk].append(j)
        for i in range(n):
            lk = left_key(records[i])
            if lk is None:
                continue
            for j in buckets.get(lk, ()):
                if i < j:
                    pairs.add((i, j))
                elif j < i:
                    pairs.add((j, i))
        _gate_max_pairs(len(pairs), config)
    return sorted(pairs)


def _gate_max_pairs(count: int, config: EntityResolutionConfig) -> None:
    if config.max_pairs is not None and count > config.max_pairs:
        raise EntityResolutionError(
            f"candidate pairs ({count}) exceed max_pairs ({config.max_pairs}); "
            "tighten blocking_rules or raise max_pairs explicitly"
        )


def _generate_candidates(
    frame: pd.DataFrame, config: EntityResolutionConfig
) -> tuple[list[tuple[int, int]], str]:
    if config.backend == "duckdb":
        return _candidates_duckdb(frame, config), "duckdb"
    return _candidates_pandas(frame, config), "pandas"


# =====================================================================
# Clustering
# =====================================================================


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _missing_ratio(row: pd.Series) -> float:
    return float(row.isna().sum()) / max(1, len(row))


def _build_clusters(
    frame: pd.DataFrame,
    pairs: list[MatchPair],
    ids: list[Any],
) -> tuple[list[EntityCluster], dict[int, str]]:
    from collections import defaultdict

    n = len(frame)
    id_to_pos = {ident: pos for pos, ident in enumerate(ids)}
    uf = _UnionFind(n)
    matched = [p for p in pairs if p.decision == "match"]
    for p in matched:
        uf.union(id_to_pos[p.left_id], id_to_pos[p.right_id])

    components: dict[int, list[int]] = defaultdict(list)
    for pos in range(n):
        components[uf.find(pos)].append(pos)

    conf_by_root: dict[int, list[float]] = defaultdict(list)
    for p in matched:
        conf_by_root[uf.find(id_to_pos[p.left_id])].append(p.match_probability)

    # Stable cluster IDs: order components by their smallest id (string key).
    ordered = sorted(components.values(), key=lambda poss: min(str(ids[p]) for p in poss))
    clusters: list[EntityCluster] = []
    cluster_of_pos: dict[int, str] = {}
    for idx, poss in enumerate(ordered):
        cid = f"er_{idx:06d}"
        for pos in poss:
            cluster_of_pos[pos] = cid
        canonical_pos = min(poss, key=lambda pos: (_missing_ratio(frame.iloc[pos]), str(ids[pos])))
        confs = conf_by_root.get(uf.find(poss[0]), [])
        confidence = sum(confs) / len(confs) if confs else 1.0
        clusters.append(
            EntityCluster(
                cluster_id=cid,
                record_ids=tuple(ids[p] for p in poss),
                size=len(poss),
                canonical_record_id=ids[canonical_pos],
                confidence=confidence,
            )
        )
    return clusters, cluster_of_pos


# =====================================================================
# Public API
# =====================================================================


def _validate(frame: pd.DataFrame, config: EntityResolutionConfig) -> list[Any]:
    if not config.comparisons:
        raise EntityResolutionError("entity resolution requires at least one ComparisonLevel")
    if not config.blocking_rules:
        raise EntityResolutionError(
            "entity resolution requires blocking_rules; the full cartesian product is "
            "disabled for safety"
        )
    if config.unique_id_column not in frame.columns:
        raise KeyError(f"unique_id_column {config.unique_id_column!r} not in frame")
    ids = frame[config.unique_id_column].tolist()
    if len(set(map(str, ids))) != len(ids):
        raise EntityResolutionError(
            f"unique_id_column {config.unique_id_column!r} must hold unique values"
        )
    return ids


def resolve_entities(
    df: Any,
    *,
    config: EntityResolutionConfig,
    return_report: bool = True,
    redact_columns: Iterable[str] | None = None,
    privacy: Any = None,
) -> Any:
    """Deduplicate *df* via probabilistic linkage.

    Returns ``(resolved_df, EntityResolutionReport)`` when ``return_report`` is
    true (the default), else just ``resolved_df`` (same frame type as the input,
    with an added ``cluster_id`` column). The input is never mutated.

    ``redact_columns`` / ``privacy`` control which value previews are masked in
    the per-field explanations (see :func:`redaction_columns`).
    """
    frame = to_pandas(df).reset_index(drop=True)
    ids = _validate(frame, config)
    redact = redaction_columns(frame.columns, privacy=privacy, extra=redact_columns)
    parsed_rules = _parse_blocking_rules(config)
    candidate_pairs, backend = _generate_candidates(frame, config)
    records = frame.to_dict("records")
    pairs = _score_pairs(
        records,
        candidate_pairs,
        config,
        ids,
        redact_columns=redact,
        parsed_rules=parsed_rules,
    )
    clusters, cluster_of_pos = _build_clusters(frame, pairs, ids)

    resolved = frame.copy()
    if config.output_clusters:
        resolved["cluster_id"] = [cluster_of_pos[pos] for pos in range(len(frame))]

    report = EntityResolutionReport(
        n_records=len(frame),
        n_candidate_pairs=len(candidate_pairs),
        n_matches=sum(1 for p in pairs if p.decision == "match"),
        n_possible_matches=sum(1 for p in pairs if p.decision == "possible_match"),
        n_clusters=sum(1 for c in clusters if c.size > 1),
        backend=backend,
        pairs=pairs,
        clusters=[c for c in clusters if c.size > 1],
        runtime_metadata={
            "link_type": config.link_type,
            "match_threshold": config.match_threshold,
            "clerical_review_threshold": config.clerical_review_threshold,
            "scoring": "rule_weighted_probabilistic_linkage",
        },
    )
    out = from_pandas(resolved, df)
    return (out, report) if return_report else out


def link_entities(
    left_df: Any,
    right_df: Any,
    *,
    config: EntityResolutionConfig,
    return_report: bool = True,
    redact_columns: Iterable[str] | None = None,
    privacy: Any = None,
) -> Any:
    """Link records across two frames (record linkage rather than dedupe).

    Stacks the inputs (tagged by source), generates candidate pairs, and — for
    ``link_type="link_only"`` — keeps only cross-source pairs. Returns
    ``(linked_df, EntityResolutionReport)`` when ``return_report`` is true.
    """
    left = to_pandas(left_df).reset_index(drop=True)
    right = to_pandas(right_df).reset_index(drop=True)
    left = left.assign(_er_source="left")
    right = right.assign(_er_source="right")
    combined = pd.concat([left, right], ignore_index=True)
    ids = _validate(combined, config)
    redact = redaction_columns(combined.columns, privacy=privacy, extra=redact_columns)
    parsed_rules = _parse_blocking_rules(config)

    candidate_pairs, backend = _generate_candidates(combined, config)
    sources = combined["_er_source"].tolist()
    if config.link_type == "link_only":
        candidate_pairs = [(i, j) for i, j in candidate_pairs if sources[i] != sources[j]]

    records = combined.to_dict("records")
    pairs = _score_pairs(
        records,
        candidate_pairs,
        config,
        ids,
        redact_columns=redact,
        parsed_rules=parsed_rules,
    )
    clusters, cluster_of_pos = _build_clusters(combined, pairs, ids)

    resolved = combined.copy()
    if config.output_clusters:
        resolved["cluster_id"] = [cluster_of_pos[pos] for pos in range(len(combined))]

    report = EntityResolutionReport(
        n_records=len(combined),
        n_candidate_pairs=len(candidate_pairs),
        n_matches=sum(1 for p in pairs if p.decision == "match"),
        n_possible_matches=sum(1 for p in pairs if p.decision == "possible_match"),
        n_clusters=sum(1 for c in clusters if c.size > 1),
        backend=backend,
        pairs=pairs,
        clusters=[c for c in clusters if c.size > 1],
        runtime_metadata={"link_type": config.link_type},
    )
    out = from_pandas(resolved, left_df)
    return (out, report) if return_report else out


# =====================================================================
# Ergonomic two-frame link wrapper (fd.link)
# =====================================================================

_LINK_ID = "_link_id"
LinkStrategy = Literal["exact", "fuzzy", "external"]


def _coerce_blocking(blocking: object) -> tuple[BlockingRule, ...]:
    """Coerce a blocking override (rule/str/sequence) to a BlockingRule tuple."""
    if blocking is None:
        return ()
    items = blocking if isinstance(blocking, (list, tuple)) else [blocking]
    rules: list[BlockingRule] = []
    for item in items:
        rules.append(item if isinstance(item, BlockingRule) else BlockingRule(sql=str(item)))
    return tuple(rules)


def _link_config(
    keys: Sequence[str],
    strategy: LinkStrategy,
    threshold: float,
    blocking: object,
    backend: str,
    review_threshold: float,
    left: pd.DataFrame,
) -> EntityResolutionConfig:
    """Build an EntityResolutionConfig from keys + strategy for exact/fuzzy linking."""
    rules = _coerce_blocking(blocking)
    if not rules:
        if strategy == "exact":
            sql = " AND ".join(f"l.{k} = r.{k}" for k in keys)
        else:  # fuzzy: block on the first key to bound the candidate space
            sql = f"l.{keys[0]} = r.{keys[0]}"
        rules = (BlockingRule(sql=sql, description=f"{strategy} block on {list(keys)}"),)

    comparisons: list[ComparisonLevel] = []
    for k in keys:
        if strategy == "fuzzy" and (
            pd.api.types.is_string_dtype(left[k]) or left[k].dtype == object
        ):
            comparisons.append(ComparisonLevel(column=k, kind="jaro_winkler", threshold=threshold))
        else:
            comparisons.append(ComparisonLevel(column=k, kind="exact"))

    match_threshold = 0.5 if strategy == "exact" else threshold
    clerical = min(review_threshold, match_threshold)
    return EntityResolutionConfig(
        enabled=True,
        backend=backend,  # type: ignore[arg-type]
        unique_id_column=_LINK_ID,
        blocking_rules=rules,
        comparisons=tuple(comparisons),
        match_threshold=match_threshold,
        clerical_review_threshold=clerical,
        link_type="link_only",
    )


def _external_report(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    adapter: Callable[..., Any],
    match_threshold: float,
    review_threshold: float,
) -> EntityResolutionReport:
    """Wrap an external matcher's candidate pairs in an explainable report.

    ``adapter(left, right, keys)`` must return an iterable of mappings with
    ``left_index``/``right_index`` (positional row indices) and ``score`` (0..1),
    optionally ``reason``. FreshData formats them as explainable ``MatchPair``s —
    it does not re-implement the external matcher.
    """
    candidates = adapter(left, right, list(keys))
    pairs: list[MatchPair] = []
    for cand in candidates:
        li = int(cand["left_index"])
        ri = int(cand["right_index"])
        score = float(cand["score"])
        decision: Literal["match", "possible_match", "non_match"] = (
            "match"
            if score >= match_threshold
            else "possible_match"
            if score >= review_threshold
            else "non_match"
        )
        reason = cand.get("reason", f"external matcher score {score:.3f}")
        explanation = [
            FieldExplanation(
                field=k,
                left_value=str(left.iloc[li][k]),
                right_value=str(right.iloc[ri][k]),
                similarity=score,
                threshold=match_threshold,
                weight=1.0,
                contribution=score,
                rationale=str(reason),
            )
            for k in keys
        ]
        pairs.append(
            MatchPair(
                left_id=f"L{li}",
                right_id=f"R{ri}",
                match_probability=score,
                match_weight=score,
                comparison_vector=dict.fromkeys(keys, score),
                decision=decision,
                explanation=explanation,
                blocking_rule_ids=("external",),
            )
        )
    return EntityResolutionReport(
        n_records=len(left) + len(right),
        n_candidate_pairs=len(pairs),
        n_matches=sum(1 for p in pairs if p.decision == "match"),
        n_possible_matches=sum(1 for p in pairs if p.decision == "possible_match"),
        n_clusters=0,
        backend="external",
        pairs=pairs,
        clusters=[],
        runtime_metadata={"link_type": "link_only", "strategy": "external"},
    )


def link(
    left: Any,
    right: Any,
    *,
    keys: Sequence[str],
    strategy: LinkStrategy = "exact",
    threshold: float = 0.85,
    blocking: object | None = None,
    backend: str = "pandas",
    adapter: Callable[..., Any] | None = None,
    review_threshold: float = 0.65,
    return_linked: bool = False,
) -> EntityResolutionReport | tuple[Any, EntityResolutionReport]:
    """Link records across two frames and explain every candidate match.

    The ergonomic, two-frame front door to FreshData's entity resolution: it
    builds the resolution config from ``keys`` + ``strategy`` and returns an
    :class:`EntityResolutionReport` carrying candidate pairs, confidence scores,
    per-field explanations, and a steward-reviewable structure (export with
    :func:`build_review_queue`). Positions FreshData as the candidate-generation
    and preprocessing layer for customer/vendor/counterparty/product matching —
    not a replacement for a dedicated matcher.

    Parameters
    ----------
    left, right:
        The two frames to link.
    keys:
        Columns to match on (must exist in both frames).
    strategy:
        ``"exact"`` — block and compare on exact key agreement; ``"fuzzy"`` —
        block on the first key and Jaro-Winkler-compare string keys at
        ``threshold``; ``"external"`` — delegate scoring to ``adapter`` (e.g. a
        Dedupe-backed callable) and format its candidate pairs explainably.
    threshold:
        Fuzzy agreement / match cut-off (0..1).
    blocking:
        Optional override — a ``BlockingRule``, a SQL predicate string, or a list
        thereof — replacing the default candidate-generation rule.
    backend:
        ``"pandas"`` (default, no optional deps) or ``"duckdb"`` for exact/fuzzy.
    adapter:
        Required for ``strategy="external"``: ``adapter(left, right, keys)``
        returning mappings with ``left_index``/``right_index``/``score``
        (and optional ``reason``).
    review_threshold:
        Score at/above which a non-match becomes a ``possible_match`` for review.
    return_linked:
        When True, also return the linked frame (exact/fuzzy only).

    Returns
    -------
    EntityResolutionReport, or ``(linked_frame, report)`` when ``return_linked``.
    """
    keys = list(keys)
    if not keys:
        raise ValueError("link requires at least one key column")
    left_pd, right_pd = to_pandas(left), to_pandas(right)
    for frame, side in ((left_pd, "left"), (right_pd, "right")):
        missing = [k for k in keys if k not in frame.columns]
        if missing:
            raise KeyError(f"{side} frame is missing key column(s): {missing}")

    if strategy == "external":
        if adapter is None:
            raise ValueError("strategy='external' requires an adapter= callable")
        report = _external_report(left_pd, right_pd, keys, adapter, threshold, review_threshold)
        return (left, report) if return_linked else report

    if strategy not in ("exact", "fuzzy"):
        raise ValueError(f"strategy must be exact|fuzzy|external, got {strategy!r}")

    left_tagged = left_pd.copy()
    right_tagged = right_pd.copy()
    left_tagged[_LINK_ID] = [f"L{i}" for i in range(len(left_tagged))]
    right_tagged[_LINK_ID] = [f"R{i}" for i in range(len(right_tagged))]
    config = _link_config(
        keys, strategy, threshold, blocking, backend, review_threshold, left_tagged
    )
    linked, report = link_entities(left_tagged, right_tagged, config=config, return_report=True)
    return (linked, report) if return_linked else report


# =====================================================================
# Reviewer queue subsystem
# =====================================================================


_REVIEW_FORMATS = ("csv", "jsonl", "parquet")
_DECISION_VALUES = ("accept", "reject", "manual_merge")


@dataclass
class ReviewQueueConfig:
    """How to turn surviving pairs into a human-review queue."""

    #: Which pair decisions become review items (default: clerical-review band).
    include_decisions: tuple[str, ...] = ("possible_match",)
    #: Hard cap on queue size after sorting (``None`` = no cap).
    max_items: int | None = None
    #: Ordering: most-likely first, least-likely first, or most-uncertain first.
    sort_by: Literal["score_desc", "score_asc", "uncertainty"] = "uncertainty"
    #: Field-preview columns to redact in the generated review items.
    redact_columns: tuple[str, ...] = ()
    #: Max fields rendered in the one-line explanation text.
    explanation_max_fields: int = 6

    def __post_init__(self) -> None:
        bad = set(self.include_decisions) - {"match", "possible_match", "non_match"}
        if bad:
            raise ValueError(f"include_decisions has unknown values: {sorted(bad)}")
        if self.max_items is not None and self.max_items < 0:
            raise ValueError("max_items must be >= 0 or None")


@dataclass
class ReviewItem:
    """One pair queued for human (clerical) adjudication."""

    item_id: str
    left_id: Any
    right_id: Any
    score: float
    match_weight: float
    comparison_vector: dict[str, float]
    blocking_rule_ids: tuple[str, ...]
    explanation: str
    created_at: str
    status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "score": round(self.score, 4),
            "match_weight": round(self.match_weight, 4),
            "comparison_vector": self.comparison_vector,
            "blocking_rule_ids": list(self.blocking_rule_ids),
            "explanation": self.explanation,
            "created_at": self.created_at,
            "status": self.status,
        }

    def to_flat(self) -> dict[str, Any]:
        """Tabular row (CSV/parquet) with nested fields JSON-encoded."""
        d = self.to_dict()
        d["comparison_vector"] = json.dumps(d["comparison_vector"], sort_keys=True)
        d["blocking_rule_ids"] = ",".join(self.blocking_rule_ids)
        return d


@dataclass
class ReviewDecision:
    """A clerical decision returned by a human reviewer."""

    decision: Literal["accept", "reject", "manual_merge"]
    left_id: Any = None
    right_id: Any = None
    item_id: str | None = None
    reviewer: str | None = None
    decided_at: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.decision not in _DECISION_VALUES:
            raise ValueError(f"decision must be one of {_DECISION_VALUES}, got {self.decision!r}")
        if self.left_id is None and self.right_id is None and self.item_id is None:
            raise ValueError("ReviewDecision needs item_id or left_id/right_id")

    @property
    def pair_key(self) -> tuple[str, str] | None:
        if self.left_id is None or self.right_id is None:
            return None
        return _pair_key(self.left_id, self.right_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "left_id": self.left_id,
            "right_id": self.right_id,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "decided_at": self.decided_at,
            "note": self.note,
        }


@dataclass
class ReviewQueueReport:
    """A built reviewer queue plus provenance for export / round-tripping."""

    items: list[ReviewItem]
    created_at: str
    config: ReviewQueueConfig
    source_summary: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.items)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([it.to_flat() for it in self.items])

    def summary(self) -> str:
        return f"review queue: {len(self.items)} item(s) (sorted by {self.config.sort_by})"

    def __str__(self) -> str:
        return self.summary()


def _pair_key(left_id: Any, right_id: Any) -> tuple[str, str]:
    """Order-independent key so (a, b) and (b, a) collide."""
    a, b = str(left_id), str(right_id)
    return (a, b) if a <= b else (b, a)


def _uncertainty(score: float, midpoint: float) -> float:
    # Smaller distance to the decision midpoint == more uncertain.
    return -abs(score - midpoint)


def build_review_queue(
    report: EntityResolutionReport,
    *,
    config: ReviewQueueConfig | None = None,
) -> ReviewQueueReport:
    """Generate a reviewer queue from a report's surviving pairs.

    By default this picks pairs whose ``decision == "possible_match"`` — the
    clerical-review band between the match and non-match thresholds.
    """
    cfg = config or ReviewQueueConfig()
    meta = report.runtime_metadata
    midpoint = 0.75
    if "match_threshold" in meta and "clerical_review_threshold" in meta:
        midpoint = (meta["match_threshold"] + meta["clerical_review_threshold"]) / 2

    selected = [p for p in report.pairs if p.decision in cfg.include_decisions]
    if cfg.sort_by == "score_desc":
        selected.sort(key=lambda p: p.match_probability, reverse=True)
    elif cfg.sort_by == "score_asc":
        selected.sort(key=lambda p: p.match_probability)
    else:  # uncertainty
        selected.sort(key=lambda p: _uncertainty(p.match_probability, midpoint), reverse=True)
    if cfg.max_items is not None:
        selected = selected[: cfg.max_items]

    redact = set(cfg.redact_columns)
    now = _utcnow_iso()
    items: list[ReviewItem] = []
    for idx, p in enumerate(selected):
        vec = dict(p.comparison_vector)
        if redact:
            # Comparison vectors are already similarities (not raw PII), but we
            # honour the redaction set by masking the explanation previews.
            pass
        explanation = p.explanation_text(max_fields=cfg.explanation_max_fields)
        if redact and p.explanation:
            masked = [
                replace(e, left_value="<redacted>", right_value="<redacted>")
                if e.field in redact
                else e
                for e in p.explanation
            ]
            tmp = MatchPair(
                p.left_id,
                p.right_id,
                p.match_probability,
                p.match_weight,
                vec,
                p.decision,
                explanation=masked,
                blocking_rule_ids=p.blocking_rule_ids,
            )
            explanation = tmp.explanation_text(max_fields=cfg.explanation_max_fields)
        items.append(
            ReviewItem(
                item_id=f"rev_{idx:06d}",
                left_id=p.left_id,
                right_id=p.right_id,
                score=p.match_probability,
                match_weight=p.match_weight,
                comparison_vector=vec,
                blocking_rule_ids=p.blocking_rule_ids,
                explanation=explanation,
                created_at=now,
            )
        )
    return ReviewQueueReport(
        items=items,
        created_at=now,
        config=cfg,
        source_summary={
            "n_pairs": len(report.pairs),
            "n_selected": len(items),
            "include_decisions": list(cfg.include_decisions),
            "backend": report.backend,
        },
    )


def _resolve_format(path: Path, fmt: str | None) -> str:
    if fmt is not None:
        if fmt not in _REVIEW_FORMATS:
            raise ValueError(f"format must be one of {_REVIEW_FORMATS}, got {fmt!r}")
        return fmt
    suffix = path.suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        return "jsonl"
    if suffix in ("parquet", "pq"):
        return "parquet"
    if suffix == "csv":
        return "csv"
    raise ValueError(f"cannot infer review format from {path.name!r}; pass format= explicitly")


def export_review_queue(
    report: EntityResolutionReport | ReviewQueueReport,
    path: str | Path,
    *,
    format: str | None = None,
    config: ReviewQueueConfig | None = None,
    sanitize_formulas: bool = True,
) -> Path:
    """Write a review queue to *path* as ``csv``, ``jsonl``, or ``parquet``.

    *report* may be a freshly-built :class:`ReviewQueueReport` or a raw
    :class:`EntityResolutionReport` (in which case a queue is built first).

    Review queues exist to be opened by humans in spreadsheets, so the
    ``csv`` format neutralizes formula-injection payloads by default: string
    cells starting with ``= + - @ <tab> <cr>`` are prefixed with ``'``
    (OWASP CSV-injection guidance). Pass ``sanitize_formulas=False`` for a
    byte-exact export. Other formats are never altered.
    """
    queue = (
        report
        if isinstance(report, ReviewQueueReport)
        else build_review_queue(report, config=config)
    )
    out = Path(path)
    fmt = _resolve_format(out, format)
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "jsonl":
        with out.open("w", encoding="utf-8") as fh:
            for it in queue.items:
                fh.write(json.dumps(it.to_dict(), default=str) + "\n")
    elif fmt == "csv":
        frame = queue.to_frame()
        if sanitize_formulas:
            frame = sanitize_csv_formulas(frame)
        frame.to_csv(out, index=False)
    else:  # parquet
        queue.to_frame().to_parquet(out, index=False)
    return out


def _coerce_id(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def load_review_decisions(path: str | Path, *, format: str | None = None) -> list[ReviewDecision]:
    """Load clerical decisions written back by a reviewer (csv/jsonl/parquet)."""
    src = Path(path)
    fmt = _resolve_format(src, format)
    rows: list[dict[str, Any]]
    if fmt == "jsonl":
        rows = []
        with src.open(encoding="utf-8") as fh:
            for raw in fh:
                text = raw.strip()
                if text:
                    rows.append(json.loads(text))
    elif fmt == "csv":
        rows = pd.read_csv(src).to_dict("records")
    else:  # parquet
        rows = pd.read_parquet(src).to_dict("records")

    decisions: list[ReviewDecision] = []
    for row in rows:
        decision = str(row.get("decision", "")).strip()
        if not decision:
            continue
        decisions.append(
            ReviewDecision(
                decision=decision,  # type: ignore[arg-type]
                left_id=_coerce_id(row.get("left_id")),
                right_id=_coerce_id(row.get("right_id")),
                item_id=_coerce_id(row.get("item_id")),
                reviewer=_coerce_id(row.get("reviewer")),
                decided_at=_coerce_id(row.get("decided_at")),
                note=str(row.get("note") or ""),
            )
        )
    return decisions


# =====================================================================
# Clerical-decision feedback & recalibration
# =====================================================================


def _recluster_from_pairs(pairs: list[MatchPair], n_records: int) -> list[EntityCluster]:
    """Rebuild multi-record clusters from match pairs alone (frame-independent)."""
    from collections import defaultdict

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[min(ra, rb)], parent[max(ra, rb)] = min(ra, rb), min(ra, rb)

    id_of: dict[str, Any] = {}
    matched = [p for p in pairs if p.decision == "match"]
    for p in matched:
        lk, rk = str(p.left_id), str(p.right_id)
        id_of.setdefault(lk, p.left_id)
        id_of.setdefault(rk, p.right_id)
        union(lk, rk)

    members: dict[str, list[str]] = defaultdict(list)
    for key in id_of:
        members[find(key)].append(key)
    conf: dict[str, list[float]] = defaultdict(list)
    for p in matched:
        conf[find(str(p.left_id))].append(p.match_probability)

    ordered = sorted(members.values(), key=min)
    clusters: list[EntityCluster] = []
    for idx, keys in enumerate(ordered):
        if len(keys) < 2:
            continue
        confs = conf.get(find(keys[0]), [])
        clusters.append(
            EntityCluster(
                cluster_id=f"er_{idx:06d}",
                record_ids=tuple(id_of[k] for k in sorted(keys)),
                size=len(keys),
                canonical_record_id=id_of[min(keys)],
                confidence=sum(confs) / len(confs) if confs else 1.0,
            )
        )
    return clusters


def apply_review_decisions(
    report: EntityResolutionReport,
    decisions: Sequence[ReviewDecision],
    *,
    config: EntityResolutionConfig | None = None,
    recalibrate: bool = False,
) -> EntityResolutionReport:
    """Fold clerical decisions back into a report.

    Accepted (and manual-merge) possible-matches are promoted to ``match`` and
    will cluster; rejected pairs are demoted to ``non_match`` and can never
    cluster. Returns a **new** report (the input is not mutated). Config is only
    recalibrated when ``recalibrate=True`` *and* a ``config`` is supplied — the
    safe default leaves your config untouched.
    """
    by_key: dict[tuple[str, str], ReviewDecision] = {}
    for d in decisions:
        key = d.pair_key
        if key is not None:
            by_key[key] = d

    counts = {"accept": 0, "reject": 0, "manual_merge": 0}
    promoted = demoted = 0
    new_pairs: list[MatchPair] = []
    for p in report.pairs:
        dec = by_key.get(_pair_key(p.left_id, p.right_id))
        if dec is None:
            new_pairs.append(p)
            continue
        counts[dec.decision] += 1
        before = p.decision
        if dec.decision in ("accept", "manual_merge"):
            after: Literal["match", "possible_match", "non_match"] = "match"
        else:
            after = "non_match"
        if before != "match" and after == "match":
            promoted += 1
        elif after == "non_match" and before in ("match", "possible_match"):
            demoted += 1
        new_pairs.append(replace(p, decision=after))

    clusters = _recluster_from_pairs(new_pairs, report.n_records)
    feedback = {
        "decisions": counts,
        "n_applied": sum(counts.values()),
        "n_promoted": promoted,
        "n_demoted": demoted,
        "updated_at": _utcnow_iso(),
    }
    if recalibrate and config is not None:
        new_config = recalibrate_weights(config, report, decisions)
        feedback["recalibrated_weights"] = {
            c.column: round(c.weight, 4) for c in new_config.comparisons
        }

    return replace(
        report,
        pairs=new_pairs,
        clusters=clusters,
        n_matches=sum(1 for p in new_pairs if p.decision == "match"),
        n_possible_matches=sum(1 for p in new_pairs if p.decision == "possible_match"),
        n_clusters=len(clusters),
        feedback_summary=feedback,
    )


def recalibrate_weights(
    config: EntityResolutionConfig,
    report: EntityResolutionReport,
    decisions: Sequence[ReviewDecision],
    *,
    learning_rate: float = 1.0,
    min_pairs: int = 1,
    clamp: tuple[float, float] = (0.5, 2.0),
) -> EntityResolutionConfig:
    """Return a **new** config with comparison weights nudged by clerical feedback.

    Simple, transparent heuristic (not EM): a field whose similarity tends to be
    *high on accepted* pairs and *low on rejected* pairs is discriminative, so its
    weight is scaled up; the reverse scales it down. The input config is never
    mutated. This is only applied when explicitly requested.
    """
    by_key: dict[tuple[str, str], str] = {}
    for d in decisions:
        key = d.pair_key
        if key is not None:
            by_key[key] = d.decision

    acc: dict[str, list[float]] = {c.column: [] for c in config.comparisons}
    rej: dict[str, list[float]] = {c.column: [] for c in config.comparisons}
    for p in report.pairs:
        verdict = by_key.get(_pair_key(p.left_id, p.right_id))
        if verdict is None:
            continue
        bucket = acc if verdict in ("accept", "manual_merge") else rej
        for col, sim in p.comparison_vector.items():
            if col in bucket:
                bucket[col].append(sim)

    lo, hi = clamp
    new_levels = []
    for c in config.comparisons:
        a_vals, r_vals = acc[c.column], rej[c.column]
        if len(a_vals) + len(r_vals) < min_pairs or not (a_vals or r_vals):
            new_levels.append(c)
            continue
        mean_acc = sum(a_vals) / len(a_vals) if a_vals else 0.0
        mean_rej = sum(r_vals) / len(r_vals) if r_vals else 0.0
        factor = max(lo, min(hi, 1.0 + learning_rate * (mean_acc - mean_rej)))
        new_levels.append(replace(c, weight=round(c.weight * factor, 6)))
    return replace(config, comparisons=tuple(new_levels))


# =====================================================================
# Golden-record merge policies
# =====================================================================


_GOLDEN_STRATEGIES = (
    "most_complete",
    "most_recent",
    "trusted_source",
    "non_null_prefer_left",
    "column_priority_map",
    "custom",
)


@dataclass
class GoldenRecordPolicy:
    """How to collapse a cluster of duplicate rows into one golden record.

    Strategies:

    - ``most_complete`` — keep the single row with the fewest missing values.
    - ``most_recent`` — keep the row with the latest ``timestamp_column``.
    - ``trusted_source`` — keep the row whose ``source_column`` value ranks
      earliest in ``source_priority``.
    - ``non_null_prefer_left`` — per field, take the first non-null value in
      cluster order (left-most record wins ties).
    - ``column_priority_map`` — per column, choose the value from the row whose
      ``source_column`` ranks earliest in that column's preference list
      (falling back to first non-null); columns absent from the map use
      ``non_null_prefer_left``.
    - ``custom`` — call ``custom(cluster_frame)`` and use the returned dict.
    """

    strategy: Literal[
        "most_complete",
        "most_recent",
        "trusted_source",
        "non_null_prefer_left",
        "column_priority_map",
        "custom",
    ] = "most_complete"
    timestamp_column: str | None = None
    source_column: str | None = None
    source_priority: tuple[str, ...] = ()
    column_priority_map: dict[str, tuple[str, ...]] | None = None
    custom: Callable[[pd.DataFrame], dict[str, Any]] | None = None
    id_column: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in _GOLDEN_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_GOLDEN_STRATEGIES}, got {self.strategy!r}"
            )
        if self.strategy == "most_recent" and not self.timestamp_column:
            raise ValueError("most_recent strategy requires timestamp_column")
        if self.strategy == "trusted_source" and not (self.source_column and self.source_priority):
            raise ValueError("trusted_source strategy requires source_column and source_priority")
        if self.strategy == "column_priority_map":
            if not self.column_priority_map:
                raise ValueError("column_priority_map strategy requires column_priority_map")
            if not self.source_column:
                raise ValueError("column_priority_map strategy requires source_column")
        if self.strategy == "custom" and self.custom is None:
            raise ValueError("custom strategy requires a custom callable")


def _completeness(row: pd.Series) -> int:
    return int(row.notna().sum())


def _source_rank(value: Any, priority: Sequence[str]) -> int:
    sval = str(value)
    try:
        return priority.index(sval)
    except ValueError:
        return len(priority)


def _merge_one_cluster(
    sub: pd.DataFrame,
    ids: list[Any],
    out_cols: list[str],
    policy: GoldenRecordPolicy,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (golden values, per-field source id) for one cluster sub-frame."""
    positions = list(range(len(sub)))

    def row_level(winner: int) -> tuple[dict[str, Any], dict[str, Any]]:
        row = sub.iloc[winner]
        wid = ids[winner]
        return ({c: row[c] for c in out_cols}, dict.fromkeys(out_cols, wid))

    if policy.strategy == "most_complete":
        winner = max(positions, key=lambda p: (_completeness(sub.iloc[p]), -p))
        return row_level(winner)

    if policy.strategy == "most_recent":
        ts = pd.to_datetime(sub[policy.timestamp_column], errors="coerce")
        winner = int(ts.reset_index(drop=True).idxmax()) if ts.notna().any() else 0
        return row_level(winner)

    if policy.strategy == "trusted_source":
        winner = min(
            positions,
            key=lambda p: (
                _source_rank(sub.iloc[p][policy.source_column], policy.source_priority),
                _completeness(sub.iloc[p]) * -1,
                p,
            ),
        )
        return row_level(winner)

    if policy.strategy == "custom":
        custom_values = policy.custom(sub.reset_index(drop=True))  # type: ignore[misc]
        return ({c: custom_values.get(c) for c in out_cols}, dict.fromkeys(out_cols, "custom"))

    # Field-level strategies: non_null_prefer_left and column_priority_map.
    values: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    col_map = policy.column_priority_map or {}
    for col in out_cols:
        order = positions
        if policy.strategy == "column_priority_map" and col in col_map:
            pref = col_map[col]
            order = sorted(
                positions,
                key=lambda p: (_source_rank(sub.iloc[p][policy.source_column], pref), p),
            )
        chosen_pos = order[0]
        chosen_val = sub.iloc[chosen_pos][col]
        for p in order:
            v = sub.iloc[p][col]
            if not _is_missing(v):
                chosen_pos, chosen_val = p, v
                break
        values[col] = chosen_val
        sources[col] = ids[chosen_pos]
    return values, sources


def merge_entities(
    df: Any,
    clusters: Sequence[EntityCluster],
    policy: GoldenRecordPolicy | None = None,
    *,
    id_column: str | None = None,
    report: EntityResolutionReport | None = None,
    return_lineage: bool = True,
) -> Any:
    """Collapse each cluster into one golden record using *policy*.

    Returns a golden-record frame (same type as *df*) with one row per cluster,
    carrying a ``cluster_id`` column. When ``return_lineage`` is true (default)
    returns ``(golden_df, lineage)`` where ``lineage`` records exactly which
    source row contributed each output field. If *report* is given, its
    ``golden_record_lineage`` is populated as a side effect.
    """
    pol = policy or GoldenRecordPolicy()
    frame = to_pandas(df).reset_index(drop=True)
    id_col = id_column or pol.id_column or "id"
    if id_col not in frame.columns:
        raise KeyError(f"id_column {id_col!r} not in frame")

    out_cols = [c for c in frame.columns if not str(c).startswith("_er")]
    by_id = {str(v): pos for pos, v in enumerate(frame[id_col].tolist())}

    golden_rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for cluster in clusters:
        member_pos = [by_id[str(rid)] for rid in cluster.record_ids if str(rid) in by_id]
        if not member_pos:
            continue
        sub = frame.iloc[member_pos].reset_index(drop=True)
        member_ids = [frame.iloc[p][id_col] for p in member_pos]
        values, sources = _merge_one_cluster(sub, member_ids, out_cols, pol)
        values = dict(values)
        values[id_col] = cluster.canonical_record_id
        values["cluster_id"] = cluster.cluster_id
        golden_rows.append(values)
        lineage.append(
            {
                "cluster_id": cluster.cluster_id,
                "golden_id": cluster.canonical_record_id,
                "strategy": pol.strategy,
                "member_ids": list(cluster.record_ids),
                "field_sources": sources,
            }
        )

    cols = out_cols + (["cluster_id"] if "cluster_id" not in out_cols else [])
    golden = pd.DataFrame(golden_rows, columns=cols)
    if report is not None:
        report.golden_record_lineage = lineage
    out = from_pandas(golden, df)
    return (out, lineage) if return_lineage else out
