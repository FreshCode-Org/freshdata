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

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import pandas as pd

from ..adapters.polars import from_pandas, to_pandas
from .config import (  # noqa: F401  (configs re-exported for discoverability)
    BlockingRule,
    ComparisonLevel,
    EntityResolutionConfig,
)

_PAIRS_SAMPLE = 50
_CLUSTERS_SAMPLE = 50


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
    jaro = (
        matches / len1 + matches / len2 + (matches - transpositions) / matches
    ) / 3.0
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
class MatchPair:
    """A scored candidate pair."""

    left_id: Any
    right_id: Any
    match_probability: float
    match_weight: float
    comparison_vector: dict[str, float]
    decision: Literal["match", "possible_match", "non_match"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "match_probability": round(self.match_probability, 4),
            "match_weight": round(self.match_weight, 4),
            "comparison_vector": self.comparison_vector,
            "decision": self.decision,
        }


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
        }

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
            out.append(QualityFinding.create(
                severity=p.decision,
                step="entity_resolution",
                column=None,
                rule_name="duplicate_match",
                message=(f"records {p.left_id} & {p.right_id} {p.decision} "
                         f"(p={p.match_probability:.3f})"),
                row_selector=f"{p.left_id} <-> {p.right_id}",
                observed_value=p.comparison_vector,
                expected_condition="distinct entities",
                action_taken=p.decision,
                lineage_run_id=lineage_run_id,
                extra={"match_probability": round(p.match_probability, 4),
                       "match_weight": round(p.match_weight, 4)},
            ))
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


def _score_pairs(
    records: list[dict[str, Any]],
    candidate_pairs: list[tuple[int, int]],
    config: EntityResolutionConfig,
    ids: list[Any],
) -> list[MatchPair]:
    comparisons = config.comparisons
    pairs: list[MatchPair] = []
    for i, j in candidate_pairs:
        vec: dict[str, float] = {}
        weighted = 0.0
        wsum = 0.0
        logodds = 0.0
        for cmp in comparisons:
            sim = _compare(cmp, records[i].get(cmp.column), records[j].get(cmp.column))
            if sim is None:
                continue
            vec[cmp.column] = round(sim, 4)
            weighted += cmp.weight * sim
            wsum += cmp.weight
            logodds += cmp.weight * (2 * sim - 1)
        prob = weighted / wsum if wsum else 0.0
        if prob >= config.match_threshold:
            decision: Literal["match", "possible_match", "non_match"] = "match"
        elif prob >= config.clerical_review_threshold:
            decision = "possible_match"
        else:
            decision = "non_match"
        pairs.append(MatchPair(ids[i], ids[j], prob, logodds, vec, decision))
    return pairs


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
        join = (
            f"FROM _er_input {lp} JOIN _er_input {rp} "
            f"ON ({on}) AND {lp}._er_pos < {rp}._er_pos"
        )
        count = con.execute(f"SELECT count(*) {join}").fetchone()[0]
        _gate_max_pairs(int(count), config)
        rows = con.execute(
            f"SELECT {lp}._er_pos, {rp}._er_pos {join}"
        ).fetchall()
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


def _parse_blocking(sql: str) -> tuple[
    Callable[[dict[str, Any]], Any], Callable[[dict[str, Any]], Any]
]:
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
        canonical_pos = min(
            poss, key=lambda pos: (_missing_ratio(frame.iloc[pos]), str(ids[pos]))
        )
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
) -> Any:
    """Deduplicate *df* via probabilistic linkage.

    Returns ``(resolved_df, EntityResolutionReport)`` when ``return_report`` is
    true (the default), else just ``resolved_df`` (same frame type as the input,
    with an added ``cluster_id`` column). The input is never mutated.
    """
    frame = to_pandas(df).reset_index(drop=True)
    ids = _validate(frame, config)
    candidate_pairs, backend = _generate_candidates(frame, config)
    records = frame.to_dict("records")
    pairs = _score_pairs(records, candidate_pairs, config, ids)
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

    candidate_pairs, backend = _generate_candidates(combined, config)
    sources = combined["_er_source"].tolist()
    if config.link_type == "link_only":
        candidate_pairs = [(i, j) for i, j in candidate_pairs if sources[i] != sources[j]]

    records = combined.to_dict("records")
    pairs = _score_pairs(records, candidate_pairs, config, ids)
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
