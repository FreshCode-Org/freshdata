"""Native distinct-value semantic path (Phase 6).

The semantic stage only ever reasons about a column's *distinct values* and
their counts: the experts turn ``value_counts`` into repair proposals, the gate
judges them, and accepted repairs apply as a value->value mapping. That maps
cleanly onto columnar engines, so we do not need to pull a whole native frame
into pandas just to run semantics.

This module extracts the bounded distinct table *natively* (Polars group-by /
DuckDB ``GROUP BY``), scores it through the *exact same* pandas proposal + gate
pipeline the reference path uses (see :func:`resolve_replacements`), then maps
accepted repairs back natively with ``replace_strict`` / SQL ``CASE``. A
fallback event is recorded only when the native path genuinely cannot handle a
frame, dtype, or configuration — never a silent full-frame materialization.

Scope: the native path handles the default deterministic backend. When a
non-default backend (``memory`` / ``profile`` / ``embedding``) or a learned
profile/cleaning-memory is configured, :func:`can_run_native` returns ``False``
so the caller keeps the existing (correct) full-frame pandas path for those.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..config import CleanConfig
from ..report import CleanReport
from .apply import resolve_replacements, run_semantic
from .context import build_semantic_context
from .profiler import profile_proposals_native

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

__all__ = ["can_run_native", "run_semantic_native"]


def can_run_native(config: CleanConfig, *, memory: object | None, profile: object | None) -> bool:
    """Whether the native distinct path can serve this configuration exactly.

    Only the default deterministic backend is supported natively; learned
    memory/profile replay and the optional embedding backend still route through
    the pandas reference path so their results are byte-identical.
    """
    return (
        config.semantic_enabled
        and tuple(config.semantic_backends) == ("deterministic",)
        and memory is None
        and profile is None
    )


def _stringify(value: object) -> str:
    """Render a repair target as the string the native engines will parse.

    Booleans become ``"true"``/``"false"`` (what Polars/DuckDB boolean casts
    expect) and timestamps become ISO-8601; everything else uses ``str``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _target_family(values: Iterable[object]) -> str:
    """Classify accepted repair targets so we can pick a tight output dtype."""
    vals = list(values)
    if not vals:
        return "str"
    if all(isinstance(v, bool) for v in vals):
        return "bool"
    if all(isinstance(v, pd.Timestamp) for v in vals):
        return "datetime"
    if all(isinstance(v, str) for v in vals):
        return "str"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
        return "int"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return "float"
    return "mixed"


# --------------------------------------------------------------------------- #
# Orchestration                                                               #
# --------------------------------------------------------------------------- #


def run_semantic_native(
    frame: Any,
    config: CleanConfig,
    report: CleanReport,
    *,
    engine: str,
) -> Any:
    """Run the semantic stage over a *native* frame without materializing it.

    *frame* is a Polars ``DataFrame``/``LazyFrame`` or a DuckDB relation (the
    un-collected handle the native engines return). Returns the same kind of
    handle, with accepted deterministic semantic repairs applied natively.
    Records semantic actions in *report* exactly like the pandas path, plus a
    fallback event for anything the native path cannot apply in place.
    """
    if isinstance(frame, pd.DataFrame):
        # A native engine that materialized to pandas (e.g. duckdb -> pandas
        # output): the frame is already in memory because pandas output was
        # requested, so score it on the reference path — not a fallback.
        return run_semantic(frame, config, report)

    extractor = _make_extractor(frame, engine)
    if extractor is None:
        # Unknown handle type: honestly disclose and leave the frame untouched
        # (the caller only reaches here on an engine we advertised support for).
        report.record_fallback(
            engine, "semantic", "native semantic path: unsupported frame handle"
        )
        return frame

    labels = extractor.labels()
    limit = config.semantic_max_distinct_values
    stats = extractor.stats(labels)

    series_by_col: dict[str, pd.Series] = {}
    used_stats: dict[object, tuple[int, int, int | None]] = {}
    for col in labels:
        st = stats.get(col)
        if st is None:
            continue
        _n_rows, _n_nonnull, nunique = st
        # Mirror ``profiler._column_eligible``: only columns whose full-column
        # cardinality is within the budget are candidates — so the distinct pull
        # is always bounded and never scans an unbounded key column.
        if nunique is None or nunique == 0 or nunique > limit:
            continue
        series = extractor.distinct_series(col, limit)
        if series is None or series.empty:
            continue
        series_by_col[str(col)] = series
        used_stats[col] = st

    if not series_by_col:
        return frame

    distinct_df = pd.DataFrame(
        {col: pd.Series(list(s.values)) for col, s in series_by_col.items()}
    )
    ctx = build_semantic_context(distinct_df, config, stats=used_stats)
    proposals = list(profile_proposals_native(series_by_col, ctx))
    if not proposals:
        return frame

    replacements = resolve_replacements(proposals, config, ctx, report)
    replacements = _drop_protected(replacements, config)
    if not replacements:
        return frame

    result, fallbacks = extractor.apply(replacements)
    for reason in fallbacks:
        report.record_fallback(engine, "semantic", reason)
    return result


def _drop_protected(replacements: dict[str, dict], config: CleanConfig) -> dict[str, dict]:
    """Belt-and-suspenders guard: never emit a repair for a protected column.

    The policy gate already skips protected/id/target columns, so this is a
    no-op in practice — but it guarantees the native applier can *never* mutate a
    protected column even if a future backend regressed the gate.
    """
    protected = set(config.preserve_columns) | set(config.id_columns)
    if config.target_column:
        protected.add(config.target_column)
    return {c: m for c, m in replacements.items() if c not in protected}


def _make_extractor(frame: Any, engine: str) -> _Extractor | None:
    if engine == "polars":
        try:
            import polars as pl  # noqa: PLC0415 - optional dependency, loaded on demand
        except ImportError:  # pragma: no cover - engine reported polars
            return None
        if isinstance(frame, pl.LazyFrame):
            return _PolarsLazyExtractor(frame, pl)
        if isinstance(frame, pl.DataFrame):
            return _PolarsEagerExtractor(frame, pl)
        return None
    if engine == "duckdb":
        try:
            import duckdb  # noqa: PLC0415 - optional dependency, loaded on demand
        except ImportError:  # pragma: no cover - engine reported duckdb
            return None
        if isinstance(frame, duckdb.DuckDBPyRelation):
            return _DuckDBExtractor(frame, duckdb)
        return None
    return None


# --------------------------------------------------------------------------- #
# Extractor protocol + shared helpers                                         #
# --------------------------------------------------------------------------- #


class _Extractor:
    """Engine-specific distinct extraction + native repair application."""

    def labels(self) -> list[str]:
        raise NotImplementedError  # pragma: no cover

    def stats(self, labels: list[str]) -> dict[str, tuple[int, int, int | None]]:
        raise NotImplementedError  # pragma: no cover

    def distinct_series(self, col: str, limit: int) -> pd.Series | None:
        raise NotImplementedError  # pragma: no cover

    def apply(self, replacements: dict[str, dict]) -> tuple[Any, list[str]]:
        raise NotImplementedError  # pragma: no cover


def _distinct_series_from_pairs(values: list, counts: list) -> pd.Series:
    """Build the distinct-value series the experts consume.

    Its *values* are the distinct values (for role/shape inference in the
    context builder); its ``attrs['fd_value_counts']`` carries the true
    value->count table (consumed by ``experts._value_counts``).
    """
    series = pd.Series(values, dtype=object)
    series.attrs["fd_value_counts"] = pd.Series(list(counts), index=pd.Index(values))
    return series


# --------------------------------------------------------------------------- #
# Polars                                                                       #
# --------------------------------------------------------------------------- #


def _pl_coerce_expr(pl: Any, colname: str, family: str) -> Any:
    """Expression that tightens a repaired Utf8 column to *family*, or ``None``.

    Polars cannot cast Utf8 straight to Boolean/Datetime, so booleans go through
    ``replace_strict`` (``_stringify`` emits ``"true"``/``"false"``) and dates
    through ``str.to_datetime``; numerics use a non-strict cast. Unparseable
    values become null, which the caller detects to keep the safe Utf8 form.
    """
    e = pl.col(colname)
    if family == "int":
        return e.cast(pl.Int64, strict=False)
    if family == "float":
        return e.cast(pl.Float64, strict=False)
    if family == "bool":
        return e.replace_strict(
            ["true", "false"], [True, False], default=None, return_dtype=pl.Boolean
        )
    if family == "datetime":
        return e.str.to_datetime(strict=False)
    return None


def _pl_repair_expr(pl: Any, col: str, mapping: dict) -> Any:
    """A Utf8 ``replace_strict`` expression: mapped values swapped, rest kept."""
    old = [str(k) for k in mapping]
    new = [_stringify(v) for v in mapping.values()]
    base = pl.col(col).cast(pl.Utf8, strict=False)
    return base.replace_strict(old, new, default=base, return_dtype=pl.Utf8).alias(col)


class _PolarsEagerExtractor(_Extractor):
    def __init__(self, df: Any, pl: Any) -> None:
        self.df = df
        self.pl = pl

    def labels(self) -> list[str]:
        return list(self.df.columns)

    def _string_columns(self) -> set[str]:
        schema = self.df.schema
        return {c for c in self.df.columns if schema[c] in (self.pl.Utf8, self.pl.String)}

    def stats(self, labels: list[str]) -> dict[str, tuple[int, int, int | None]]:
        pl = self.pl
        n_rows = self.df.height
        if not labels:
            return {}
        aggs: list[Any] = []
        for c in labels:
            aggs.append(pl.col(c).null_count().alias(f"__null__{c}"))
            aggs.append(pl.col(c).drop_nulls().n_unique().alias(f"__nu__{c}"))
        row = self.df.select(aggs).row(0, named=True)
        out: dict[str, tuple[int, int, int | None]] = {}
        for c in labels:
            n_null = int(row[f"__null__{c}"])
            out[c] = (n_rows, n_rows - n_null, int(row[f"__nu__{c}"]))
        return out

    def distinct_series(self, col: str, limit: int) -> pd.Series | None:
        vc = self.df.get_column(col).drop_nulls().value_counts(sort=True)
        if vc.height == 0:
            return None
        count_col = next(c for c in vc.columns if c != col)
        return _distinct_series_from_pairs(
            vc.get_column(col).to_list(), vc.get_column(count_col).to_list()
        )

    def apply(self, replacements: dict[str, dict]) -> tuple[Any, list[str]]:
        pl = self.pl
        df = self.df
        string_cols = self._string_columns()
        fallbacks: list[str] = []
        exprs: list[Any] = []
        recast: dict[str, str] = {}
        non_string: dict[str, dict] = {}
        for col, mapping in replacements.items():
            if col in string_cols:
                exprs.append(_pl_repair_expr(pl, col, mapping))
                recast[col] = _target_family(mapping.values())
            else:
                non_string[col] = mapping
        if exprs:
            df = df.with_columns(exprs)
        for col, family in recast.items():
            expr = _pl_coerce_expr(pl, col, family)
            if expr is None:
                continue
            casted = df.select(expr.alias(col)).get_column(col)
            # Tighten dtype only when no *new* nulls appear (mirrors the pandas
            # ``_maybe_downcast`` guard): otherwise keep the safe Utf8 values.
            if casted.null_count() == df.get_column(col).null_count():
                df = df.with_columns(casted.alias(col))
        for col, mapping in non_string.items():
            fallbacks.append(
                f"native semantic apply for non-string column {col!r} used a "
                "bounded per-column pandas map"
            )
            mapped = df.get_column(col).to_pandas().map(lambda v, m=mapping: m.get(v, v))
            df = df.with_columns(pl.Series(col, mapped.to_list()))
        return df, fallbacks


class _PolarsLazyExtractor(_Extractor):
    def __init__(self, lf: Any, pl: Any) -> None:
        self.lf = lf
        self.pl = pl
        self._schema = lf.collect_schema()

    def labels(self) -> list[str]:
        return list(self._schema.names())

    def _string_columns(self) -> set[str]:
        return {c for c in self.labels() if self._schema[c] in (self.pl.Utf8, self.pl.String)}

    def stats(self, labels: list[str]) -> dict[str, tuple[int, int, int | None]]:
        pl = self.pl
        if not labels:
            return {}
        aggs: list[Any] = [pl.len().alias("__n_rows__")]
        for c in labels:
            aggs.append(pl.col(c).null_count().alias(f"__null__{c}"))
            aggs.append(pl.col(c).drop_nulls().n_unique().alias(f"__nu__{c}"))
        row = self.lf.select(aggs).collect().row(0, named=True)
        n_rows = int(row["__n_rows__"])
        out: dict[str, tuple[int, int, int | None]] = {}
        for c in labels:
            n_null = int(row[f"__null__{c}"])
            out[c] = (n_rows, n_rows - n_null, int(row[f"__nu__{c}"]))
        return out

    def distinct_series(self, col: str, limit: int) -> pd.Series | None:
        pl = self.pl
        vc = (
            self.lf.select(col)
            .filter(pl.col(col).is_not_null())
            .group_by(col)
            .len()
            .collect()
        )
        if vc.height == 0:
            return None
        count_col = next(c for c in vc.columns if c != col)
        return _distinct_series_from_pairs(
            vc.get_column(col).to_list(), vc.get_column(count_col).to_list()
        )

    def apply(self, replacements: dict[str, dict]) -> tuple[Any, list[str]]:
        pl = self.pl
        lf = self.lf
        string_cols = self._string_columns()
        fallbacks: list[str] = []
        exprs: list[Any] = []
        recast: dict[str, str] = {}
        for col, mapping in replacements.items():
            if col in string_cols:
                exprs.append(_pl_repair_expr(pl, col, mapping))
                recast[col] = _target_family(mapping.values())
            else:
                # Reinserting a materialized column would break laziness; keep the
                # frame out-of-core and disclose that this repair was not applied.
                fallbacks.append(
                    f"native lazy semantic path cannot apply repair to non-string "
                    f"column {col!r}; left unchanged"
                )
        if exprs:
            lf = lf.with_columns(exprs)
        cast_cols = [c for c, fam in recast.items() if _pl_coerce_expr(pl, c, fam) is not None]
        if cast_cols:
            # One bounded collect of null counts decides which tightenings are
            # lossless — the frame itself is never materialized.
            checks: list[Any] = []
            for c in cast_cols:
                checks.append(pl.col(c).is_null().sum().alias(f"__n__{c}"))
                checks.append(_pl_coerce_expr(pl, c, recast[c]).is_null().sum().alias(f"__c__{c}"))
            row = lf.select(checks).collect().row(0, named=True)
            safe = [c for c in cast_cols if row[f"__n__{c}"] == row[f"__c__{c}"]]
            if safe:
                lf = lf.with_columns([_pl_coerce_expr(pl, c, recast[c]).alias(c) for c in safe])
        return lf, fallbacks


# --------------------------------------------------------------------------- #
# DuckDB                                                                       #
# --------------------------------------------------------------------------- #

_DUCK_TYPE = {"int": "BIGINT", "float": "DOUBLE", "bool": "BOOLEAN"}

#: Keeps a parent DuckDB relation (and thus its connection) alive for the
#: lifetime of a derived relation returned to the caller.
_DUCK_RELATION_PARENTS: dict[int, Any] = {}


def _tie_relation_lifetime(child: Any, parent: Any) -> None:
    key = id(child)
    _DUCK_RELATION_PARENTS[key] = parent
    weakref.finalize(child, _DUCK_RELATION_PARENTS.pop, key, None)


class _DuckDBExtractor(_Extractor):
    def __init__(self, relation: Any, duckdb: Any) -> None:
        # Reuse the backend's identifier/literal quoting so the SQL we build here
        # is escaped identically to the deterministic pipeline's SQL.
        from ..execution.backends._duckdb import _lit, _q  # noqa: PLC0415 - avoid cycle

        self.rel = relation
        self.duckdb = duckdb
        self._q = _q
        self._lit = _lit

    def labels(self) -> list[str]:
        return list(self.rel.columns)

    def _string_columns(self) -> set[str]:
        return {c for c, t in zip(self.rel.columns, self.rel.types) if str(t) == "VARCHAR"}

    def stats(self, labels: list[str]) -> dict[str, tuple[int, int, int | None]]:
        if not labels:
            return {}
        parts = ["count(*) AS n_rows"]
        for i, c in enumerate(labels):
            parts.append(f"count({self._q(c)}) AS nn_{i}")
            parts.append(f"count(DISTINCT {self._q(c)}) AS nu_{i}")
        row = self.rel.aggregate(", ".join(parts)).fetchone()
        n_rows = int(row[0])
        out: dict[str, tuple[int, int, int | None]] = {}
        for i, c in enumerate(labels):
            out[c] = (n_rows, int(row[1 + 2 * i]), int(row[2 + 2 * i]))
        return out

    def distinct_series(self, col: str, limit: int) -> pd.Series | None:
        frame = (
            self.rel.filter(f"{self._q(col)} IS NOT NULL")
            .aggregate(f"{self._q(col)} AS v, count(*) AS cnt", self._q(col))
            .df()
        )
        if frame.empty:
            return None
        return _distinct_series_from_pairs(frame["v"].tolist(), frame["cnt"].tolist())

    def _case_expr(self, col: str, mapping: dict) -> str:
        whens = " ".join(
            f"WHEN {self._lit(str(old))} THEN {self._lit(_stringify(new))}"
            for old, new in mapping.items()
        )
        return f"CASE {self._q(col)} {whens} ELSE {self._q(col)} END AS {self._q(col)}"

    def apply(self, replacements: dict[str, dict]) -> tuple[Any, list[str]]:
        string_cols = self._string_columns()
        fallbacks: list[str] = []
        projection: list[str] = []
        recast: dict[str, str] = {}
        for col in self.rel.columns:
            mapping = replacements.get(col)
            if mapping is None:
                projection.append(self._q(col))
                continue
            if col not in string_cols:
                fallbacks.append(
                    f"native duckdb semantic path cannot apply repair to non-string "
                    f"column {col!r}; left unchanged"
                )
                projection.append(self._q(col))
                continue
            projection.append(self._case_expr(col, mapping))
            fam = _target_family(mapping.values())
            if fam in _DUCK_TYPE:
                recast[col] = fam
        rel = self.rel.project(", ".join(projection))
        rel = self._coerce(rel, recast)
        # The connection is closed by the *original* relation's finalizer; keep
        # that relation alive for as long as the derived one lives so the shared
        # connection stays open (the caller drops the original handle).
        _tie_relation_lifetime(rel, self.rel)
        return rel, fallbacks

    def _coerce(self, rel: Any, recast: dict[str, str]) -> Any:
        _q = self._q
        if not recast:
            return rel
        cols = list(recast)
        checks = []
        for i, c in enumerate(cols):
            t = _DUCK_TYPE[recast[c]]
            checks.append(
                f"count(*) FILTER (WHERE {_q(c)} IS NOT NULL AND "
                f"TRY_CAST({_q(c)} AS {t}) IS NULL) AS bad_{i}"
            )
        row = rel.aggregate(", ".join(checks)).fetchone()
        safe = {c for i, c in enumerate(cols) if int(row[i]) == 0}
        if not safe:
            return rel
        projection = [
            f"TRY_CAST({_q(c)} AS {_DUCK_TYPE[recast[c]]}) AS {_q(c)}" if c in safe else _q(c)
            for c in rel.columns
        ]
        return rel.project(", ".join(projection))
