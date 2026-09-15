"""Spark engine fixes: atomic renames, ordered keep-aware dedup, NaN/inf, type names.

pyspark is optional, so most tests drive :class:`SparkEngine` through a small
in-memory stand-in for ``pyspark.sql`` (``_FakeFrame`` plus fake ``functions`` /
``Window`` modules) that evaluates the column expressions the engine builds with
Spark's null semantics. The tests at the bottom run the same repros on a real
local SparkSession and skip when pyspark or a JVM is unavailable.
"""

from __future__ import annotations

import datetime
import math
import operator
import sys
import types
from functools import cmp_to_key
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.execution._config import EngineConfig
from freshdata.execution._plan import PlanGenerator
from freshdata.execution.backends._spark import (
    SparkEngine,
    _renamed_columns,
    _spark_type_name,
)
from freshdata.report import CleanReport

NAN = float("nan")
INF = float("inf")


# ---------------------------------------------------------------------------
# A minimal evaluating stand-in for pyspark.sql
# ---------------------------------------------------------------------------


class AmbiguousColumnError(Exception):
    """Raised like Spark's AnalysisException when a name matches several columns."""


class _Row:
    """Case-insensitive, ambiguity-checking column lookup (Spark's default)."""

    def __init__(self, columns: list[str], values: list[Any]) -> None:
        self._columns = columns
        self._values = values

    def __getitem__(self, name: str) -> Any:
        hits = [i for i, c in enumerate(self._columns) if c.lower() == name.lower()]
        if not hits:
            raise KeyError(name)
        if len(hits) > 1:
            raise AmbiguousColumnError(name)
        return self._values[hits[0]]


def _and(a: Any, b: Any) -> Any:
    if a is False or b is False:
        return False
    if a is None or b is None:
        return None
    return True


def _or(a: Any, b: Any) -> Any:
    if a is True or b is True:
        return True
    if a is None or b is None:
        return None
    return False


class _Expr:
    def __init__(
        self,
        fn: Callable[[_Row], Any] | None = None,
        *,
        name: str | None = None,
        source: str | None = None,
        agg: Callable[[list[_Row]], Any] | None = None,
        window: Callable[[list[_Row]], list[Any]] | None = None,
        descending: bool = False,
        rank: bool = False,
    ) -> None:
        self.fn = fn
        self.name = name
        self.source = source
        self.agg = agg
        self.window = window
        self.descending = descending
        self.rank = rank

    def _copy(self, **changes: Any) -> _Expr:
        attrs = {
            k: getattr(self, k)
            for k in ("fn", "name", "source", "agg", "window", "descending", "rank")
        }
        attrs.update(changes)
        fn = attrs.pop("fn")
        return _Expr(fn, **attrs)

    def _binary(self, other: Any, op: Callable[[Any, Any], Any]) -> _Expr:
        rhs = _as_expr(other)

        def fn(row: _Row) -> Any:
            a, b = self.fn(row), rhs.fn(row)
            return None if a is None or b is None else op(a, b)

        return _Expr(fn)

    def __eq__(self, other: Any) -> _Expr:  # type: ignore[override]
        return self._binary(other, operator.eq)

    def __ne__(self, other: Any) -> _Expr:  # type: ignore[override]
        return self._binary(other, operator.ne)

    def __lt__(self, other: Any) -> _Expr:
        return self._binary(other, operator.lt)

    def __gt__(self, other: Any) -> _Expr:
        return self._binary(other, operator.gt)

    __hash__ = None  # type: ignore[assignment]

    def __and__(self, other: Any) -> _Expr:
        rhs = _as_expr(other)
        return _Expr(lambda r: _and(self.fn(r), rhs.fn(r)))

    def __or__(self, other: Any) -> _Expr:
        rhs = _as_expr(other)
        return _Expr(lambda r: _or(self.fn(r), rhs.fn(r)))

    def __invert__(self) -> _Expr:
        def fn(row: _Row) -> Any:
            v = self.fn(row)
            return None if v is None else not v

        return _Expr(fn)

    def isNull(self) -> _Expr:  # noqa: N802 - pyspark API
        return _Expr(lambda r: self.fn(r) is None)

    def isNotNull(self) -> _Expr:  # noqa: N802 - pyspark API
        return _Expr(lambda r: self.fn(r) is not None)

    def cast(self, _type: str) -> _Expr:
        return _Expr(lambda r: None if self.fn(r) is None else int(self.fn(r)))

    def alias(self, name: str) -> _Expr:
        return self._copy(name=name)

    def asc(self) -> _Expr:
        return self._copy(descending=False)

    def desc(self) -> _Expr:
        return self._copy(descending=True)

    def over(self, spec: _WindowSpec) -> _Expr:
        assert self.rank, "only row_number() is supported over a window"

        def window(rows: list[_Row]) -> list[Any]:
            groups: dict[tuple, list[int]] = {}
            for i, row in enumerate(rows):
                key = tuple(repr(p.fn(row)) for p in spec.partition)
                groups.setdefault(key, []).append(i)
            out: list[Any] = [None] * len(rows)
            for members in groups.values():
                for n, i in enumerate(_sorted_indices(rows, members, spec.order), start=1):
                    out[i] = n
            return out

        return _Expr(window=window)


def _as_expr(value: Any) -> _Expr:
    return value if isinstance(value, _Expr) else _Expr(lambda r: value)


def _sorted_indices(rows: list[_Row], indices: list[int], keys: list[_Expr]) -> list[int]:
    def compare(i: int, j: int) -> int:
        for key in keys:
            a, b = key.fn(rows[i]), key.fn(rows[j])
            if a == b:
                continue
            # Nulls sort first ascending, like Spark.
            before = a is None if a is None or b is None else a < b
            result = -1 if before else 1
            return -result if key.descending else result
        return 0

    return sorted(indices, key=cmp_to_key(compare))


class _When(_Expr):
    def __init__(self, branches: list[tuple[_Expr, _Expr]]) -> None:
        super().__init__()
        self.branches = branches

    def when(self, cond: _Expr, value: Any) -> _When:
        return _When([*self.branches, (cond, _as_expr(value))])

    def otherwise(self, value: Any) -> _Expr:
        default = _as_expr(value)
        branches = self.branches

        def fn(row: _Row) -> Any:
            for cond, result in branches:
                if cond.fn(row) is True:
                    return result.fn(row)
            return default.fn(row)

        return _Expr(fn, source=default.source)


def _values(expr: _Expr, rows: list[_Row]) -> list[Any]:
    return [v for v in (expr.fn(r) for r in rows) if v is not None]


def _mean(expr: _Expr) -> _Expr:
    def agg(rows: list[_Row]) -> Any:
        vals = _values(expr, rows)
        return sum(vals) / len(vals) if vals else None

    return _Expr(agg=agg)


def _stddev(expr: _Expr) -> _Expr:
    def agg(rows: list[_Row]) -> Any:
        vals = _values(expr, rows)
        if len(vals) < 2:
            return None
        m = sum(vals) / len(vals)
        return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

    return _Expr(agg=agg)


def _isnan(expr: _Expr) -> _Expr:
    return _Expr(lambda r: isinstance(expr.fn(r), float) and math.isnan(expr.fn(r)))


def _abs(expr: _Expr) -> _Expr:
    return _Expr(lambda r: None if expr.fn(r) is None else abs(expr.fn(r)))


def _coalesce(*exprs: _Expr) -> _Expr:
    def fn(row: _Row) -> Any:
        for e in exprs:
            v = e.fn(row)
            if v is not None:
                return v
        return None

    return _Expr(fn)


def _monotonically_increasing_id() -> _Expr:
    return _Expr(window=lambda rows: list(range(len(rows))))


_FUNCTIONS: dict[str, Any] = {
    "col": lambda name: _Expr(lambda r: r[name], source=name),
    "lit": _as_expr,
    "when": lambda cond, value: _When([(cond, _as_expr(value))]),
    "isnan": _isnan,
    "abs": _abs,
    "mean": _mean,
    "stddev": _stddev,
    "sum": lambda e: _Expr(agg=lambda rows: sum(_values(e, rows)) if _values(e, rows) else None),
    "count": lambda e: _Expr(agg=lambda rows: len(_values(e, rows))),
    "coalesce": _coalesce,
    "monotonically_increasing_id": _monotonically_increasing_id,
    "row_number": lambda: _Expr(rank=True),
}


class _WindowSpec:
    def __init__(self, partition: list[_Expr], order: list[_Expr]) -> None:
        self.partition = partition
        self.order = order

    def orderBy(self, *keys: _Expr) -> _WindowSpec:  # noqa: N802 - pyspark API
        return _WindowSpec(self.partition, list(keys))


class _Window:
    @staticmethod
    def partitionBy(*cols: _Expr) -> _WindowSpec:  # noqa: N802 - pyspark API
        return _WindowSpec(list(cols), [])


_FAKE_NUMERIC = {"tinyint", "smallint", "int", "bigint", "float", "double", "decimal"}


class _FakeFrame:
    """Just enough of ``pyspark.sql.DataFrame`` for the Spark engine's stages."""

    def __init__(self, schema: list[tuple[str, str]], rows: list[tuple]) -> None:
        self._schema = list(schema)
        self._rows = [list(r) for r in rows]
        assert all(len(r) == len(self._schema) for r in self._rows)

    @classmethod
    def from_columns(cls, schema: list[tuple[str, str]], data: dict[str, list]) -> _FakeFrame:
        names = [n for n, _ in schema]
        return cls(schema, list(zip(*(data[n] for n in names))))

    @property
    def columns(self) -> list[str]:
        return [n for n, _ in self._schema]

    @property
    def dtypes(self) -> list[tuple[str, str]]:
        return list(self._schema)

    def column(self, name: str) -> list[Any]:
        return [row[name] for row in self._views()]

    def _views(self) -> list[_Row]:
        return [_Row(self.columns, r) for r in self._rows]

    def _index(self, name: str) -> list[int]:
        return [i for i, c in enumerate(self.columns) if c.lower() == name.lower()]

    def _dtype_of(self, expr: _Expr) -> str:
        if expr.source is not None:
            hits = self._index(expr.source)
            if hits:
                return self._schema[hits[0]][1]
        return "double"

    def count(self) -> int:
        return len(self._rows)

    def first(self) -> _Row | None:
        views = self._views()
        return views[0] if views else None

    def select(self, *exprs: Any) -> _FakeFrame:
        flat = list(exprs[0]) if len(exprs) == 1 and isinstance(exprs[0], list) else list(exprs)
        views = self._views()
        if all(e.agg is not None for e in flat):
            names = [e.name or f"_{i}" for i, e in enumerate(flat)]
            return _FakeFrame([(n, "double") for n in names], [tuple(e.agg(views) for e in flat)])
        schema = [(e.name or e.source or f"_{i}", self._dtype_of(e)) for i, e in enumerate(flat)]
        return _FakeFrame(schema, [tuple(e.fn(v) for e in flat) for v in views])

    def withColumn(self, name: str, expr: _Expr) -> _FakeFrame:  # noqa: N802 - pyspark API
        views = self._views()
        values = expr.window(views) if expr.window is not None else [expr.fn(v) for v in views]
        hits = self._index(name)
        if hits:
            idx = hits[0]
            rows = [[*r[:idx], v, *r[idx + 1 :]] for r, v in zip(self._rows, values)]
            return _FakeFrame(self._schema, [tuple(r) for r in rows])
        dtype = "boolean" if all(isinstance(v, bool) for v in values) else "bigint"
        return _FakeFrame(
            [*self._schema, (name, dtype)], [(*r, v) for r, v in zip(self._rows, values)]
        )

    def filter(self, expr: _Expr) -> _FakeFrame:
        kept = [r for r, v in zip(self._rows, self._views()) if expr.fn(v) is True]
        return _FakeFrame(self._schema, [tuple(r) for r in kept])

    def orderBy(self, *keys: _Expr) -> _FakeFrame:  # noqa: N802 - pyspark API
        views = self._views()
        order = _sorted_indices(views, list(range(len(views))), list(keys))
        return _FakeFrame(self._schema, [tuple(self._rows[i]) for i in order])

    def drop(self, *names: str) -> _FakeFrame:
        keep = [i for i, c in enumerate(self.columns) if c not in names]
        return _FakeFrame(
            [self._schema[i] for i in keep], [tuple(r[i] for i in keep) for r in self._rows]
        )

    def toDF(self, *names: str) -> _FakeFrame:  # noqa: N802 - pyspark API
        assert len(names) == len(self._schema)
        return _FakeFrame(
            [(n, t) for n, (_, t) in zip(names, self._schema)], [tuple(r) for r in self._rows]
        )

    def withColumnRenamed(self, old: str, new: str) -> _FakeFrame:  # noqa: N802 - pyspark API
        # Spark renames every case-insensitive match.
        schema = [(new if n.lower() == old.lower() else n, t) for n, t in self._schema]
        return _FakeFrame(schema, [tuple(r) for r in self._rows])

    def dropDuplicates(self) -> _FakeFrame:  # noqa: N802 - pyspark API
        seen: dict[str, list] = {}
        for r in self._rows:
            seen.setdefault(repr(r), r)
        # A shuffle aggregate: surviving rows come back in no particular order.
        return _FakeFrame(self._schema, [tuple(r) for r in sorted(seen.values(), key=repr)])

    def approxQuantile(  # noqa: N802 - pyspark API
        self, name: str, probabilities: list[float], _error: float
    ) -> list[float]:
        dtype = self._schema[self._index(name)[0]][1]
        if _spark_type_name(dtype) not in _FAKE_NUMERIC:
            raise TypeError(f"approxQuantile requires a numeric column, got {dtype}")
        vals = sorted(v for v in self.column(name) if v is not None)
        if not vals:
            return []
        return [vals[max(0, math.ceil(p * len(vals)) - 1)] for p in probabilities]


@pytest.fixture
def fake_pyspark(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route ``from pyspark.sql import ...`` inside the engine to the fakes."""
    pkg = types.ModuleType("pyspark")
    sql = types.ModuleType("pyspark.sql")
    functions = types.ModuleType("pyspark.sql.functions")
    for name, obj in _FUNCTIONS.items():
        setattr(functions, name, obj)
    sql.functions = functions  # type: ignore[attr-defined]
    sql.Window = _Window  # type: ignore[attr-defined]
    pkg.sql = sql  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyspark", pkg)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)


def _run(frame: _FakeFrame, config: CleanConfig) -> tuple[_FakeFrame, CleanReport]:
    engine = SparkEngine()
    engine._session = lambda engine_config, source: None  # type: ignore[method-assign]
    engine._to_spark = lambda source, session: source  # type: ignore[method-assign]
    out, report = engine.execute(frame, config, EngineConfig(engine="spark"))
    assert report.backend == "spark", report.fallback_events
    return out, report


def _cfg(**overrides: Any) -> CleanConfig:
    return CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False, **overrides)


def _steps(report: CleanReport, step: str) -> list[tuple[str | None, int]]:
    return [(a.column, a.count) for a in report.actions if a.step == step]


# ---------------------------------------------------------------------------
# #333: exact numeric type names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dtype", "numeric", "integer"),
    [
        ("tinyint", True, True),
        ("smallint", True, True),
        ("int", True, True),
        ("bigint", True, True),
        ("float", True, False),
        ("double", True, False),
        ("decimal(10,2)", True, False),
        ("interval day to second", False, False),
        ("interval year to month", False, False),
        ("interval day", False, False),
        ("boolean", False, False),
        ("string", False, False),
        ("timestamp", False, False),
        ("array<int>", False, False),
        ("map<int,bigint>", False, False),
    ],
)
def test_numeric_type_names_match_exactly(dtype: str, numeric: bool, integer: bool) -> None:
    frame = types.SimpleNamespace(dtypes=[("c", dtype)])
    engine = SparkEngine()
    assert (engine._numeric_columns(frame) == ["c"]) is numeric
    assert engine._is_integer_column(frame, "c") is integer


def test_spark_type_name_strips_parameters() -> None:
    assert _spark_type_name("decimal(38,18)") == "decimal"
    assert _spark_type_name("interval day to second") == "interval day to second"


def test_interval_column_is_skipped_by_outliers_and_impute(fake_pyspark: None) -> None:
    durations = [datetime.timedelta(seconds=s) for s in (1, 2, 3, 400)]
    frame = _FakeFrame.from_columns(
        [("dur", "interval day to second"), ("v", "double")],
        {"dur": [*durations[:3], None], "v": [1.0, 2.0, 3.0, 400.0]},
    )
    out, report = _run(frame, _cfg(outliers="flag", impute="mean"))
    assert "dur_outlier" not in out.columns
    assert out.column("dur")[3] is None
    assert ("dur", 0) in _steps(report, "impute")  # reported as skipped, not filled


# ---------------------------------------------------------------------------
# #330: renames applied in one positional projection
# ---------------------------------------------------------------------------


def test_renamed_columns_is_simultaneous() -> None:
    assert _renamed_columns(["a", "b"], {"a": "b", "b": "a"}) == ["b", "a"]
    assert _renamed_columns(["a b", "a_b", "z"], {"a b": "a_b", "a_b": "a_b_2"}) == [
        "a_b",
        "a_b_2",
        "z",
    ]


@pytest.mark.parametrize(
    ("columns", "expected"),
    [(["a b", "a_b"], ["a_b", "a_b_2"]), (["A", "a"], ["a", "a_2"])],
)
def test_stage_rename_handles_collisions(columns: list[str], expected: list[str]) -> None:
    cfg = _cfg()
    plan = PlanGenerator(cfg).plan(columns)
    frame = _FakeFrame([(c, "string") for c in columns], [("x", "p"), ("y", "q")])
    report = CleanReport(backend="spark")
    out = SparkEngine()._stage_rename(frame, plan, report)
    assert out.columns == expected
    assert out.column(expected[0]) == ["x", "y"]
    assert out.column(expected[1]) == ["p", "q"]
    assert _steps(report, "column_names") == [(None, 2)]


def test_rename_collision_runs_end_to_end(fake_pyspark: None) -> None:
    frame = _FakeFrame.from_columns(
        [("a b", "double"), ("a_b", "double")], {"a b": [1.0, 2.0], "a_b": [3.0, 4.0]}
    )
    out, _ = _run(frame, _cfg())
    assert out.columns == ["a_b", "a_b_2"]
    assert out.column("a_b_2") == [3.0, 4.0]


# ---------------------------------------------------------------------------
# #331: dedup honours duplicate_keep and keeps row order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keep", ["first", "last"])
def test_dedup_matches_pandas_keep_and_order(fake_pyspark: None, keep: str) -> None:
    values = [3, 1, 3, 2, 1]
    frame = _FakeFrame.from_columns([("a", "bigint")], {"a": values})
    out, report = _run(frame, _cfg(drop_duplicates=True, duplicate_keep=keep))

    expected, pandas_report = fd.clean(
        pd.DataFrame({"a": values}),
        config=_cfg(drop_duplicates=True, duplicate_keep=keep),
        engine="pandas",
        return_report=True,
    )
    assert out.column("a") == expected["a"].tolist()
    assert out.columns == ["a"]  # helper columns are dropped
    assert report.duplicates_removed == pandas_report.duplicates_removed == 2


def test_dedup_keep_last_picks_last_occurrence_of_each_row(fake_pyspark: None) -> None:
    frame = _FakeFrame.from_columns(
        [("k", "bigint"), ("s", "string"), ("__fd_row_ordinal__", "bigint")],
        {"k": [1, 2, 1, 1], "s": [None, "x", None, "y"], "__fd_row_ordinal__": [7, 7, 7, 7]},
    )
    out = SparkEngine()._dedup_in_order(frame, "last")
    assert out.columns == ["k", "s", "__fd_row_ordinal__"]
    # (1, None) repeats at positions 0 and 2 (nulls compare equal); the last one survives.
    assert list(zip(out.column("k"), out.column("s"))) == [(2, "x"), (1, None), (1, "y")]


def test_dedup_detection_only_keeps_every_row(fake_pyspark: None) -> None:
    values = [3, 1, 3, 2, 1]
    frame = _FakeFrame.from_columns([("a", "bigint")], {"a": values})
    out, report = _run(frame, _cfg())
    assert out.column("a") == values
    assert report.duplicates_removed == 0
    assert any("detected 2 duplicate" in a.description for a in report.actions)


# ---------------------------------------------------------------------------
# #332: NaN is missing; fences ignore ±inf
# ---------------------------------------------------------------------------


def test_nan_is_missing_for_counts_and_empty_drops(fake_pyspark: None) -> None:
    data = {"a": [1.0, NAN, 3.0], "b": [NAN, NAN, NAN], "c": [1, None, None]}
    frame = _FakeFrame.from_columns([("a", "double"), ("b", "float"), ("c", "bigint")], data)
    out, report = _run(frame, _cfg())

    pandas_out, pandas_report = fd.clean(
        pd.DataFrame(data), config=_cfg(), engine="pandas", return_report=True
    )
    assert report.missing_before == pandas_report.missing_before == 6
    assert out.columns == list(pandas_out.columns) == ["a", "c"]
    assert report.columns_dropped == ["b"]
    assert out.count() == len(pandas_out) == 2
    assert report.missing_after == pandas_report.missing_after


def test_nan_as_null_leaves_non_float_columns_alone(fake_pyspark: None) -> None:
    frame = _FakeFrame.from_columns(
        [("d", "decimal(10,2)"), ("s", "string"), ("f", "double")],
        {"d": [1, 2], "s": ["NaN", None], "f": [NAN, 2.5]},
    )
    out = SparkEngine()._nan_as_null(frame)
    assert out.dtypes == frame.dtypes
    assert out.column("s") == ["NaN", None]
    assert out.column("f") == [None, 2.5]


def test_impute_mean_with_nulls_and_nan_fills_a_finite_value(fake_pyspark: None) -> None:
    # "k" keeps the null/NaN rows from being dropped as all-missing rows first.
    data = {"a": [1.0, NAN, None, 3.0], "k": [1, 2, 3, 4]}
    frame = _FakeFrame.from_columns([("a", "double"), ("k", "bigint")], data)
    out, report = _run(frame, _cfg(impute="mean"))

    pandas_out, pandas_report = fd.clean(
        pd.DataFrame({"a": pd.Series(data["a"], dtype="float64"), "k": data["k"]}),
        config=_cfg(impute="mean"),
        engine="pandas",
        return_report=True,
    )
    assert out.column("a") == pandas_out["a"].tolist() == [1.0, 2.0, 2.0, 3.0]
    assert _steps(report, "impute") == _steps(pandas_report, "impute") == [("a", 2)]


def test_outlier_fences_ignore_infinity_but_flag_it(fake_pyspark: None) -> None:
    data = {"a": [1.0, 2.0, 3.0, 4.0, 5.0, INF, -INF]}
    frame = _FakeFrame.from_columns([("a", "double")], data)
    cfg = _cfg(outliers="flag", outlier_method="zscore")

    bounds = SparkEngine()._outlier_bounds(frame, "a", "zscore", 3.0, cfg)
    assert bounds is not None
    std = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
    assert bounds == pytest.approx((3.0 - 3 * std, 3.0 + 3 * std))

    out, report = _run(frame, cfg)
    pandas_out, pandas_report = fd.clean(
        pd.DataFrame(data), config=cfg, engine="pandas", return_report=True
    )
    assert out.column("a_outlier") == pandas_out["a_outlier"].tolist()
    assert _steps(report, "outliers") == _steps(pandas_report, "outliers") == [("a", 2)]


def test_iqr_fences_ignore_infinity(fake_pyspark: None) -> None:
    frame = _FakeFrame.from_columns([("a", "double")], {"a": [1.0, 2.0, 3.0, 4.0, INF, INF]})
    bounds = SparkEngine()._outlier_bounds(frame, "a", "iqr", 1.5, _cfg(outliers="flag"))
    assert bounds is not None
    assert all(math.isfinite(b) for b in bounds)


# ---------------------------------------------------------------------------
# The same repros on a real SparkSession (skipped without pyspark + a JVM)
# ---------------------------------------------------------------------------


def _spark_clean(sdf: Any, cfg: CleanConfig) -> tuple[pd.DataFrame, CleanReport]:
    out, report = fd.clean(sdf, engine="spark", config=cfg, return_report=True)
    assert report.backend == "spark", report.fallback_events
    return out.toPandas(), report


def test_spark_rename_collision(spark_session: Any) -> None:
    sdf = spark_session.createDataFrame([("x", "p"), ("y", "q")], ["a b", "a_b"])
    out, _ = _spark_clean(sdf, _cfg())
    assert list(out.columns) == ["a_b", "a_b_2"]
    assert out["a_b_2"].tolist() == ["p", "q"]


@pytest.mark.parametrize("keep", ["first", "last"])
def test_spark_dedup_keep_and_order(spark_session: Any, keep: str) -> None:
    values = [3, 1, 3, 2, 1]
    sdf = spark_session.createDataFrame([(v,) for v in values], "a bigint").coalesce(1)
    cfg = _cfg(drop_duplicates=True, duplicate_keep=keep)
    out, report = _spark_clean(sdf, cfg)
    expected = fd.clean(pd.DataFrame({"a": values}), config=cfg, engine="pandas")
    assert out["a"].tolist() == expected["a"].tolist()
    assert list(out.columns) == ["a"]
    assert report.duplicates_removed == 2


def test_spark_nan_is_missing(spark_session: Any) -> None:
    rows = [(1.0, NAN, 1), (NAN, NAN, None), (3.0, NAN, None)]
    sdf = spark_session.createDataFrame(rows, "a double, b double, c bigint")
    out, report = _spark_clean(sdf, _cfg())
    assert report.missing_before == 6
    assert list(out.columns) == ["a", "c"]
    assert report.columns_dropped == ["b"]
    assert len(out) == 2


def test_spark_impute_mean_with_nulls_and_nan(spark_session: Any) -> None:
    rows = [(1.0, 1), (NAN, 2), (None, 3), (3.0, 4)]
    sdf = spark_session.createDataFrame(rows, "a double, k bigint")
    out, report = _spark_clean(sdf, _cfg(impute="mean"))
    assert out["a"].tolist() == [1.0, 2.0, 2.0, 3.0]
    assert _steps(report, "impute") == [("a", 2)]


def test_spark_outlier_fences_ignore_infinity(spark_session: Any) -> None:
    rows = [(v,) for v in (1.0, 2.0, 3.0, 4.0, 5.0, INF, -INF)]
    sdf = spark_session.createDataFrame(rows, "a double")
    out, report = _spark_clean(sdf, _cfg(outliers="flag", outlier_method="zscore"))
    assert out["a_outlier"].tolist() == [False] * 5 + [True, True]
    assert _steps(report, "outliers") == [("a", 2)]


def test_spark_interval_column_is_not_numeric(spark_session: Any) -> None:
    pyspark = pytest.importorskip("pyspark")
    major, minor = (int(p) for p in pyspark.__version__.split(".")[:2])
    if (major, minor) < (3, 3):
        pytest.skip("ANSI interval types need Spark >= 3.3")
    sdf = spark_session.range(4).selectExpr(
        "make_dt_interval(0, 0, 0, CAST(id * id * 100 AS DECIMAL(18, 6))) AS dur",
        "CAST(id AS DOUBLE) AS v",
    )
    assert dict(sdf.dtypes)["dur"].startswith("interval")
    out, _ = _spark_clean(sdf, _cfg(outliers="flag", impute="mean"))
    assert "dur_outlier" not in out.columns
