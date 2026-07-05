"""Native distinct-value semantic path (Phase 6).

Proves the semantic stage runs over a *natively extracted* distinct table on the
Polars and DuckDB engines — repairs land, protected columns are untouchable, the
full frame is never materialized to pandas, output formats are preserved, and
anything the native path cannot serve degrades with a recorded fallback event.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig
from freshdata.execution import run_with_engine

pl = pytest.importorskip("polars")

SEMANTIC = {
    "strategy": "conservative",
    "fix_dtypes": False,
    "semantic_mode": "auto",
    "verbose": False,
}


def messy_frame(rows: int = 60) -> pd.DataFrame:
    """Boolean synonyms, spelled numbers, and a protected id column."""
    reps = rows // 6
    return pd.DataFrame(
        {
            "active": ["yes", "y", "YES", "no", "n", "NO"] * reps,
            "qty": ["one", "two", "three", "4", "5", "6"] * reps,
            "user_id": [f"U{i}" for i in range(6 * reps)],
        }
    )


# --------------------------------------------------------------------------- #
# Polars                                                                       #
# --------------------------------------------------------------------------- #


def test_polars_eager_repairs_apply():
    df = messy_frame()
    out, report = run_with_engine(
        pl.from_pandas(df), CleanConfig(**SEMANTIC),
        engine="polars", output_format="polars", return_report=True,
    )
    assert report.backend == "polars"
    assert isinstance(out, pl.DataFrame)  # output format preserved
    assert out.get_column("active").dtype == pl.Boolean
    assert out.get_column("active").to_list()[:2] == [True, True]
    assert out.get_column("qty").dtype == pl.Int64
    assert out.get_column("qty").to_list()[:3] == [1, 2, 3]


def test_polars_lazy_stays_lazy_and_repairs_apply():
    df = messy_frame()
    out, report = run_with_engine(
        pl.from_pandas(df).lazy(), CleanConfig(**SEMANTIC),
        engine="polars", output_format="polars-lazy", return_report=True,
    )
    assert isinstance(out, pl.LazyFrame)  # never collected
    collected = out.collect()
    assert collected.get_column("active").to_list()[:2] == [True, True]


def test_polars_protected_column_unchanged():
    df = messy_frame()
    out = run_with_engine(
        pl.from_pandas(df), CleanConfig(semantic_mode="auto", strategy="conservative",
                                        fix_dtypes=False, id_columns=["user_id"]),
        engine="polars", output_format="polars",
    )
    assert out.get_column("user_id").to_list() == df["user_id"].tolist()


def test_polars_distinct_path_never_materializes_full_frame(monkeypatch):
    # The proof that we do not fall back: materialize_to_pandas (the whole-frame
    # bridge) must never be called on the native semantic happy path.
    import freshdata.execution.backends._pandas as pmod

    def _boom(_source):  # pragma: no cover - only fires on regression
        raise AssertionError("full-frame materialize_to_pandas was called")

    monkeypatch.setattr(pmod, "materialize_to_pandas", _boom)
    out, report = run_with_engine(
        pl.from_pandas(messy_frame()), CleanConfig(**SEMANTIC),
        engine="polars", output_format="polars", return_report=True,
    )
    assert report.backend == "polars"
    assert out.get_column("active").to_list()[0] is True


def test_polars_high_cardinality_column_not_pulled():
    # A column whose true cardinality exceeds semantic_max_distinct_values is not
    # a semantic candidate (and its distinct table is never pulled) — mirrors the
    # pandas eligibility gate. It must pass through untouched.
    df = pd.DataFrame({"code": [f"c{i}" for i in range(200)]})
    cfg = CleanConfig(**{**SEMANTIC, "semantic_max_distinct_values": 50})
    out = run_with_engine(pl.from_pandas(df), cfg, engine="polars", output_format="polars")
    assert out.get_column("code").to_list() == df["code"].tolist()


def test_polars_matches_pandas_reference_where_representable():
    df = messy_frame()
    ref = fd.clean(df, **SEMANTIC)
    native = run_with_engine(
        pl.from_pandas(df), CleanConfig(**SEMANTIC), engine="polars", output_format="pandas",
    )
    # qty (fully numeric) and user_id (untouched) match exactly.
    assert native["qty"].tolist() == ref["qty"].tolist()
    assert native["user_id"].tolist() == ref["user_id"].tolist()


# --------------------------------------------------------------------------- #
# DuckDB                                                                       #
# --------------------------------------------------------------------------- #


def test_duckdb_pandas_output_repairs_apply():
    df = messy_frame()
    out, report = run_with_engine(
        df, CleanConfig(**SEMANTIC), engine="duckdb", output_format="pandas", return_report=True,
    )
    assert report.backend == "duckdb"
    assert set(out["active"]) == {True, False}
    assert out["qty"].tolist()[:3] == [1, 2, 3]


def test_duckdb_relation_output_repairs_apply_and_survive_gc():
    import gc

    df = messy_frame()
    rel, report = run_with_engine(
        df, CleanConfig(**SEMANTIC), engine="duckdb", output_format="duckdb", return_report=True,
    )
    gc.collect()  # the original relation handle is dropped; connection must persist
    frame = rel.df()
    assert report.backend == "duckdb"
    assert set(frame["active"]) == {True, False}
    assert frame["qty"].tolist()[:3] == [1, 2, 3]


def test_duckdb_protected_column_unchanged():
    df = messy_frame()
    out = run_with_engine(
        df, CleanConfig(semantic_mode="auto", strategy="conservative",
                        fix_dtypes=False, id_columns=["user_id"]),
        engine="duckdb", output_format="pandas",
    )
    assert out["user_id"].tolist() == df["user_id"].tolist()


# --------------------------------------------------------------------------- #
# Disclosed fallback for configurations the native path does not reproduce     #
# --------------------------------------------------------------------------- #


def test_non_default_backend_records_semantic_fallback():
    # A non-default semantic backend is not reproduced by the native distinct
    # path, so the whole clean is routed through pandas with a disclosed event
    # (never silently skipped, never silently degraded).
    df = messy_frame()
    cfg = CleanConfig(**{**SEMANTIC, "semantic_backends": ("deterministic", "memory")})
    _out, report = run_with_engine(
        df, cfg, engine="polars", output_format="pandas", return_report=True,
    )
    assert report.backend == "pandas"
    assert any(e["fallback_step"] == "semantic" for e in report.fallback_events)
