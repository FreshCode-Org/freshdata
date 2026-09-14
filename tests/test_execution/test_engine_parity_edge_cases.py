"""Native engines match the pandas reference on edge-case inputs.

Covers non-finite floats and exact missing counts (#199), all-empty columns
(#201), order/keep of full-row dedup (#202) and Unicode whitespace (#204).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata._util import PY_WHITESPACE
from freshdata.config import CleanConfig
from freshdata.execution import EngineConfig

pytest.importorskip("duckdb")
pytest.importorskip("polars")
pa = pytest.importorskip("pyarrow")

ENGINES = ("pandas", "polars", "duckdb")


def _config(**overrides) -> CleanConfig:
    return CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False, **overrides)


def _actions(report):
    return [(a.step, a.count) for a in report.actions]


# -- #204 whitespace ---------------------------------------------------------


def test_py_whitespace_is_exactly_the_str_isspace_set():
    python_whitespace = "".join(c for c in map(chr, range(0x110000)) if c.isspace())
    assert python_whitespace == PY_WHITESPACE


@pytest.mark.parametrize("engine", ENGINES)
def test_strip_matches_python_str_strip(engine):
    values = [f"{ch}x{i}{ch}" for i, ch in enumerate(PY_WHITESPACE)]
    values += ["\xa0N/A　", "\x1f\x1c"]  # a padded sentinel, whitespace-only text
    out = fd.clean(pd.DataFrame({"s": values}), config=_config(drop_empty_rows=False),
                   engine=engine)
    expected = [v.strip() for v in values[:-2]]
    assert out["s"].tolist()[:-2] == expected
    assert out["s"].iloc[-2:].isna().all()  # "N/A" and "" are sentinels once stripped


# -- #199 non-finite floats and exact missing counts ---------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_infinite_values_do_not_crash(engine):
    df = pd.DataFrame({"x": [1.0, np.inf, 2.0, -np.inf, 3.0]})
    out = fd.clean(df, config=_config(), engine=engine)
    assert len(out) == 5


def test_duckdb_mean_impute_with_infinite_values_matches_pandas():
    df = pd.DataFrame({"x": [1.0, np.inf, np.nan, 2.0], "k": [1, 2, 3, 4]})
    config = _config(impute="mean")
    ref = fd.clean(df, config=config, engine="pandas")
    out = fd.clean(df, config=config, engine="duckdb")
    assert out["x"].tolist() == ref["x"].tolist() == [1.0, np.inf, np.inf, 2.0]


def test_duckdb_clip_uses_finite_fences():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, np.inf, np.inf, np.inf, np.inf]})
    config = _config(outliers="clip", outlier_method="iqr")
    ref = fd.clean(df, config=config, engine="pandas")
    out = fd.clean(df, config=config, engine="duckdb")
    assert np.isfinite(out["x"]).all()
    assert out["x"].tolist() == ref["x"].tolist()


def test_duckdb_arrow_nan_counts_as_missing():
    table = pa.table({
        "x": pa.array([1.0, float("nan"), None, 4.0], from_pandas=False),
        "y": pa.array([float("nan")] * 4, from_pandas=False),
    })
    ref_out, ref = fd.clean(table.to_pandas(), config=_config(), engine="pandas",
                            return_report=True)
    out, report = fd.clean(table, config=_config(), engine="duckdb", return_report=True)
    assert report.missing_before == ref.missing_before == 6
    assert list(out.columns) == list(ref_out.columns) == ["x"]
    assert report.rows_after == ref.rows_after == 2
    assert _actions(report) == _actions(ref)


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_before_is_exact(engine):
    rng = np.random.default_rng(0)
    x = rng.normal(size=100_003)
    x[rng.random(100_003) < 0.2663] = np.nan
    _, report = fd.clean(pd.DataFrame({"x": x}), config=_config(), engine=engine,
                         return_report=True)
    assert report.missing_before == int(np.isnan(x).sum())


# -- #201 all-empty columns ----------------------------------------------------


def _all_empty() -> pd.DataFrame:
    return pd.DataFrame({"a": [np.nan, np.nan], "b": [None, None]})


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_all_empty_columns_match_pandas(engine):
    ref_out, ref = fd.clean(_all_empty(), config=_config(), engine="pandas", return_report=True)
    out, report = fd.clean(_all_empty(), config=_config(), engine=engine, return_report=True)
    assert list(out.columns) == list(ref_out.columns) == []
    assert len(out) == len(ref_out) == 2
    assert (report.rows_after, report.cols_after) == (ref.rows_after, ref.cols_after) == (2, 0)
    assert _actions(report) == _actions(ref)
    assert report.backend_differences == []


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
@pytest.mark.parametrize("output_format", ["polars", "arrow"])
def test_all_empty_columns_non_pandas_output_is_disclosed(engine, output_format):
    out, report = fd.clean(_all_empty(), config=_config(), engine=engine,
                           output_format=output_format, return_report=True)
    assert len(out.columns if output_format == "polars" else out.column_names) == 0
    assert any(d["step"] == "drop_empty_columns" for d in report.backend_differences)


def test_duckdb_native_handle_keeps_all_empty_columns_and_says_so():
    rel, report = fd.clean(_all_empty(), config=_config(), engine="duckdb",
                           output_format="duckdb", return_report=True)
    assert rel.columns == ["a", "b"]
    assert not any(a.step == "drop_empty_columns" for a in report.actions)
    assert any(d["step"] == "drop_empty_columns" for d in report.backend_differences)


# -- #202 dedup order and keep -------------------------------------------------


def _dup_frame(n: int) -> pd.DataFrame:
    if n <= 5:
        return pd.DataFrame({"a": [3, 1, 3, 2, 1]})
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "a": rng.integers(0, n // 10, n),
        "b": rng.choice(["x", "y", None], n),
        "c": rng.choice([1.5, np.nan], n),
    })


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("n", [5, 20_000])
def test_duckdb_dedup_matches_pandas_order_and_keep(keep, n):
    df = _dup_frame(n)
    config = _config(drop_duplicates=True, duplicate_keep=keep)
    ref = fd.clean(df, config=config, engine="pandas").reset_index(drop=True)
    engine_config = EngineConfig(engine="duckdb", duckdb_threads=4)
    out = fd.clean(df, config=config, engine_config=engine_config).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, ref, check_dtype=False)


def test_duckdb_dedup_keeps_order_for_parquet_sources(tmp_path):
    df = _dup_frame(20_000)
    path = str(tmp_path / "dups.parquet")
    df.to_parquet(path, index=False)
    config = _config(drop_duplicates=True, duplicate_keep="last")
    ref = fd.clean(df, config=config, engine="pandas").reset_index(drop=True)
    engine_config = EngineConfig(engine="duckdb", duckdb_threads=4)
    out = fd.clean(path, config=config, engine_config=engine_config).reset_index(drop=True)
    pd.testing.assert_frame_equal(out, ref, check_dtype=False)
