"""Native engines keep results out of pandas unless pandas is requested.

* DuckDB fetches ``output_format="arrow"`` / ``"polars"`` directly instead of
  building a pandas frame with ``fetchdf()`` and converting it again (#52).
* Polars dedup row counts go through the streaming collect path instead of a
  plain in-memory ``collect()`` of the whole upstream plan (#53).
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata import EngineConfig

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")

from freshdata.execution.backends._polars import PolarsEngine  # noqa: E402


def _dupes_and_empty_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k": ["a", "a", None, "b", "b", "c", "a"],
            "v": [1.0, 1.0, None, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture
def no_fetchdf(monkeypatch):
    def fail(self, *args, **kwargs):
        raise AssertionError("fetchdf() must not run for non-pandas output")

    monkeypatch.setattr(duckdb.DuckDBPyRelation, "fetchdf", fail)
    monkeypatch.setattr(duckdb.DuckDBPyRelation, "df", fail)


@pytest.mark.parametrize(
    ("output_format", "kind"), [("arrow", pa.Table), ("polars", pl.DataFrame)]
)
def test_duckdb_fetches_requested_format_directly(
    no_fetchdf, small_df, native_config, output_format, kind
):
    out, report = fd.clean(
        small_df.copy(), config=native_config, engine="duckdb",
        output_format=output_format, return_report=True,
    )
    assert isinstance(out, kind)
    assert report.backend == "duckdb"
    assert report.rows_after == out.shape[0]
    assert report.cols_after == out.shape[1]


@pytest.mark.parametrize("output_format", ["arrow", "polars"])
def test_duckdb_direct_fetch_matches_pandas_output(small_df, native_config, output_format):
    expected, expected_report = fd.clean(
        small_df.copy(), config=native_config, engine="duckdb", return_report=True
    )
    out, report = fd.clean(
        small_df.copy(), config=native_config, engine="duckdb",
        output_format=output_format, return_report=True,
    )
    pd.testing.assert_frame_equal(
        out.to_pandas().reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert report.missing_after == expected_report.missing_after
    assert report.duplicates_removed == expected_report.duplicates_removed


@pytest.mark.parametrize("drop_duplicates", [True, False])
@pytest.mark.parametrize("streaming", [True, False])
def test_polars_dedup_counts_follow_the_engine_streaming_setting(
    monkeypatch, native_config, drop_duplicates, streaming
):
    engines: list[object] = []
    in_dedup = False
    original_collect = pl.LazyFrame.collect
    original_stage = PolarsEngine._stage_drop_duplicates

    def recording_collect(self, *args, **kwargs):
        if in_dedup:
            engines.append(kwargs.get("engine"))
        return original_collect(self, *args, **kwargs)

    def tracked_stage(self, *args, **kwargs):
        nonlocal in_dedup
        in_dedup = True
        try:
            return original_stage(self, *args, **kwargs)
        finally:
            in_dedup = False

    monkeypatch.setattr(pl.LazyFrame, "collect", recording_collect)
    monkeypatch.setattr(PolarsEngine, "_stage_drop_duplicates", tracked_stage)
    fd.clean(
        _dupes_and_empty_rows(), config=native_config, engine="polars",
        engine_config=EngineConfig(engine="polars", streaming=streaming),
        drop_duplicates=drop_duplicates, return_report=True,
    )
    assert engines, "the dedup stage should count rows"
    assert set(engines) == ({"streaming"} if streaming else {None})


@pytest.mark.parametrize("keep", ["first", "last"])
@pytest.mark.parametrize("subset", [None, ["k"]])
def test_polars_dedup_after_empty_row_drop_matches_pandas(native_config, keep, subset):
    options = {"drop_duplicates": True, "duplicate_keep": keep, "duplicate_subset": subset}
    expected, expected_report = fd.clean(
        _dupes_and_empty_rows(), config=native_config, engine="pandas",
        return_report=True, **options,
    )
    out, report = fd.clean(
        _dupes_and_empty_rows(), config=native_config, engine="polars",
        engine_config=EngineConfig(engine="polars", streaming_dedup=False),
        return_report=True, **options,
    )
    assert report.backend == "polars"
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected.reset_index(drop=True), check_dtype=False
    )
    assert report.duplicates_removed == expected_report.duplicates_removed
    assert report.rows_after == expected_report.rows_after
