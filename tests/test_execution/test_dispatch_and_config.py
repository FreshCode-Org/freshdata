"""Metadata dispatch over Arrow/polars sources and engine construction."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.execution._config import ENGINE_NAMES, EngineConfig, EngineSelector
from freshdata.execution._metadata import MetadataScanner

pa = pytest.importorskip("pyarrow")
pl = pytest.importorskip("polars")


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3, None], "b": ["x", "y", "x", "z"]})


def test_from_source_pandas(frame):
    meta = MetadataScanner.from_source(frame, "pandas")
    assert {m.name for m in meta} == {"a", "b"}


def test_from_source_arrow_table(frame):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = MetadataScanner.from_source(table, "duckdb")
    assert {m.name for m in meta} == {"a", "b"}
    assert any(m.is_numeric for m in meta)


def test_from_source_arrow_record_batch(frame):
    batch = pa.Table.from_pandas(frame, preserve_index=False).to_batches()[0]
    meta = MetadataScanner.from_source(batch, "duckdb")
    assert {m.name for m in meta} == {"a", "b"}


def test_from_source_polars(frame):
    pl_df = pl.from_pandas(frame)
    assert {m.name for m in MetadataScanner.from_source(pl_df, "polars")} == {"a", "b"}
    assert {m.name for m in MetadataScanner.from_source(pl_df.lazy(), "polars")} == {"a", "b"}


def test_from_source_rejects_unknown():
    with pytest.raises(TypeError):
        MetadataScanner.from_source(object(), "pandas")


def test_spark_in_engine_names():
    assert "spark" in ENGINE_NAMES


def test_get_engine_constructs_each_backend():
    for name in ("pandas", "polars", "duckdb", "spark"):
        engine = EngineSelector.get_engine(name, EngineConfig(engine=name))
        assert engine.name == name


def test_engine_config_spark_options():
    cfg = EngineConfig(engine="spark", output_format="spark",
                       spark_shuffle_partitions=4)
    assert cfg.spark_shuffle_partitions == 4
    assert cfg.output_format == "spark"
