"""Spark backend: same audit contract as pandas, when a JVM is available.

These tests skip automatically when pyspark is not installed or no JVM can start
(see the ``spark_session`` fixture). When Spark runs, the cleaned frame and its
CleanReport must agree with the pandas reference on the shared action schema.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd

pytest.importorskip("pyspark")


def _ref_actions(actions):
    return [(a.step, a.column, a.count) for a in actions]


def test_spark_native_subset_matches_pandas(spark_session, small_df, native_config):
    sdf = spark_session.createDataFrame(small_df)
    out, rep = fd.clean(sdf, config=native_config, return_report=True)  # auto -> spark
    assert rep.backend == "spark"

    out_pd = out.toPandas() if not isinstance(out, pd.DataFrame) else out
    pandas_out, pandas_rep = fd.clean(small_df.copy(), config=native_config,
                                      engine="pandas", return_report=True)

    assert _ref_actions(rep.actions) == _ref_actions(pandas_rep.actions)
    assert set(out_pd.columns) == set(pandas_out.columns)
    assert out_pd.shape == pandas_out.shape


def test_spark_output_format(spark_session, small_df, native_config):
    from pyspark.sql import DataFrame as SparkDataFrame

    out = fd.clean(small_df.copy(), engine="spark", output_format="spark",
                   config=native_config)
    assert isinstance(out, SparkDataFrame)


def test_spark_auto_selection(spark_session, small_df, native_config):
    from freshdata.execution._config import EngineConfig, EngineSelector

    sdf = spark_session.createDataFrame(small_df)
    assert EngineSelector.select(sdf, EngineConfig(engine="auto")) == "spark"


def test_spark_impute_and_outliers(spark_session, numeric_df):
    cfg = fd.CleanConfig(strategy="conservative", fix_dtypes=False,
                         impute="median", outliers="flag", outlier_method="iqr")
    sdf = spark_session.createDataFrame(numeric_df)
    out, rep = fd.clean(sdf, config=cfg, return_report=True)
    out_pd = out.toPandas()

    assert rep.backend == "spark"
    assert "amount_outlier" in out_pd.columns
    assert int(out_pd["amount"].isna().sum()) == 0  # median filled
    steps = {d["step"] for d in rep.backend_differences}
    assert {"impute", "outliers"} <= steps


def test_spark_fallback_for_decision_engine(spark_session, small_df):
    # Default balanced strategy runs the decision engine -> pandas fallback.
    cfg = fd.CleanConfig(strategy="balanced")
    sdf = spark_session.createDataFrame(small_df)
    out, rep = fd.clean(sdf, config=cfg, return_report=True)
    assert rep.backend == "pandas"
    assert rep.fallback_events and rep.fallback_events[0]["backend"] == "spark"
    assert isinstance(out, pd.DataFrame)
