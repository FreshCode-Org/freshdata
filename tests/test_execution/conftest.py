"""Shared fixtures for the execution-engine tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freshdata.config import CleanConfig


@pytest.fixture
def native_config() -> CleanConfig:
    """A config fully inside the native (out-of-core) subset.

    ``strategy="conservative"`` disables the decision engine and
    ``fix_dtypes=False`` skips the sampled dtype heuristics, so all three engines
    execute natively and must agree byte-for-byte.
    """
    return CleanConfig(strategy="conservative", fix_dtypes=False, verbose=False)


@pytest.fixture
def small_df() -> pd.DataFrame:
    """20 rows exercising rename, whitespace, sentinels, empties, and dedup."""
    return pd.DataFrame(
        {
            "patient_id": ["P001", "P002", None, "P004"] * 5,
            "age": [34.0, np.nan, 52.0, 89.0] * 5,
            "revenue": [1200.0, np.nan, 3400.0, np.nan] * 5,
            "category": ["A", None, "B", "A"] * 5,
            "diagnosis": ["X10", "Y20", None, "Z30"] * 5,
            " Name ": ["alice", " bob", "carol ", None] * 5,
            "n/a_col": ["N/A", None, "-", "#REF!"] * 5,
            "empty_col": [None] * 20,
        }
    )


@pytest.fixture
def parquet_10k(tmp_path) -> str:
    from freshdata.benchmarks._data_gen import generate_parquet

    path = str(tmp_path / "bench_10k.parquet")
    generate_parquet(10_000, path, batch_size=5_000)
    return path


@pytest.fixture
def impute_outlier_config() -> CleanConfig:
    """Native subset plus the opt-in median impute + IQR clip overrides."""
    return CleanConfig(
        strategy="conservative", fix_dtypes=False, verbose=False,
        impute="median", outliers="clip", outlier_method="iqr",
    )


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    """Numeric/string frame with unambiguous missing values and outliers.

    The outliers (10_000) sit far outside the bulk so every backend's quantile
    interpolation agrees on the flagged/clipped count.
    """
    return pd.DataFrame(
        {
            "amount": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0, 7.0, 8.0, 9.0, 10_000.0],
            "qty": [10, 12, 11, 13, 12, 11, 10, 14, 12, 9_999],
            "label": ["a", "b", "a", None, "a", "b", "a", "b", "a", "c"],
        }
    )


@pytest.fixture(scope="session")
def spark_session():
    """A local SparkSession, or skip the test when no JVM/Spark is available."""
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        session = (
            SparkSession.builder.master("local[1]")
            .appName("freshdata-tests")
            .config("spark.sql.shuffle.partitions", "1")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")
    except Exception as exc:  # pragma: no cover - environment without a JVM
        pytest.skip(f"SparkSession unavailable (no JVM?): {exc}")
    yield session
    session.stop()
