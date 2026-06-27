"""Lazy optional-dependency import guards for the execution backends.

The out-of-core backends depend on optional packages (``polars``, ``duckdb``,
``pyarrow``). Importing :mod:`freshdata` must never require them, so every
backend resolves its dependency through these helpers at call time and raises a
clear, install-pointing error if it is missing.
"""

from __future__ import annotations

from typing import Any


def require_polars() -> Any:
    """Return the imported :mod:`polars` module or raise a helpful error."""
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised via message
        raise ImportError(
            "The Polars engine requires polars. "
            "Install it with: pip install 'freshdata-cleaner[polars]'"
        ) from exc
    return pl


def require_duckdb() -> Any:
    """Return the imported :mod:`duckdb` module or raise a helpful error."""
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised via message
        raise ImportError(
            "The DuckDB engine requires duckdb. "
            "Install it with: pip install 'freshdata-cleaner[duckdb]'"
        ) from exc
    return duckdb


def require_pyarrow() -> Any:
    """Return the imported :mod:`pyarrow` module or raise a helpful error."""
    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via message
        raise ImportError(
            "Reading Parquet metadata requires pyarrow. "
            "Install it with: pip install 'freshdata-cleaner[pyarrow]'"
        ) from exc
    return pyarrow


def require_pyspark() -> Any:
    """Return the imported :mod:`pyspark.sql` module or raise a helpful error."""
    try:
        from pyspark import sql as pyspark_sql
    except ImportError as exc:  # pragma: no cover - exercised via message
        raise ImportError(
            "The Spark engine requires pyspark. "
            "Install it with: pip install 'freshdata-cleaner[spark]'"
        ) from exc
    return pyspark_sql


def has_pyspark() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def has_polars() -> bool:
    try:
        import polars  # noqa: F401
    except ImportError:
        return False
    return True


def has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
    except ImportError:
        return False
    return True
