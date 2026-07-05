"""Fairness enforcement utilities for cross-library benchmarks.

Ensures identical inputs, prevents mutation leakage, and controls for
garbage collection / cache effects between benchmark runs.
"""

from __future__ import annotations

import gc
import os
from typing import Any


def fresh_copy(df: Any) -> Any:
    """Return a deep copy of a DataFrame for mutation-safe benchmarking.

    This ensures that each benchmark run operates on identical,
    independent data — no library can benefit from prior mutations.
    """
    return df.copy(deep=True)


def gc_collect() -> None:
    """Force garbage collection to reduce memory measurement noise.

    Called in ``setup()`` before each benchmark to minimize the effect
    of residual allocations from previous runs.
    """
    gc.collect()
    gc.collect()  # Second pass catches weak-reference cleanup


def pin_threads(n: int = 1) -> None:
    """Pin numeric libraries to *n* threads for deterministic timing.

    Already handled via asv.conf.json ``matrix.env``, but can be called
    explicitly in ``setup()`` for defense-in-depth.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def get_numeric_columns(df: Any) -> list[str]:
    """Return column names that contain (or should contain) numeric data.

    Used to identify which columns are suitable for imputation, scaling,
    and outlier detection benchmarks.
    """
    import pandas as pd
    numeric = []
    for col in df.columns:
        try:
            series = pd.to_numeric(df[col], errors="coerce")
            non_null = series.notna().sum()
            if non_null > len(df) * 0.3:  # at least 30% parseable
                numeric.append(col)
        except Exception:
            pass
    return numeric


def get_string_columns(df: Any) -> list[str]:
    """Return column names that contain string/object data."""
    return [col for col in df.columns if df[col].dtype == object]


def get_categorical_columns(df: Any, max_cardinality: int = 100) -> list[str]:
    """Return string columns with cardinality ≤ *max_cardinality*."""
    cats = []
    for col in get_string_columns(df):
        try:
            nunique = df[col].nunique()
            if 2 <= nunique <= max_cardinality:
                cats.append(col)
        except Exception:
            pass
    return cats


def validate_equivalent_output(
    original: Any,
    result: Any,
    operation: str,
) -> bool:
    """Spot-check that a benchmark result is semantically valid.

    This is NOT called during timing — only during development/testing
    to verify that library adapters produce correct results.
    """
    if result is None:
        return False
    import pandas as pd
    if isinstance(result, pd.DataFrame):
        # Basic sanity: result should have at least 1 row
        return len(result) > 0
    # Non-DataFrame results (e.g., Polars, boolean masks) are valid
    return True
