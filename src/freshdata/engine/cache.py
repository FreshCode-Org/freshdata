"""Shared engine artifacts computed once per clean pass."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import CleanConfig
from .context import build_contexts, numeric_corr_matrix


@dataclass
class EngineCache:
    """Column contexts plus numeric correlation matrix reused across engine steps."""

    contexts: dict
    numeric_corr: pd.DataFrame | None = None


def build_engine_cache(df: pd.DataFrame, config: CleanConfig) -> EngineCache:
    """Profile columns and precompute numeric correlations when useful."""
    # The engine steps only ever read a context for a column that has missing
    # cells (auto_missing) or is numeric (auto_outliers). Both gates are
    # stable between here and the step runs — nothing in between adds missing
    # values or changes a dtype — so profiling just that superset is
    # observationally identical to profiling every column, and skips the
    # expensive per-column profile for pristine text/id columns.
    needed = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].isna().any()
    ]
    contexts = build_contexts(df, config, columns=needed)
    corr = None
    if config.engine_mode is not None and len(df) >= 30:
        corr = numeric_corr_matrix(df)
    return EngineCache(contexts=contexts, numeric_corr=corr)
