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
    contexts = build_contexts(df, config)
    corr = None
    if config.engine_mode is not None and len(df) >= 30:
        corr = numeric_corr_matrix(df)
    return EngineCache(contexts=contexts, numeric_corr=corr)
