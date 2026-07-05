"""Cached dataset access for benchmarks.

Usage::

    from benchmarks.datasets import get_dataset
    df = get_dataset(100_000)
"""

from __future__ import annotations
import os
import pandas as pd
from benchmarks.datasets.data_generator import generate_dataset

def get_dataset(n_rows: int) -> "pd.DataFrame":
    """Return a cached messy dataset with *n_rows* rows.

    The dataset is generated once per ``n_rows`` value and cached in
    memory and on disk to survive subprocess creation in ASV.
    """
    cache_path = f"/tmp/freshdata_bench_cache_{n_rows}.pkl"
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    
    df = generate_dataset(n_rows)
    df.to_pickle(cache_path)
    return df


__all__ = ["get_dataset", "generate_dataset"]
