"""Cached dataset access for benchmarks.

Usage::

    from benchmarks.datasets import get_dataset
    df = get_dataset(100_000)
"""

from __future__ import annotations

import functools

from .data_generator import generate_dataset


@functools.lru_cache(maxsize=8)
def get_dataset(n_rows: int) -> "pd.DataFrame":
    """Return a cached messy dataset with *n_rows* rows.

    The dataset is generated once per ``n_rows`` value and cached in
    memory for the lifetime of the process. Benchmark ``setup()``
    methods should ``.copy()`` the returned frame before mutation.
    """
    return generate_dataset(n_rows)


__all__ = ["get_dataset", "generate_dataset"]
