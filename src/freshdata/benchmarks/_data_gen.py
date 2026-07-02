"""Synthetic Parquet generator for benchmarks.

Writes a realistic "dirty" tabular dataset (numeric columns with nulls and
outliers, categorical/text/email string columns with injected sentinels and
stray whitespace, an id and a target) at an arbitrary row count, in 100k-row
batches so the full frame never lives in memory at once.
"""

from __future__ import annotations

import os

_SENTINELS = ["N/A", "-", "", "null", "#REF!", "missing"]
_CATEGORIES = [f"cat_{i:02d}" for i in range(20)]


def _make_batch(
    n: int,
    rng,
    n_numeric: int,
    n_categorical: int,
    n_pii: int,
    n_text: int,
    null_rate: float,
    sentinel_rate: float,
    id_offset: int,
):
    """Build one :class:`pyarrow.RecordBatch` of *n* rows."""
    import numpy as np
    import pyarrow as pa

    arrays: list = []
    names: list[str] = []

    ids = np.arange(id_offset, id_offset + n, dtype=np.int64)
    arrays.append(pa.array(ids))
    names.append("id")

    for i in range(n_numeric):
        base = rng.normal(100.0, 25.0, n)
        # ~1% extreme outliers
        outlier_mask = rng.random(n) < 0.01
        base[outlier_mask] *= rng.uniform(8, 15, outlier_mask.sum())
        vals = base.astype(object)
        vals[rng.random(n) < null_rate] = None
        arrays.append(pa.array(vals, type=pa.float64()))
        names.append(f"num_{i}")

    def _inject_strings(values: list[str]) -> list:
        out: list = []
        for v in values:
            r = rng.random()
            if r < null_rate:
                out.append(None)
            elif r < null_rate + sentinel_rate:
                out.append(_SENTINELS[rng.integers(0, len(_SENTINELS))])
            elif r < null_rate + 2 * sentinel_rate:
                out.append(f"  {v} ")  # stray whitespace
            else:
                out.append(v)
        return out

    for i in range(n_categorical):
        picks = [_CATEGORIES[j] for j in rng.integers(0, len(_CATEGORIES), n)]
        arrays.append(pa.array(_inject_strings(picks), type=pa.string()))
        names.append(f"cat_{i}")

    for i in range(n_pii):
        emails = [f"user{int(v)}@example.com" for v in rng.integers(0, 10_000_000, n)]
        arrays.append(pa.array(_inject_strings(emails), type=pa.string()))
        names.append(f"email_{i}")

    for i in range(n_text):
        words = [
            " ".join(_CATEGORIES[j] for j in rng.integers(0, len(_CATEGORIES), 3))
            for _ in range(n)
        ]
        arrays.append(pa.array(_inject_strings(words), type=pa.string()))
        names.append(f"text_{i}")

    arrays.append(pa.array(rng.integers(0, 2, n), type=pa.int8()))
    names.append("target")

    return pa.RecordBatch.from_arrays(arrays, names=names)


def generate_parquet(
    n_rows: int,
    output_path: str,
    *,
    n_numeric_cols: int = 10,
    n_categorical_cols: int = 5,
    n_pii_cols: int = 2,
    n_text_cols: int = 2,
    null_rate: float = 0.15,
    sentinel_rate: float = 0.05,
    seed: int = 42,
    batch_size: int = 100_000,
) -> str:
    """Generate an *n_rows* synthetic dataset, written to *output_path* as Parquet.

    Returns *output_path*. Never holds more than ``batch_size`` rows in memory.
    """
    import numpy as np
    import pyarrow.parquet as pq

    rng = np.random.default_rng(seed)
    writer = None
    written = 0
    try:
        while written < n_rows:
            cur = min(batch_size, n_rows - written)
            batch = _make_batch(
                cur, rng, n_numeric_cols, n_categorical_cols, n_pii_cols,
                n_text_cols, null_rate, sentinel_rate, id_offset=written,
            )
            if writer is None:
                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                writer = pq.ParquetWriter(output_path, batch.schema, compression="snappy")
            writer.write_batch(batch)
            written += cur
    finally:
        if writer is not None:
            writer.close()
    return output_path
