"""Scale helpers for the T5 track: grow a labeled fixture without new labels.

Tiling preserves per-cell ground truth exactly (row ``i`` of tile ``k`` maps
back to source row ``i``), so T5 measures throughput and memory on a frame
whose correctness metrics are still fully machine-verifiable.
"""

from __future__ import annotations

import random

import pandas as pd


def scale_frame(df: pd.DataFrame, *, target_rows: int, seed: int = 0) -> pd.DataFrame:
    """Tile *df* to ``target_rows``, remapping obvious ID columns to stay unique."""
    if df.empty or target_rows <= len(df):
        return df.copy(deep=True)
    rng = random.Random(f"freshdata-scale:{seed}")
    tiles = -(-target_rows // len(df))  # ceil
    parts = []
    id_columns = [
        c for c in df.columns if str(c).lower().endswith("_id") or str(c).lower() == "id"
    ]
    for k in range(tiles):
        part = df.copy(deep=True)
        for col in id_columns:
            part[col] = [f"{v}-{k}" for v in part[col]]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).head(target_rows)
    # Deterministic row shuffle so batch effects do not line up with tiles.
    order = list(range(len(out)))
    rng.shuffle(order)
    return out.iloc[order].reset_index(drop=True)
