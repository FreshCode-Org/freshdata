from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WIDTHS = {"narrow": 8, "medium": 32, "wide": 128}
DATASET_TYPES = (
    "mixed",
    "numeric",
    "categorical",
    "string",
    "nullable",
    "datetime",
    "high_cardinality",
)


@dataclass(frozen=True)
class DatasetSpec:
    rows: int
    width: str
    seed: int = 42
    dataset_type: str = "mixed"

    def __post_init__(self) -> None:
        if self.rows < 1:
            raise ValueError("rows must be >= 1")
        if self.width not in WIDTHS:
            raise ValueError(f"width must be one of {sorted(WIDTHS)}")
        if self.dataset_type not in DATASET_TYPES:
            raise ValueError(f"dataset_type must be one of {DATASET_TYPES}")


def _nullable_int(rng: np.random.Generator, rows: int) -> pd.Series:
    values = pd.array(rng.integers(0, 10_000, rows), dtype="Int64")
    values[rng.random(rows) < 0.10] = pd.NA
    return pd.Series(values)


def make_mixed_frame(spec: DatasetSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    rows = spec.rows
    numeric = rng.normal(100.0, 20.0, rows)
    numeric[rng.random(rows) < 0.10] = np.nan
    numeric[rng.choice(rows, max(1, rows // 100), replace=False)] *= 20.0
    categories = pd.Categorical(
        rng.choice(["alpha", "beta", "gamma", None], rows, p=[0.35, 0.3, 0.25, 0.1])
    )
    frame = pd.DataFrame(
        {
            "record_id": np.arange(rows, dtype=np.int64),
            "target": pd.array(rng.choice([0, 1, None], rows, p=[0.47, 0.48, 0.05]), dtype="Int8"),
            "numeric_0": numeric,
            "nullable_int_0": _nullable_int(rng, rows),
            "category_0": categories,
            "text_0": pd.array(
                [f"free form note {i % 97}" if i % 11 else None for i in range(rows)],
                dtype="string",
            ),
            "event_time_0": pd.date_range("2024-01-01", periods=rows, freq="min", tz="UTC"),
            "high_cardinality_0": pd.array(
                [f"key-{spec.seed}-{i}" for i in range(rows)], dtype="string"
            ),
        }
    )
    factories = (
        lambda i: pd.Series(rng.normal(i, 1.0, rows), name=f"numeric_{i}"),
        lambda i: _nullable_int(rng, rows).rename(f"nullable_int_{i}"),
        lambda i: pd.Series(
            pd.Categorical(rng.choice(["a", "b", "c", None], rows)), name=f"category_{i}"
        ),
        lambda i: pd.Series(
            pd.array([f"value {i}-{j % 211}" for j in range(rows)], dtype="string"),
            name=f"text_{i}",
        ),
        lambda i: pd.Series(
            pd.date_range("2020-01-01", periods=rows, freq="h"), name=f"datetime_{i}"
        ),
        lambda i: pd.Series(
            pd.array([f"hc-{i}-{j}" for j in range(rows)], dtype="string"),
            name=f"high_cardinality_{i}",
        ),
    )
    family_index = {
        "numeric": 0,
        "nullable": 1,
        "categorical": 2,
        "string": 3,
        "datetime": 4,
        "high_cardinality": 5,
    }.get(spec.dataset_type)
    if family_index is not None:
        frame = frame[["record_id", "target"]].copy()
    i = 1
    while frame.shape[1] < WIDTHS[spec.width]:
        index = family_index if family_index is not None else (i - 1) % len(factories)
        series = factories[index](i)
        frame[series.name] = series
        i += 1
    if rows >= 100:
        duplicate_count = max(1, rows // 100)
        frame = pd.concat(
            [frame.iloc[:-duplicate_count], frame.iloc[:duplicate_count].copy()],
            ignore_index=True,
        )
    return frame
