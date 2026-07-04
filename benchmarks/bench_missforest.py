#!/usr/bin/env python
"""Compare simple, KNN, and MissForest-style imputation on mixed synthetic data.

Run after installing the ML extra:

    pip install -e ".[ml]"
    python benchmarks/bench_missforest.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

import freshdata as fd


@dataclass(frozen=True)
class Result:
    method: str
    rows: int
    seconds: float
    numeric_mae: float
    categorical_accuracy: float


def make_frame(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(42, 9, rows)
    visits = rng.poisson(4, rows)
    income = 900 + age * 1100 + visits * 700 + rng.normal(0, 1200, rows)
    segment = np.where(income > np.median(income), "enterprise", "starter")
    return pd.DataFrame(
        {
            "customer_id": np.arange(rows),
            "age": age.round(2),
            "visits": visits.astype(float),
            "income": income.round(2),
            "segment": segment.astype(object),
            "churn": rng.integers(0, 2, rows),
        }
    )


def inject_missing(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    dirty = df.copy(deep=True)
    masks = {
        "income": rng.choice(len(df), size=max(1, len(df) // 5), replace=False),
        "segment": rng.choice(len(df), size=max(1, len(df) // 6), replace=False),
    }
    dirty.loc[masks["income"], "income"] = np.nan
    dirty.loc[masks["segment"], "segment"] = None
    return dirty, masks


def run_method(
    method: str,
    dirty: pd.DataFrame,
    gold: pd.DataFrame,
    masks: dict[str, np.ndarray],
) -> Result:
    kwargs: dict[str, object] = {
        "drop_duplicates": False,
        "drop_empty_rows": False,
        "target_column": "churn",
        "id_columns": ("customer_id",),
        "verbose": False,
    }
    if method == "median/mode":
        kwargs["impute"] = "auto"
    elif method == "knn":
        kwargs["strategy"] = "aggressive"
    elif method == "missforest":
        kwargs["impute_method"] = "missforest"
    else:
        raise ValueError(method)

    started = time.perf_counter()
    cleaned = fd.clean(dirty, **kwargs)
    elapsed = time.perf_counter() - started
    income_idx = masks["income"]
    segment_idx = masks["segment"]
    numeric_mae = float(
        np.mean(
            np.abs(cleaned.loc[income_idx, "income"] - gold.loc[income_idx, "income"])
        )
    )
    categorical_accuracy = float(
        (
            cleaned.loc[segment_idx, "segment"].to_numpy()
            == gold.loc[segment_idx, "segment"].to_numpy()
        ).mean()
    )
    return Result(method, len(dirty), elapsed, numeric_mae, categorical_accuracy)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="*", type=int, default=[500, 2_000])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows: list[Result] = []
    for size in args.sizes:
        gold = make_frame(size, args.seed)
        dirty, masks = inject_missing(gold, args.seed + 1)
        for method in ("median/mode", "knn", "missforest"):
            rows.append(run_method(method, dirty, gold, masks))

    print(f"{'method':14s} {'rows':>7s} {'sec':>8s} {'income_mae':>12s} {'segment_acc':>12s}")
    for result in rows:
        print(
            f"{result.method:14s} {result.rows:7d} {result.seconds:8.3f} "
            f"{result.numeric_mae:12.2f} {result.categorical_accuracy:12.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
