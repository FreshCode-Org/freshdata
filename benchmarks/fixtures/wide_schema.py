"""Fixture 4 — Wide-schema synthetic (dynamic column count).

Stress-tests report generation and column-inference caching. The column count
is a parameter (``n_cols``); families are mixed deterministically: numeric with
banded missingness, categoricals of varying cardinality, datetime-ish strings,
high-cardinality free text (role=text, never force-filled), boolean-ish tokens,
sentinel-heavy columns, plus exactly one UUID id column and one float target
column (never imputed).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import (
    Defect,
    GoldLabel,
    ROLE_BOOL,
    ROLE_CATEGORICAL,
    ROLE_DATETIME,
    ROLE_ID,
    ROLE_NUMERIC,
    ROLE_TARGET,
    ROLE_TEXT,
    defect_mask,
    gold_to_records,
    manifest_to_records,
    pick,
    resolve_rate,
    rng_for,
    uuid_series,
)

_MISSING_BANDS = (0.01, 0.10, 0.30, 0.60)
_CARDINALITIES = (2, 10, 100, 1000)
_SENTINELS = ("N/A", "--", "null")
_BOOL_TOKENS = ("yes", "no", "true", "false", "1", "0")
_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")

ID_COL = "row_uuid"
TARGET_COL = "y_target"


def _column_plan(n_cols: int) -> list[tuple[str, str]]:
    """Deterministic (name, family) plan for ``n_cols`` columns.

    Reserves one id and one target column; the rest follow the family mix
    described in the module docstring.
    """
    plan: list[tuple[str, str]] = [(ID_COL, "id"), (TARGET_COL, "target")]
    body = n_cols - 2
    # family -> share
    shares = [
        ("numeric", 0.20),
        ("categorical", 0.20),
        ("datetime", 0.20),
        ("text", 0.20),
        ("bool", 0.10),
        ("sentinel", 0.10),
    ]
    counts = {fam: int(round(share * body)) for fam, share in shares}
    # absorb rounding drift into numeric
    counts["numeric"] += body - sum(counts.values())
    k = 0
    for fam, _ in shares:
        for _ in range(counts[fam]):
            plan.append((f"{fam}_{k:04d}", fam))
            k += 1
    return plan[:n_cols]


def generate(
    n_rows: int,
    seed: int = 42,
    defect_rate: float | None = None,
    n_cols: int = 100,
) -> pd.DataFrame:
    rng = rng_for(seed)
    n = int(n_rows)
    plan = _column_plan(int(n_cols))
    data: dict[str, Any] = {}
    base = np.datetime64("2020-01-01")

    for j, (name, fam) in enumerate(plan):
        if fam == "id":
            data[name] = uuid_series(rng, n, prefix="row-")
        elif fam == "target":
            data[name] = np.round(rng.normal(0, 1, size=n), 5)
        elif fam == "numeric":
            band = _MISSING_BANDS[j % len(_MISSING_BANDS)]
            col = rng.lognormal(3, 1, size=n)
            m = defect_mask(rng, n, resolve_rate(band, defect_rate))
            col = col.astype(object)
            col[m] = np.nan
            data[name] = col
        elif fam == "categorical":
            card = _CARDINALITIES[j % len(_CARDINALITIES)]
            pool = tuple(f"c{j}_{i}" for i in range(card))
            data[name] = pick(rng, pool, n)
        elif fam == "datetime":
            d = base + rng.integers(0, 1500, size=n).astype("timedelta64[D]")
            if j % 2:
                data[name] = np.array(
                    [pd.Timestamp(x).strftime("%m/%d/%Y") for x in d], dtype=object
                )
            else:
                data[name] = np.array([str(x) for x in d], dtype=object)
        elif fam == "text":
            data[name] = np.array(
                [" ".join(rng.choice(_WORDS, size=4)) + f" {rng.integers(0, 10**6)}" for _ in range(n)],
                dtype=object,
            )
        elif fam == "bool":
            data[name] = pick(rng, _BOOL_TOKENS, n)
        elif fam == "sentinel":
            col = rng.lognormal(2, 1, size=n).round(3).astype(object)
            m = defect_mask(rng, n, resolve_rate(0.15, defect_rate))
            col[m] = pick(rng, _SENTINELS, int(m.sum()))
            data[name] = col

    return pd.DataFrame(data)


def gold_labels(n_cols: int = 100) -> dict[str, dict[str, Any]]:
    """GOLD_LABELS depend on the chosen column count, so they are a function."""
    labels: dict[str, GoldLabel] = {}
    for name, fam in _column_plan(int(n_cols)):
        if fam == "id":
            labels[name] = GoldLabel(ROLE_ID, "object", "preserve", True)
        elif fam == "target":
            labels[name] = GoldLabel(ROLE_TARGET, "float64", "preserve", True)
        elif fam == "numeric":
            labels[name] = GoldLabel(ROLE_NUMERIC, "float64", "impute_or_preserve", False)
        elif fam == "categorical":
            labels[name] = GoldLabel(ROLE_CATEGORICAL, "object", "preserve", False)
        elif fam == "datetime":
            labels[name] = GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False)
        elif fam == "text":
            labels[name] = GoldLabel(ROLE_TEXT, "object", "preserve", True)
        elif fam == "bool":
            labels[name] = GoldLabel(ROLE_BOOL, "object", "coerce_or_preserve", False)
        elif fam == "sentinel":
            labels[name] = GoldLabel(ROLE_NUMERIC, "float64", "sentinel_normalize", False)
    return gold_to_records(labels)


# Default labels for the 100-column variant; harness recomputes for others.
GOLD_LABELS: dict[str, dict[str, Any]] = gold_labels(100)

DEFECT_MANIFEST: list[dict[str, Any]] = manifest_to_records(
    [
        Defect("wide-missing-bands", "numeric_*", "banded_missingness", 0.25, "impute_or_preserve"),
        Defect("wide-sentinel", "sentinel_*", "sentinel", 0.15, "sentinel_normalize"),
        Defect("wide-datetime-fmt", "datetime_*", "mixed_date_format", 1.0, "dtype_coerce"),
        Defect("wide-bool-mixed", "bool_*", "mixed_bool_tokens", 1.0, "coerce_or_preserve"),
    ]
)

SCALE_VARIANTS = (1_000, 10_000, 100_000)
COL_VARIANTS = (100, 500, 1_000, 5_000)
ID_COLUMNS = (ID_COL,)
TARGET_COLUMN = TARGET_COL
TEXT_COLUMNS: tuple[str, ...] = ()  # discovered by name prefix "text_"
N_COLS = 100
