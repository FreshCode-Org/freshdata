"""Fixture 3 — Event-log / CDC stream (25 columns).

A change-data-capture event stream with CDC-shaped defects: late arrivals
(event_time far past server_time), replay risk (duplicate entity_key at the
same event_time), out-of-order payload versions within an entity, operations
outside the approved set, missing event_time (flag only, never impute), and
sentinel tokens in payload columns. All defects are representation- or
profile-level — within FreshData's repair and flagging scope.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import (
    BAD_OPERATION,
    OPERATION_REF,
    Defect,
    GoldLabel,
    ROLE_CATEGORICAL,
    ROLE_DATETIME,
    ROLE_ID,
    ROLE_NUMERIC,
    ROLE_TEXT,
    defect_mask,
    gold_to_records,
    manifest_to_records,
    pick,
    resolve_rate,
    rng_for,
    uuid_series,
)

N_EXTRA = 19  # 6 named + 19 payload = 25 columns
_PAYLOAD_SENTINELS = ("NULL", "N/A")


def generate(n_rows: int, seed: int = 42, defect_rate: float | None = None) -> pd.DataFrame:
    rng = rng_for(seed)
    n = int(n_rows)
    data: dict[str, Any] = {}

    data["event_id"] = uuid_series(rng, n, prefix="evt-")
    # a bounded set of entity keys so duplicates / ordering is meaningful
    n_entities = max(1, n // 5)
    ent = np.array([f"ent-{i % n_entities:07d}" for i in range(n)], dtype=object)
    rng.shuffle(ent)
    data["entity_key"] = ent

    base = np.datetime64("2024-01-01T00:00:00")
    secs = rng.integers(0, 60 * 60 * 24 * 30, size=n)
    et = base + secs.astype("timedelta64[s]")
    data["event_time"] = np.array([str(x) for x in et], dtype=object)
    # server_time normally a few seconds after event_time
    st = et + rng.integers(1, 30, size=n).astype("timedelta64[s]")
    data["server_time"] = np.array([str(x) for x in st], dtype=object)

    data["operation"] = pick(rng, OPERATION_REF, n)
    data["payload_version"] = rng.integers(1, 20, size=n).astype("int64")

    for i in range(N_EXTRA):
        if i % 2 == 0:
            data[f"payload_num_{i:02d}"] = np.round(rng.normal(0, 1, size=n), 4)
        else:
            data[f"payload_str_{i:02d}"] = pick(rng, ("a", "b", "c", "d", "e"), n)

    df = pd.DataFrame(data)
    return _inject(df, rng, defect_rate)


def _inject(df: pd.DataFrame, rng: np.random.Generator, defect_rate: float | None) -> pd.DataFrame:
    n = len(df)

    def rate(base: float) -> float:
        return resolve_rate(base, defect_rate)

    # 5% late arrivals: event_time > server_time + 48h
    m = defect_mask(rng, n, rate(0.05))
    for i in np.where(m)[0]:
        st = pd.Timestamp(df.at[i, "server_time"])
        df.at[i, "event_time"] = str((st + pd.Timedelta(hours=72)).to_datetime64())

    # 3% replay: duplicate entity_key at same event_time (copy a sibling's key+time)
    m = defect_mask(rng, n, rate(0.03))
    idxs = np.where(m)[0]
    for i in idxs:
        j = int(rng.integers(0, n))
        df.at[i, "entity_key"] = df.at[j, "entity_key"]
        df.at[i, "event_time"] = df.at[j, "event_time"]

    # 4% out-of-order payload_version (set to a low version)
    m = defect_mask(rng, n, rate(0.04))
    df.loc[m, "payload_version"] = 0

    # 2% operation outside approved set
    m = defect_mask(rng, n, rate(0.02))
    df.loc[m, "operation"] = pick(rng, BAD_OPERATION, int(m.sum()))

    # 1% missing event_time (flag, never impute)
    m = defect_mask(rng, n, rate(0.01))
    df.loc[m, "event_time"] = None

    # 2% sentinels in payload string columns
    pcols = [c for c in df.columns if c.startswith("payload_str_")]
    for c in pcols:
        m = defect_mask(rng, n, rate(0.02))
        vals = df[c].astype(object).to_numpy(copy=True)
        vals[m] = pick(rng, _PAYLOAD_SENTINELS, int(m.sum()))
        df[c] = vals

    return df.reset_index(drop=True)


GOLD_LABELS: dict[str, dict[str, Any]] = gold_to_records(
    {
        "event_id": GoldLabel(ROLE_ID, "object", "preserve", True),
        "entity_key": GoldLabel(ROLE_ID, "object", "preserve", True),
        "event_time": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "flag_missing", False),
        "server_time": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False),
        "operation": GoldLabel(ROLE_CATEGORICAL, "object", "reference_flag", False, OPERATION_REF),
        "payload_version": GoldLabel(ROLE_NUMERIC, "int64", "preserve", False),
    }
)

DEFECT_MANIFEST: list[dict[str, Any]] = manifest_to_records(
    [
        Defect("evt-late", "event_time", "late_arrival", 0.05, "cdc_flag"),
        Defect("evt-replay", "entity_key", "replay_duplicate", 0.03, "cdc_flag"),
        Defect("evt-ooo", "payload_version", "out_of_order_version", 0.04, "cdc_flag"),
        Defect("evt-op-ref", "operation", "reference_violation", 0.02, "reference_flag"),
        Defect("evt-et-missing", "event_time", "missing_event_time", 0.01, "flag_missing", preservation=True),
        Defect("evt-payload-sentinel", "payload_str_*", "sentinel", 0.02, "sentinel_normalize"),
    ]
)

SCALE_VARIANTS = (10_000, 1_000_000, 10_000_000, 50_000_000)
ID_COLUMNS = ("event_id", "entity_key")
TARGET_COLUMN: str | None = None
TEXT_COLUMNS: tuple[str, ...] = ()
N_COLS = 25
