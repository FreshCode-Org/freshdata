"""Full-suite fixtures for CleanBench tracks T1, T4, and T5.

Same contract as the Phase-2/3 mini fixtures: deterministic, generated in
code, returning ``(truth, corrupted, kwargs)`` where kwargs go straight into
``freshdata.clean``. T4 additionally returns the paired-learning frames and
a drifted frame; T5 scales the base frame without inventing new labels.
"""

from __future__ import annotations

import random

import pandas as pd

from ..corruptors import (
    allowed_value_casing,
    whitespace_corruption,
)
from . import _base_frame

#: Vendor-style status mangling a learning profile can capture (T4): a
#: stable typo plus a stable casing form, exactly the transform families
#: ``fd.learn`` extracts into a value map.
STATUS_MESS = {"delivered": "Deliverd", "shipped": "SHIPPED", "returned": "Returnd"}

#: Synthetic sensitive literals planted for the privacy-leak count. These
#: exact strings must never appear in a profile saved with privacy="mask".
PLANTED_PII = ("leaky.email@example.com", "+919876500001")


def make_t1_representation_fixture(seed: int = 30):
    """T1: representation dirt whose correct repair is machine-checkable.

    ``truth`` here is the *expected cleaned output*: thousands separators
    parse back to the numeric value, sentinels become missing, whitespace
    is stripped. The default (model-free) cleaner must reproduce it.
    """
    rng = random.Random(f"t1:{seed}")
    base = _base_frame()
    corrupted = whitespace_corruption(base, ["email_addr", "status"], seed=seed, share=0.4)
    truth = base.copy(deep=True)
    # Thousands separators on revenue strings >= 1000: repaired by fix_dtypes.
    for idx in corrupted.index:
        value = str(corrupted.at[idx, "monthly_revenue"])
        if value.isdigit() and int(value) >= 1000 and rng.random() < 0.5:
            corrupted.at[idx, "monthly_revenue"] = f"{int(value):,}"
    # Sentinels: the only correct repair is "missing", so truth holds NaN.
    sentinel_rows = [i for i in corrupted.index if rng.random() < 0.15]
    for idx in sentinel_rows:
        corrupted.at[idx, "email_addr"] = rng.choice(("N/A", "null", "-"))
        truth.at[idx, "email_addr"] = None
    kwargs = {
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
    }
    return truth, corrupted, kwargs


def _t4_frames(n: int, *, id_prefix: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(messy, clean) order-style pair with learnable status/email/phone dirt."""
    rng = random.Random(f"t4:{seed}")
    statuses = ("delivered", "shipped", "returned")
    clean_rows = []
    messy_rows = []
    for i in range(n):
        status = statuses[i % len(statuses)]
        email = f"user{i}@example.com"
        ten_digit = f"98{seed % 7}65{43200 + i:05d}"
        clean_rows.append({
            "order_id": f"{id_prefix}{i + 1}",
            "email": email,
            "phone": f"+91{ten_digit}",
            "status": status,
        })
        local, _, domain = email.partition("@")
        messy_rows.append({
            "order_id": f"{id_prefix}{i + 1}",
            "email": f"{local}@@{domain}" if i % 2 == 0 else f" {local.title()}@{domain.upper()} ",
            "phone": ten_digit if i % 2 == 0 else f"0{ten_digit}",
            "status": STATUS_MESS[status] if rng.random() < 0.85 else status,
        })
    return pd.DataFrame(messy_rows), pd.DataFrame(clean_rows)


def make_t4_profile_fixture(seed: int = 40):
    """T4: learning/profile replay.

    Returns ``(pair_messy, pair_clean, batch_truth, batch_corrupted,
    drifted, kwargs, sensitive_messy, sensitive_clean)``:

    - the *pair* frames teach ``fd.learn`` stable status/email/phone maps;
    - the *batch* frames are a fresh batch with the same dirt, where replaying
      the profile should lift repair F1;
    - *drifted* renames/drops enough columns that replay must be blocked;
    - the *sensitive* pair plants PII literals for the privacy-leak count.
    """
    pair_messy, pair_clean = _t4_frames(24, id_prefix="A", seed=seed)
    batch_corrupted, batch_truth = _t4_frames(12, id_prefix="B", seed=seed + 1)

    drifted = batch_corrupted.rename(columns={"status": "status_v2", "email": "contact"})
    drifted = drifted.drop(columns=["phone"])

    sensitive_messy, sensitive_clean = _t4_frames(24, id_prefix="S", seed=seed + 2)
    sensitive_clean.loc[0, "email"] = PLANTED_PII[0]
    sensitive_messy.loc[0, "email"] = f" {PLANTED_PII[0].title()} "
    sensitive_clean.loc[1, "phone"] = PLANTED_PII[1]
    sensitive_messy.loc[1, "phone"] = PLANTED_PII[1][3:]

    kwargs = {
        "verbose": False,
        "drop_duplicates": False,
        "reset_index": False,
        "semantic_mode": "auto",
    }
    return (pair_messy, pair_clean, batch_truth, batch_corrupted, drifted, kwargs,
            sensitive_messy, sensitive_clean)


def make_t5_scale_fixture(seed: int = 50, target_rows: int = 50_000):
    """T5: the T2-style frame tiled to ``target_rows`` (ids kept unique)."""
    truth, corrupted, kwargs = _scaled_pair(seed=seed, target_rows=target_rows)
    return truth, corrupted, kwargs


def _scaled_pair(*, seed: int, target_rows: int):
    base_truth = _base_frame()
    base_corrupted = allowed_value_casing(base_truth, ["status"], seed=seed, share=0.4)
    base_corrupted = whitespace_corruption(
        base_corrupted, ["email_addr"], seed=seed + 1, share=0.3)
    tiles = -(-target_rows // len(base_truth))

    def tile(df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for k in range(tiles):
            part = df.copy(deep=True)
            part["cust_id"] = [f"{v}-{k}" for v in part["cust_id"]]
            parts.append(part)
        return pd.concat(parts, ignore_index=True).head(target_rows)

    kwargs = {"verbose": False, "drop_duplicates": False, "reset_index": False}
    return tile(base_truth), tile(base_corrupted), kwargs
