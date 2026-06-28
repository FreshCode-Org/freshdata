"""Fixture 2 — Finance ledger (60 columns).

A general-ledger export with the messy-money problems FreshData is built to
repair: currency-symbol and thousands-separated amount strings, accounting
negatives in parentheses, mixed debit/credit encodings, reference-set currency
violations, duplicate transaction rows, and dates in ambiguous locale formats.
``value_date`` missingness must be flagged, not blindly filled.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import (
    BAD_CURRENCY,
    CURRENCY_REF,
    DEBIT_CREDIT_REF,
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

N_EXTRA = 50  # 10 named + 50 generated = 60 columns
ENTITY_REF = ("ENT-A", "ENT-B", "ENT-C", "ENT-D")


def generate(n_rows: int, seed: int = 42, defect_rate: float | None = None) -> pd.DataFrame:
    rng = rng_for(seed)
    n = int(n_rows)
    data: dict[str, Any] = {}

    data["transaction_id"] = uuid_series(rng, n, prefix="txn-")
    data["account_code"] = np.array(
        [f"AC{rng.integers(10000, 99999)}" for _ in range(n)], dtype=object
    )
    base = np.datetime64("2022-01-01")
    post = base + rng.integers(0, 700, size=n).astype("timedelta64[D]")
    data["posting_date"] = np.array([str(d) for d in post], dtype=object)
    val = post + rng.integers(0, 5, size=n).astype("timedelta64[D]")
    data["value_date"] = np.array([str(d) for d in val], dtype=object)
    # clean amounts as plain numeric strings; defects overwrite a subset
    amt = np.round(rng.lognormal(7.0, 1.3, size=n), 2)
    data["amount"] = np.array([f"{a:.2f}" for a in amt], dtype=object)
    data["currency"] = pick(rng, CURRENCY_REF, n)
    data["debit_credit"] = pick(rng, ("D", "C"), n)
    data["gl_account"] = np.array(
        [f"GL-{rng.integers(1000, 9999)}" for _ in range(n)], dtype=object
    )
    data["cost_center"] = np.array(
        [f"CC-{rng.integers(100, 999)}" for _ in range(n)], dtype=object
    )
    data["entity"] = pick(rng, ENTITY_REF, n)

    for i in range(N_EXTRA):
        fam = i % 3
        if fam == 0:
            data[f"measure_{i:02d}"] = np.round(rng.normal(1000, 250, size=n), 2)
        elif fam == 1:
            data[f"tag_{i:02d}"] = pick(rng, ("alpha", "beta", "gamma", "delta"), n)
        else:
            d = base + rng.integers(0, 700, size=n).astype("timedelta64[D]")
            data[f"asof_{i:02d}"] = np.array([str(x) for x in d], dtype=object)

    df = pd.DataFrame(data)
    return _inject(df, rng, defect_rate)


def _inject(df: pd.DataFrame, rng: np.random.Generator, defect_rate: float | None) -> pd.DataFrame:
    n = len(df)

    def rate(base: float) -> float:
        return resolve_rate(base, defect_rate)

    # Each amount-string defect is derived from the *original* numeric value so
    # the families compose cleanly even when their masks overlap (which happens
    # under a high uniform defect_rate); the last-applied family simply wins.
    orig = df["amount"].astype(float).to_numpy()
    amt = df["amount"].to_numpy().astype(object)

    # 10% currency-symbol amount strings. Only the "$" symbol is used: FreshData's
    # generic (non-domain) dtype coercion strips "$" and thousands commas but not
    # foreign-currency *words* ("EUR 500") — those are finance-domain-pack
    # territory, so injecting them here would test out-of-scope behaviour.
    m = defect_mask(rng, n, rate(0.10))
    for i in np.where(m)[0]:
        amt[i] = f"${orig[i]:,.2f}"

    # 5% comma-thousands strings
    m = defect_mask(rng, n, rate(0.05))
    for i in np.where(m)[0]:
        amt[i] = f"{orig[i]:,.2f}"

    # 3% accounting-negative parentheses. Generic coercion does not understand
    # "(1234.56)", so these cells are coerced to NaN (flagged-as-missing) while
    # the column as a whole still crosses numeric_threshold and becomes float —
    # i.e. the in-scope outcome is "column typed, un-coercible cell nulled".
    m = defect_mask(rng, n, rate(0.03))
    for i in np.where(m)[0]:
        amt[i] = f"({orig[i]:.2f})"
    df["amount"] = amt

    # 4% currency outside reference set
    m = defect_mask(rng, n, rate(0.04))
    df.loc[m, "currency"] = pick(rng, BAD_CURRENCY, int(m.sum()))

    # 3% mixed debit_credit representations
    m = defect_mask(rng, n, rate(0.03))
    df.loc[m, "debit_credit"] = pick(rng, ("DR", "CR", "debit", "credit"), int(m.sum()))

    # 6% missing value_date (flag, do not fill)
    m = defect_mask(rng, n, rate(0.06))
    df.loc[m, "value_date"] = None

    # 1% posting_date ambiguous locale (DD/MM that is not also valid MM/DD)
    m = defect_mask(rng, n, rate(0.01))
    alt = []
    for v in df["posting_date"].to_numpy()[m]:
        try:
            ts = pd.Timestamp(np.datetime64(v))
            alt.append(ts.strftime("%d/%m/%Y"))
        except Exception:
            alt.append(v)
    df.loc[m, "posting_date"] = alt

    # 2% exact-duplicate transaction rows
    k = int(round(rate(0.02) * n))
    if k:
        dup_idx = rng.permutation(n)[:k]
        df = pd.concat([df, df.iloc[dup_idx].copy()], ignore_index=True)

    return df.reset_index(drop=True)


GOLD_LABELS: dict[str, dict[str, Any]] = gold_to_records(
    {
        "transaction_id": GoldLabel(ROLE_ID, "object", "preserve", True),
        "account_code": GoldLabel(ROLE_ID, "object", "preserve", True),
        "posting_date": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False),
        "value_date": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "flag_missing", False),
        "amount": GoldLabel(ROLE_NUMERIC, "float64", "coerce_numeric", False),
        "currency": GoldLabel(ROLE_CATEGORICAL, "object", "reference_flag", False, CURRENCY_REF),
        "debit_credit": GoldLabel(ROLE_CATEGORICAL, "object", "preserve", False, DEBIT_CREDIT_REF),
        "gl_account": GoldLabel(ROLE_TEXT, "object", "preserve", False),
        "cost_center": GoldLabel(ROLE_TEXT, "object", "preserve", False),
        "entity": GoldLabel(ROLE_CATEGORICAL, "object", "preserve", False, ENTITY_REF),
    }
)

DEFECT_MANIFEST: list[dict[str, Any]] = manifest_to_records(
    [
        Defect("fin-amt-symbol", "amount", "currency_symbol_string", 0.10, "dtype_coerce"),
        Defect("fin-amt-comma", "amount", "thousands_separator", 0.05, "dtype_coerce"),
        Defect("fin-amt-paren", "amount", "accounting_negative", 0.03, "dtype_coerce"),
        Defect("fin-currency-ref", "currency", "reference_violation", 0.04, "reference_flag"),
        Defect("fin-dc-mixed", "debit_credit", "mixed_encoding", 0.03, "preserve"),
        Defect("fin-vdate-missing", "value_date", "missing_date", 0.06, "flag_missing", preservation=True),
        Defect("fin-pdate-locale", "posting_date", "ambiguous_locale_date", 0.01, "dtype_coerce"),
        Defect("fin-dupes", "transaction_id", "exact_duplicate_row", 0.02, "drop_duplicate"),
    ]
)

SCALE_VARIANTS = (10_000, 500_000, 5_000_000, 25_000_000)
ID_COLUMNS = ("transaction_id", "account_code")
TARGET_COLUMN: str | None = None
TEXT_COLUMNS = ("gl_account", "cost_center")
N_COLS = 60
