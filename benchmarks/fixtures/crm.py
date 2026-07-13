"""Fixture 1 — CRM customer records (40 columns).

A wide customer table shaped like a real CRM export: an id, free-text name,
contact fields, reference-constrained categoricals (country, account status),
a pair of dates, a monetary lifetime value, a segment, plus 30 generated score
/ flag / date columns. Defects are limited to FreshData's declared repair
scope: sentinel tokens, missing numerics, reference-set violations, exact
duplicate rows, casing/whitespace noise, non-ISO date strings, and the
preservation checks (null id, whitespace-only name) that must *not* be
repaired.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import (
    ACCOUNT_STATUS_REF,
    BAD_COUNTRY,
    COUNTRY_REF,
    ROLE_CATEGORICAL,
    ROLE_DATETIME,
    ROLE_ID,
    ROLE_NUMERIC,
    ROLE_TEXT,
    Defect,
    GoldLabel,
    defect_mask,
    format_iso_date,
    gold_to_records,
    manifest_to_records,
    pick,
    resolve_rate,
    rng_for,
    uuid_series,
)

N_EXTRA = 30  # 10 named + 30 generated = 40 columns

# Tokens FreshData's normalize_sentinels actually recognises. "999" is
# deliberately excluded: the library does not treat it as missing (it is a
# legitimate number), so injecting it as a "sentinel" would test out-of-scope
# behaviour (HARD CONSTRAINT 5).
_NUMERIC_SENTINELS = ("N/A", "null", "--", "n.a.")
# Name pools large enough that ``full_name`` is genuinely free text: three
# tokens (>= 3 words) and high cardinality (unique_ratio > 0.6) so FreshData's
# role detector classifies it as role=text and never force-fills it. A smaller
# two-token name is inferred as a low-cardinality *categorical* and would
# (correctly, by the engine's own rules) be mode/"Unknown"-filled.
_FIRST = ("Ada", "Bo", "Cy", "Di", "Eve", "Finn", "Gus", "Hana", "Ivo", "Jo",
          "Kai", "Lia", "Mo", "Noa", "Ola", "Pia", "Rex", "Sky", "Tio", "Uma",
          "Vic", "Wyn", "Xan", "Yas", "Zev", "Ari", "Bex", "Cleo", "Dex", "Esi",
          "Fai", "Gio", "Hugo", "Ines", "Jax", "Kit", "Lux", "Max", "Nia", "Ozzy")
_MIDDLE = ("Q.", "R.", "S.", "T.", "U.", "V.", "W.", "X.", "Y.", "Z.",
           "A.", "B.", "C.", "D.", "E.", "F.", "G.", "H.", "I.", "J.",
           "Rey", "Sol", "Tam", "Val", "Wren", "Ash", "Bay", "Cove", "Dove", "Elm",
           "Fox", "Gray", "Hale", "Iris", "Jett", "Kane", "Lark", "Moss", "Nash", "Oak")
_LAST = ("Lee", "Ng", "Park", "Rao", "Sun", "Tan", "Vue", "Wei", "Xu", "Yi",
         "Adler", "Boyd", "Cruz", "Diaz", "Eaton", "Frye", "Gold", "Holt", "Iqbal", "Jung",
         "Khan", "Lowe", "Mora", "Nair", "Ohms", "Pace", "Quinn", "Reed", "Shah", "Toth",
         "Udall", "Vance", "Walsh", "Xiong", "Yoon", "Zane", "Abara", "Bose", "Choi", "Dunn")


def _names(rng: np.random.Generator, n: int) -> np.ndarray:
    f = pick(rng, _FIRST, n)
    m = pick(rng, _MIDDLE, n)
    last = pick(rng, _LAST, n)
    return np.array([f"{a} {b} {c}" for a, b, c in zip(f, m, last)], dtype=object)


def generate(n_rows: int, seed: int = 42, defect_rate: float | None = None) -> pd.DataFrame:
    rng = rng_for(seed)
    n = int(n_rows)

    data: dict[str, Any] = {}
    data["customer_id"] = uuid_series(rng, n, prefix="cust-")
    data["full_name"] = _names(rng, n)
    data["email"] = np.array([f"user{i}@example.com" for i in range(n)], dtype=object)
    data["phone"] = np.array(
        [f"+1-555-{rng.integers(1000, 9999)}" for _ in range(n)], dtype=object
    )
    data["country"] = pick(rng, COUNTRY_REF, n)
    data["account_status"] = pick(rng, ACCOUNT_STATUS_REF, n)
    # base ISO dates as strings (fix_dtypes should coerce to datetime)
    base = np.datetime64("2021-01-01")
    signup = base + rng.integers(0, 900, size=n).astype("timedelta64[D]")
    data["signup_date"] = np.array([str(d) for d in signup], dtype=object)
    lastp = signup + rng.integers(1, 400, size=n).astype("timedelta64[D]")
    data["last_purchase_date"] = np.array([str(d) for d in lastp], dtype=object)
    # skewed monetary value so the engine's robust default (median) is chosen
    data["lifetime_value"] = np.round(rng.lognormal(6.0, 1.1, size=n), 2)
    data["segment"] = pick(rng, ("smb", "mid", "enterprise", "consumer"), n)

    # 30 generated columns: scores (float), flags (bool-ish), dates
    for i in range(N_EXTRA):
        fam = i % 3
        col = f"score_{i:02d}" if fam == 0 else (f"flag_{i:02d}" if fam == 1 else f"date_{i:02d}")
        if fam == 0:
            data[col] = np.round(rng.normal(50, 12, size=n), 3)
        elif fam == 1:
            data[col] = pick(rng, ("yes", "no", "true", "false", "1", "0"), n)
        else:
            d = base + rng.integers(0, 1200, size=n).astype("timedelta64[D]")
            data[col] = np.array([str(x) for x in d], dtype=object)

    df = pd.DataFrame(data)
    return _inject(df, rng, defect_rate)


def _inject(df: pd.DataFrame, rng: np.random.Generator, defect_rate: float | None) -> pd.DataFrame:
    n = len(df)

    def rate(base: float) -> float:
        return resolve_rate(base, defect_rate)

    # 8% sentinels scattered across numeric score columns
    score_cols = [c for c in df.columns if c.startswith("score_")]
    for c in score_cols:
        m = defect_mask(rng, n, rate(0.08))
        vals = df[c].astype(object).to_numpy(copy=True)
        sent = pick(rng, _NUMERIC_SENTINELS, int(m.sum()))
        vals[m] = sent
        df[c] = vals

    # 6% missing lifetime_value (skewed -> median fill)
    m = defect_mask(rng, n, rate(0.06))
    df.loc[m, "lifetime_value"] = np.nan

    # 3% country outside reference set
    m = defect_mask(rng, n, rate(0.03))
    df.loc[m, "country"] = pick(rng, BAD_COUNTRY, int(m.sum()))

    # 5% mixed-case / whitespace account_status
    m = defect_mask(rng, n, rate(0.05))
    noisy = []
    raw = df["account_status"].to_numpy()
    for v in raw[m]:
        choice = rng.integers(0, 3)
        noisy.append(f"  {v.upper()} " if choice == 0 else (v.capitalize() if choice == 1 else f"{v} "))
    df.loc[m, "account_status"] = noisy

    # 3% signup_date in non-ISO formats
    m = defect_mask(rng, n, rate(0.03))
    alt = []
    for v in df["signup_date"].to_numpy()[m]:
        try:
            alt.append(
                format_iso_date(v, "%m/%d/%Y")
                if rng.integers(0, 2)
                else format_iso_date(v, "%b %d %Y")
            )
        except Exception:
            alt.append(v)
    df.loc[m, "signup_date"] = alt

    # 1% NULL customer_id (preservation check — must be flagged, never filled)
    m = defect_mask(rng, n, rate(0.01))
    df.loc[m, "customer_id"] = None

    # 0.5% missing full_name (free text — must never be force-filled).
    # NB: whitespace-only text is intentionally not used: FreshData strips
    # surrounding whitespace as in-scope representation repair, so "   " would
    # correctly become empty/NA. The free-text contract under test is the
    # stronger "never fabricate a value", so we inject genuine missingness.
    m = defect_mask(rng, n, rate(0.005))
    df.loc[m, "full_name"] = np.nan

    # 2% exact duplicate rows (appended verbatim, then index reset)
    k = int(round(rate(0.02) * n))
    if k:
        dup_idx = rng.permutation(n)[:k]
        dup = df.iloc[dup_idx].copy()
        df = pd.concat([df, dup], ignore_index=True)

    return df.reset_index(drop=True)


# -- ground truth ----------------------------------------------------------
GOLD_LABELS: dict[str, dict[str, Any]] = gold_to_records(
    {
        "customer_id": GoldLabel(ROLE_ID, "object", "preserve", True),
        "full_name": GoldLabel(ROLE_TEXT, "object", "preserve", True),
        "email": GoldLabel(ROLE_TEXT, "object", "preserve", True),
        "phone": GoldLabel(ROLE_TEXT, "object", "preserve", True),
        "country": GoldLabel(ROLE_CATEGORICAL, "object", "flag_unexpected", False, COUNTRY_REF),
        "account_status": GoldLabel(
            ROLE_CATEGORICAL, "object", "normalize_whitespace", False, ACCOUNT_STATUS_REF
        ),
        "signup_date": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False),
        "last_purchase_date": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False),
        "lifetime_value": GoldLabel(ROLE_NUMERIC, "float64", "median_fill", False),
        "segment": GoldLabel(ROLE_CATEGORICAL, "object", "preserve", False),
    }
)

DEFECT_MANIFEST: list[dict[str, Any]] = manifest_to_records(
    [
        Defect("crm-sentinels", "score_*", "numeric_sentinel", 0.08, "sentinel_normalize", note="tokens: N/A, null, --, n.a. (999 excluded — not a FreshData sentinel)"),
        Defect("crm-ltv-missing", "lifetime_value", "missing_numeric", 0.06, "median_fill"),
        Defect("crm-country-ref", "country", "reference_violation", 0.03, "reference_flag"),
        Defect("crm-status-noise", "account_status", "case_whitespace", 0.05, "normalize_whitespace"),
        Defect("crm-date-format", "signup_date", "non_iso_date", 0.03, "dtype_coerce"),
        Defect("crm-null-id", "customer_id", "null_id", 0.01, "preserve", preservation=True),
        Defect("crm-missing-name", "full_name", "missing_text", 0.005, "preserve", preservation=True),
        Defect("crm-dupes", "*", "exact_duplicate_row", 0.02, "drop_duplicate"),
    ]
)

SCALE_VARIANTS = (10_000, 100_000, 1_000_000, 5_000_000)
ID_COLUMNS = ("customer_id",)
TARGET_COLUMN: str | None = None
TEXT_COLUMNS = ("full_name", "email", "phone")
N_COLS = 40
