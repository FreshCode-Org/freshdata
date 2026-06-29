"""Fixture 5 — Document-extracted tables (OCR / PDF provenance).

Simulates an invoice/statement table lifted from a PDF by an OCR pipeline,
together with a parallel per-column provenance frame (source_file, page_number,
region_id, parser_confidence, extraction_ts). Defects are the OCR-typical kind
FreshData's ``source_provenance=`` path is meant to audit: low-confidence
amount cells, letter-for-digit OCR artifacts in numbers, garbled dates, and a
missing invoice_number (role=id, flag only).

``generate`` returns the data frame; ``generate_provenance`` returns the
parallel column-level provenance dict suitable for ``fd.clean(df,
source_provenance=...)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import (
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

N_EXTRA = 10  # 8 named + 10 = 18 columns
_TERMS = ("net15", "net30", "net60", "due_on_receipt")
_VENDORS = ("Acme Corp", "Globex", "Initech", "Umbrella", "Wonka Inc", "Stark Ind")
AMOUNT_COLS = ("line_amount", "tax_amount", "total_amount")


def generate(n_rows: int, seed: int = 42, defect_rate: float | None = None) -> pd.DataFrame:
    rng = rng_for(seed)
    n = int(n_rows)
    data: dict[str, Any] = {}

    data["record_id"] = uuid_series(rng, n, prefix="rec-")
    data["vendor_name"] = pick(rng, _VENDORS, n)
    data["invoice_number"] = np.array(
        [f"INV-{rng.integers(100000, 999999)}" for _ in range(n)], dtype=object
    )
    line = np.round(rng.lognormal(5, 0.8, size=n), 2)
    tax = np.round(line * 0.08, 2)
    total = np.round(line + tax, 2)
    data["line_amount"] = np.array([f"{a:.2f}" for a in line], dtype=object)
    data["tax_amount"] = np.array([f"{a:.2f}" for a in tax], dtype=object)
    data["total_amount"] = np.array([f"{a:.2f}" for a in total], dtype=object)
    base = np.datetime64("2023-01-01")
    due = base + rng.integers(0, 400, size=n).astype("timedelta64[D]")
    data["due_date"] = np.array([str(d) for d in due], dtype=object)
    data["payment_terms"] = pick(rng, _TERMS, n)

    for i in range(N_EXTRA):
        if i % 2 == 0:
            data[f"field_num_{i:02d}"] = np.round(rng.normal(100, 30, size=n), 2)
        else:
            data[f"field_cat_{i:02d}"] = pick(rng, ("x", "y", "z"), n)

    df = pd.DataFrame(data)
    df, conf = _inject(df, rng, defect_rate)
    # stash the parser confidence so generate_provenance can reuse it
    df.attrs["_parser_confidence"] = conf
    df.attrs["_seed"] = seed
    return df


def _inject(df: pd.DataFrame, rng: np.random.Generator, defect_rate: float | None):
    n = len(df)

    def rate(base: float) -> float:
        return resolve_rate(base, defect_rate)

    # parser confidence per row, default high
    conf = np.round(rng.uniform(0.75, 0.99, size=n), 3)

    # 10% low-confidence cells on amount columns
    m = defect_mask(rng, n, rate(0.10))
    conf[m] = np.round(rng.uniform(0.1, 0.49, size=int(m.sum())), 3)

    # 5% OCR artifacts: letter-O for zero in an amount column
    m = defect_mask(rng, n, rate(0.05))
    col = df["line_amount"].to_numpy(copy=True)
    for i in np.where(m)[0]:
        col[i] = str(col[i]).replace("0", "O", 1)
    df["line_amount"] = col

    # 3% garbled due_date (OCR line-break artifact)
    m = defect_mask(rng, n, rate(0.03))
    dd = df["due_date"].to_numpy(copy=True)
    for i in np.where(m)[0]:
        dd[i] = str(dd[i]).replace("-", "- ") + "\n"
    df["due_date"] = dd

    # 2% missing invoice_number (role=id, flag only)
    m = defect_mask(rng, n, rate(0.02))
    df.loc[m, "invoice_number"] = None

    return df.reset_index(drop=True), conf


def generate_provenance(n_rows: int, seed: int = 42, defect_rate: float | None = None) -> dict[str, Any]:
    """Per-column provenance metadata for ``fd.clean(df, source_provenance=...)``.

    Returns a dict keyed by amount column name -> provenance summary, matching
    the column-level shape FreshData accepts. parser_confidence is the per-row
    mean confidence for that fixture seed.
    """
    df = generate(n_rows, seed=seed, defect_rate=defect_rate)
    conf = df.attrs["_parser_confidence"]
    rng = rng_for(seed + 1)
    out: dict[str, Any] = {}
    for c in AMOUNT_COLS:
        out[c] = {
            "source_file": "statements_2023.pdf",
            "page_number": int(rng.integers(1, 50)),
            "region_id": f"tbl-{rng.integers(1, 9)}",
            "parser_confidence": float(np.round(conf.mean(), 3)),
            "extraction_ts": "2023-02-01T09:00:00",
        }
    return out


GOLD_LABELS: dict[str, dict[str, Any]] = gold_to_records(
    {
        "record_id": GoldLabel(ROLE_ID, "object", "preserve", True),
        "vendor_name": GoldLabel(ROLE_TEXT, "object", "preserve", True),
        "invoice_number": GoldLabel(ROLE_ID, "object", "flag_missing", True),
        "line_amount": GoldLabel(ROLE_NUMERIC, "float64", "coerce_numeric", False),
        "tax_amount": GoldLabel(ROLE_NUMERIC, "float64", "coerce_numeric", False),
        "total_amount": GoldLabel(ROLE_NUMERIC, "float64", "coerce_numeric", False),
        "due_date": GoldLabel(ROLE_DATETIME, "datetime64[ns]", "coerce_datetime", False),
        "payment_terms": GoldLabel(ROLE_CATEGORICAL, "object", "preserve", False, _TERMS),
    }
)

DEFECT_MANIFEST: list[dict[str, Any]] = manifest_to_records(
    [
        Defect("prov-lowconf", "line_amount|tax_amount|total_amount", "low_parser_confidence", 0.10, "provenance_flag"),
        Defect("prov-ocr-digit", "line_amount", "ocr_letter_for_digit", 0.05, "dtype_coerce_or_flag"),
        Defect("prov-garbled-date", "due_date", "ocr_garbled_date", 0.03, "dtype_coerce_or_flag"),
        Defect("prov-missing-invoice", "invoice_number", "missing_id", 0.02, "flag_missing", preservation=True),
    ]
)

SCALE_VARIANTS = (1_000, 10_000, 100_000)
ID_COLUMNS = ("record_id", "invoice_number")
TARGET_COLUMN: str | None = None
TEXT_COLUMNS = ("vendor_name",)
N_COLS = 18
