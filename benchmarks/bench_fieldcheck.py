"""Benchmark fd.validate_fields against regex and pandas-coercion baselines.

Reproducible (seeded) invalid-cell detection benchmark on a synthetic
financial feed with labeled corruptions::

    python benchmarks/bench_fieldcheck.py            # default 20k rows
    python benchmarks/bench_fieldcheck.py 100000     # custom row count

For each validator we report precision / recall / F1 over the labeled
invalid cells, plus wall time and throughput. Baselines:

* ``regex``   — per-column regular expressions only (no context, no ranges);
* ``pandas``  — ``pd.to_numeric`` / ``pd.to_datetime`` coercion NaN-diffing
  (types only: no vocabulary, format or semantic checks);
* ``fieldcheck`` — ``fd.validate_fields`` with a declared schema.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time

import pandas as pd

from freshdata.fieldcheck import FieldSpec, validate_fields

SEED = 20260711
CORRUPTION_RATE = 0.02

SCHEMA = {
    "transaction_id": FieldSpec(semantic_type="identifier", required=True, nullable=False),
    "transaction_amount": FieldSpec(semantic_type="currency_amount"),
    "currency": FieldSpec(allowed_values=frozenset({"USD", "EUR", "GBP", "INR"})),
    "stock_ticker": FieldSpec(semantic_type="ticker"),
    "interest_rate": FieldSpec(semantic_type="rate", min_value=0, max_value=1),
    "transaction_date": FieldSpec(semantic_type="date"),
}

#: (column, corrupt value) pool — all are invalid for their column.
CORRUPTIONS = [
    ("transaction_amount", "apple"),
    ("transaction_amount", "approved"),
    ("transaction_amount", "12O.50"),
    ("currency", "BTC"),
    ("currency", "dollars"),
    ("stock_ticker", "apple"),
    ("stock_ticker", "AА PL"),
    ("interest_rate", "high"),
    ("interest_rate", "5"),          # out of [0, 1]
    ("transaction_date", "2026-02-30"),
    ("transaction_date", "not a date"),
]


def make_dataset(n: int) -> tuple[pd.DataFrame, set]:
    rng = random.Random(SEED)
    tickers = ["AAPL", "MSFT", "GOOG", "TSLA", "BRK.B", "AMZN"]
    rows = []
    for i in range(n):
        rows.append({
            "transaction_id": f"T{i:07d}",
            "transaction_amount": f"{rng.uniform(1, 5000):.2f}",
            "currency": rng.choice(["USD", "EUR", "GBP", "INR"]),
            "stock_ticker": rng.choice(tickers),
            "interest_rate": f"{rng.uniform(0.001, 0.2):.4f}",
            "transaction_date": f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        })
    df = pd.DataFrame(rows)
    truth: set = set()
    n_bad = max(1, int(n * CORRUPTION_RATE))
    for row in rng.sample(range(n), n_bad):
        col, bad = rng.choice(CORRUPTIONS)
        df.loc[row, col] = bad
        truth.add((row, col))
    return df, truth


# --- baselines ---------------------------------------------------------------

REGEXES = {
    "transaction_id": re.compile(r"^T\d{7}$"),
    "transaction_amount": re.compile(r"^-?\d+(\.\d+)?$"),
    "currency": re.compile(r"^[A-Z]{3}$"),
    "stock_ticker": re.compile(r"^[A-Z]{1,6}([.\-][A-Z0-9]{1,4})?$"),
    "interest_rate": re.compile(r"^-?\d+(\.\d+)?$"),
    "transaction_date": re.compile(r"^\d{4}-\d{2}-\d{2}$"),
}


def regex_validator(df: pd.DataFrame) -> set:
    flagged = set()
    for col, rx in REGEXES.items():
        bad = ~df[col].astype(str).str.fullmatch(rx)
        flagged.update((row, col) for row in df.index[bad])
    return flagged


def pandas_validator(df: pd.DataFrame) -> set:
    flagged = set()
    for col in ("transaction_amount", "interest_rate"):
        bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
        flagged.update((row, col) for row in df.index[bad])
    dates = pd.to_datetime(df["transaction_date"], errors="coerce", format="%Y-%m-%d")
    bad = dates.isna() & df["transaction_date"].notna()
    flagged.update((row, "transaction_date") for row in df.index[bad])
    return flagged


def fieldcheck_validator(df: pd.DataFrame) -> set:
    report = validate_fields(df, SCHEMA)
    return {(i.row, i.column) for i in report.issues if i.severity == "error"}


# --- scoring -----------------------------------------------------------------


def score(name: str, fn, df: pd.DataFrame, truth: set) -> dict:
    start = time.perf_counter()
    flagged = fn(df)
    elapsed = time.perf_counter() - start
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn_ = len(truth - flagged)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn_) if tp + fn_ else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "validator": name,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_accepts": fn_,
        "false_rejects": fp,
        "seconds": round(elapsed, 3),
        "rows_per_s": int(len(df) / elapsed) if elapsed else None,
    }


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000
    df, truth = make_dataset(n)
    print(f"rows={n} corrupted_cells={len(truth)} seed={SEED}\n")
    results = [
        score("regex", regex_validator, df, truth),
        score("pandas_coercion", pandas_validator, df, truth),
        score("fd.validate_fields", fieldcheck_validator, df, truth),
    ]
    print(json.dumps(results, indent=2))
    print()
    header = f"{'validator':<20}{'precision':>10}{'recall':>8}{'f1':>8}{'sec':>8}{'rows/s':>10}"
    print(header)
    for r in results:
        print(f"{r['validator']:<20}{r['precision']:>10}{r['recall']:>8}"
              f"{r['f1']:>8}{r['seconds']:>8}{r['rows_per_s']:>10}")


if __name__ == "__main__":
    main()
