"""Deterministic finance gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"fin-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "account_id": [f"TB-FIN-{i:04d}" for i in range(1, 17)],
            "asset": [
                "bond",
                "fund",
                "apple",
                "stock",
                "cash",
                "fx",
                "etf",
                "bond",
                "fund",
                "stock",
                "cash",
                "fx",
                "etf",
                "bond",
                "fund",
                "stock",
            ],
            "company": [
                "Acme Bank",
                "Northstar",
                "Acme",
                "Apple",
                "Cedar",
                "Delta",
                "Evergreen",
                "Futura",
                "Globex",
                "Helios",
                "Ion",
                "Juniper",
                "Kappa",
                "Lumen",
                "Mosaic",
                "Nadir",
            ],
            "ticker": [
                "ACME",
                "NSTAR",
                "APPL",
                "AAPL",
                "CEDR",
                "DLTA",
                "EVER",
                "FUTR",
                "GLOB",
                "HELI",
                "ION",
                "JUN",
                "KAPP",
                "LUME",
                "MOSC",
                "NADI",
            ],
            "price": [
                10.25,
                0.00,
                "apple",
                155.20,
                -4.5,
                9999999999.99,
                "₹1,23,456.70",
                123.45,
                9.99,
                42.0,
                18.75,
                21.0,
                33.0,
                7.5,
                8.25,
                12.0,
            ],
            "currency": [
                "USD",
                "USD",
                "USD",
                "USD",
                "EUR",
                "USD",
                "INR",
                "EUR",
                "USD",
                "USD",
                "USD",
                "EUR",
                "INR",
                "USD",
                "USD",
                "USD",
            ],
            "trade_date": ["2026-01-15"] * 16,
            "memo": ["ordinary"] * 16,
            "balance": [
                100.0,
                200.0,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
                1300.0,
                1400.0,
                1500.0,
                1600.0,
            ],
            "locale": ["en_US"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "finance",
        frame,
        seed=int(seed),
        schema={
            "columns": list(frame.columns),
            "dtypes": {c: str(frame[c].dtype) for c in frame.columns},
        },
        policy={
            "reference_date": "2026-01-15",
            "timezone": "UTC",
            "locale": "en_US",
            "currency": "USD",
            "supported_currencies": ["USD", "EUR", "INR"],
            "protected_ticker_policy": "preserve",
        },
        protected_columns=("account_id", "ticker"),
    )
    builder.inject("fin-03", "price", "apple", Disposition.REVIEW, family="semantic-apple-price")
    builder.inject("fin-04", "company", "Apple", Disposition.PRESERVE, family="apple-company")
    builder.inject(
        "fin-04", "ticker", "AAPL", Disposition.REVIEW, family="protected-ticker-policy-conflict"
    )
    builder.inject("fin-02", "price", "0.00", Disposition.PRESERVE, family="zero-price")
    builder.inject("fin-05", "price", -4.5, Disposition.FLAG, family="negative-value")
    builder.inject("fin-06", "price", 9999999999.99, Disposition.FLAG, family="extreme-value")
    builder.inject(
        "fin-07",
        "price",
        "₹1,23,456.70",
        Disposition.REPAIR,
        expected=123456.70,
        expected_dtype="float64",
        family="indian-grouped-currency",
    )
    builder.inject("fin-08", "currency", "EUR", Disposition.REVIEW, family="usd-eur-inr-conflict")
    builder.inject("fin-09", "currency", "INR", Disposition.REVIEW, family="usd-eur-inr-conflict")
    builder.inject(
        "fin-10", "trade_date", "01/02/2025", Disposition.REVIEW, family="ambiguous-date-format"
    )
    builder.inject(
        "fin-11",
        "memo",
        "tb.finance+memo@example.invalid",
        Disposition.FLAG,
        family="invisible-pii-memo",
        sensitive=True,
    )
    builder.inject(
        "fin-12",
        "memo",
        "tb.finance+hidden@example.invalid\u200b",
        Disposition.FLAG,
        family="zero-width-memo",
    )
    builder.inject(
        "fin-16",
        "account_id",
        "TB-FIN-ACCOUNT-TAIL",
        Disposition.FLAG,
        family="tail-row-account-canary",
        sensitive=True,
    )
    builder.add_row_case(
        "exact-duplicate-fin-02-fin-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-fin-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case("type-drifted-price", Disposition.REVIEW, family="type-drifted-column")
    return builder.build()
