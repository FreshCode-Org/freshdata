"""Deterministic retail gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"ret-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "sku": [
                "001234",
                "000007",
                "123456",
                "888888",
                "100001",
                "100002",
                "100003",
                "100004",
                "100005",
                "100006",
                "100007",
                "100008",
                "100009",
                "100010",
                "100011",
                "100012",
            ],
            "gtin": [
                "00012345678905",
                "00000012345678",
                "12345678901234",
                "88888888888888",
                "10000000000001",
                "10000000000002",
                "10000000000003",
                "10000000000004",
                "10000000000005",
                "10000000000006",
                "10000000000007",
                "10000000000008",
                "10000000000009",
                "10000000000010",
                "10000000000011",
                "10000000000012",
            ],
            "product_name": [
                "Widget",
                "Free Sample",
                "CafÃ© &amp; Tea",
                "茶",
                "Book",
                "Lamp",
                "Chair",
                "Desk",
                "Glass",
                "Mug",
                "Bag",
                "Pencil",
                "Shoes",
                "Coat",
                "Hat",
                "Socks",
            ],
            "quantity": [1, 1, 2, 1, 1, -2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "price": [
                "12.50",
                "0.00",
                "1.234,56",
                "₹1,23,456.70",
                "9.99",
                "10.00",
                "11.00",
                "12.00",
                "13.00",
                "14.00",
                "15.00",
                "16.00",
                "17.00",
                "18.00",
                "19.00",
                "20.00",
            ],
            "currency": [
                "USD",
                "USD",
                "EUR",
                "INR",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
                "USD",
            ],
            "review": ["great"] * 16,
            "email": ["ordinary@example.invalid"] * 16,
            "card": ["TB-RETAIL-CARD-0001"] * 16,
            "order_date": ["2026-01-15"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "retail",
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
            "locales": ["en_US", "de_DE", "hi_IN", "zh_CN"],
            "currency": "USD",
        },
        protected_columns=("sku", "gtin"),
    )
    builder.inject("ret-01", "sku", "001234", Disposition.PRESERVE, family="leading-zero-sku")
    builder.inject(
        "ret-01", "gtin", "00012345678905", Disposition.PRESERVE, family="leading-zero-gtin"
    )
    builder.inject("ret-02", "price", "0.00", Disposition.PRESERVE, family="free-item")
    builder.inject("ret-06", "quantity", -2, Disposition.FLAG, family="return-quantity")
    builder.inject(
        "ret-03",
        "price",
        "1.234,56",
        Disposition.REPAIR,
        expected=1234.56,
        expected_dtype="float64",
        family="mixed-decimal-grouping",
    )
    builder.inject("ret-04", "price", "₹1,23,456.70", Disposition.REVIEW, family="mixed-currency")
    builder.inject("ret-05", "currency", "EUR", Disposition.REVIEW, family="mixed-currency")
    builder.inject(
        "ret-03",
        "product_name",
        "CafÃ© &amp; Tea",
        Disposition.REVIEW,
        family="html-entity-mojibake",
    )
    builder.inject(
        "ret-04", "product_name", "茶", Disposition.PRESERVE, family="multilingual-product"
    )
    builder.inject(
        "ret-07",
        "review",
        "customer@example.invalid",
        Disposition.REVIEW,
        family="email-in-review",
        sensitive=True,
    )
    builder.inject(
        "ret-08",
        "card",
        "TB-RETAIL-CARD-REVIEW",
        Disposition.REVIEW,
        family="card-in-review",
        sensitive=True,
    )
    builder.inject(
        "ret-16",
        "email",
        "tail@example.invalid",
        Disposition.FLAG,
        family="tail-email-canary",
        sensitive=True,
    )
    builder.add_row_case(
        "exact-duplicate-ret-02-ret-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-ret-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case(
        "type-drifted-quantity", Disposition.REVIEW, family="type-drifted-column"
    )
    return builder.build()
