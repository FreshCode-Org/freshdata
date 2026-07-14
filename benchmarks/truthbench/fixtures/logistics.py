"""Deterministic logistics gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"log-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "shipment_id": [f"TB-LOG-{i:04d}" for i in range(1, 17)],
            "origin_code": ["USLAX"] * 16,
            "destination_code": [
                "NLRTM",
                "DEHAM",
                "GBFXT",
                "INBOM",
                "SGSIN",
                "CNSHA",
                "JPTYO",
                "AUMEL",
                "BRSSZ",
                "ZADUR",
                "AEJEA",
                "NOOSL",
                "ESBCN",
                "CATOR",
                "MYPKG",
                "INBOM",
            ],
            "weight": [
                10.0,
                "10 lb",
                12.0,
                8.0,
                4.0,
                15.0,
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
                60.0,
                65.0,
            ],
            "weight_unit": ["kg"] * 16,
            "temperature": [
                20.0,
                21.0,
                98.6,
                19.0,
                18.0,
                22.0,
                23.0,
                24.0,
                25.0,
                26.0,
                27.0,
                28.0,
                29.0,
                30.0,
                31.0,
                32.0,
            ],
            "temperature_unit": ["C"] * 16,
            "delivery_window": ["2026-01-15 09:00-17:00"] * 16,
            "transport_time": ["12:00"] * 16,
            "address": ["Warehouse"] * 16,
            "tracking_status": ["on-time"] * 16,
            "timezone": ["UTC"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "logistics",
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
            "weight_unit": "kg",
            "temperature_unit": "C",
            "protected_shipment_policy": "preserve",
        },
        protected_columns=("shipment_id",),
    )
    builder.inject(
        "log-04", "destination_code", "INBOM", Disposition.PRESERVE, family="rare-unlocode-valid"
    )
    builder.inject(
        "log-02",
        "weight",
        "10 lb",
        Disposition.REPAIR,
        expected=4.5359237,
        expected_dtype="float64",
        family="kg-lb-conversion",
    )
    builder.inject("log-02", "weight_unit", "lb", Disposition.PRESERVE, family="weight-unit-lb")
    builder.inject(
        "log-03", "temperature", 98.6, Disposition.REVIEW, family="celsius-fahrenheit-conflict"
    )
    builder.inject(
        "log-03", "temperature_unit", "F", Disposition.PRESERVE, family="temperature-unit-f"
    )
    builder.inject(
        "log-05",
        "delivery_window",
        "2026-01-15 23:30-2026-01-16 01:00",
        Disposition.REVIEW,
        family="cross-timezone-window",
    )
    builder.inject(
        "log-05",
        "timezone",
        "Asia/Kolkata→America/New_York",
        Disposition.REVIEW,
        family="cross-timezone-window",
    )
    builder.inject(
        "log-06",
        "transport_time",
        "24:00",
        Disposition.REPAIR,
        expected="00:00",
        family="twentyfour-hour-transport",
    )
    builder.inject(
        "log-09",
        "address",
        "TB-LOG-ADDRESS-0001",
        Disposition.FLAG,
        family="address-pii",
        sensitive=True,
    )
    builder.inject(
        "log-15", "tracking_status", "delayed", Disposition.FLAG, family="late-tracking-canary"
    )
    builder.inject(
        "log-16",
        "shipment_id",
        "TB-LOG-SHIPMENT-TAIL",
        Disposition.REVIEW,
        family="protected-shipment-id-conflict",
        sensitive=True,
    )
    builder.add_row_case(
        "exact-duplicate-log-02-log-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-log-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case(
        "type-drifted-weight", Disposition.REVIEW, family="type-drifted-column"
    )
    return builder.build()
