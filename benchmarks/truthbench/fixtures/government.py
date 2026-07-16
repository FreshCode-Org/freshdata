"""Deterministic government gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"gov-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "district_id": [
                "001",
                "002",
                "003",
                "004",
                "005",
                "006",
                "007",
                "008",
                "009",
                "010",
                "011",
                "012",
                "013",
                "014",
                "015",
                "016",
            ],
            "case_id": [
                "000001",
                "000123",
                "000003",
                "000004",
                "000005",
                "000006",
                "000007",
                "000008",
                "000009",
                "000010",
                "000011",
                "000012",
                "000013",
                "000014",
                "000015",
                "000016",
            ],
            "agency": [
                "Revenue Department",
                "भारत सरकार / Government of India",
                "Ministry of Health",
                "City Council",
                "税務局",
                "Prefecture Office",
                "State Secretariat",
                "Public Works",
                "Education Board",
                "Transport Authority",
                "Treasury",
                "Registry",
                "Civil Court",
                "Labor Office",
                "Statistics Bureau",
                "Revenue Department",
            ],
            "fiscal_year": ["2026"] * 16,
            "calendar_year": ["2026"] * 16,
            "language": ["en"] * 16,
            "notes": ["routine"] * 16,
            "encoding": ["UTF-8"] * 16,
            "retention_policy": ["retain 7 years"] * 16,
            "repair_policy": ["repair after 30 days"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "government",
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
            "supported_languages": ["en", "hi", "de", "zh"],
            "protected_case_policy": "preserve",
        },
        protected_columns=("district_id", "case_id"),
    )
    builder.inject(
        "gov-01", "district_id", "007", Disposition.PRESERVE, family="leading-zero-district-id"
    )
    builder.inject(
        "gov-02", "case_id", "000123", Disposition.PRESERVE, family="leading-zero-case-id"
    )
    builder.inject(
        "gov-03",
        "agency",
        "भारत सरकार / Government of India",
        Disposition.PRESERVE,
        family="indian-international-grouping",
    )
    builder.inject(
        "gov-05",
        "fiscal_year",
        "2025-26",
        Disposition.REVIEW,
        family="fiscal-calendar-year-conflict",
    )
    builder.inject(
        "gov-06",
        "encoding",
        "CafÃ©",
        Disposition.REPAIR,
        expected="Café",
        family="mixed-legacy-encoding",
    )
    builder.inject(
        "gov-07", "language", "हिन्दी / English", Disposition.PRESERVE, family="multilingual-agency"
    )
    builder.inject(
        "gov-09",
        "notes",
        "TB-GOV-NATIONAL-ID-0001",
        Disposition.FLAG,
        family="restricted-national-id",
        sensitive=True,
    )
    builder.inject(
        "gov-11",
        "retention_policy",
        "retain 7 years",
        Disposition.REVIEW,
        family="contradictory-retention-repair-policy",
    )
    builder.inject(
        "gov-11",
        "repair_policy",
        "repair after 30 days",
        Disposition.REVIEW,
        family="contradictory-retention-repair-policy",
    )
    builder.inject(
        "gov-16",
        "case_id",
        "TB-GOV-CASE-TAIL",
        Disposition.PRESERVE,
        family="protected-case-id-conflict",
        sensitive=True,
    )
    builder.add_row_case(
        "exact-duplicate-gov-02-gov-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-gov-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case(
        "type-drifted-fiscal-year", Disposition.REVIEW, family="type-drifted-column"
    )
    return builder.build()
