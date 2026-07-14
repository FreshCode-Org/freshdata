"""Deterministic CRM gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"crm-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "customer_id": [f"TB-CRM-{i:04d}" for i in range(1, 17)],
            "first_name": [
                "Ana",
                "Jose\u0301",
                "Miyuki",
                "Zoë",
                "Liam",
                "Nia",
                "Omar",
                "Riya",
                "Noel",
                "Saanvi",
                "Evan",
                "Ivy",
                "Kai",
                "Mara",
                "Theo",
                "Uma",
            ],
            "last_name": [
                "Stone",
                "García",
                "Chen",
                "Müller",
                "Jones",
                "Patel",
                "Khan",
                "Roy",
                "Smith",
                "Das",
                "Lee",
                "Brown",
                "Cruz",
                "Mori",
                "Cole",
                "Singh",
            ],
            "email": [
                "ana@example.invalid",
                "jose@example.invalid",
                "miyuki@example.invalid",
                "zoe@example.invalid",
                "liam@example.invalid",
                "nia@example.invalid",
                "omar@example.invalid",
                "riya@example.invalid",
                "noel@example.invalid",
                "saanvi@example.invalid",
                "evan@example.invalid",
                "ivy@example.invalid",
                "kai@example.invalid",
                "mara@example.invalid",
                "theo@example.invalid",
                "uma@example.invalid",
            ],
            "phone": ["555-0101"] * 16,
            "country": [
                "US",
                "DE",
                "IN",
                "GB",
                "US",
                "CA",
                "AU",
                "JP",
                "BR",
                "FR",
                "US",
                "IN",
                "MX",
                "ES",
                "ZA",
                "SG",
            ],
            "language": [
                "en",
                "de",
                "hi",
                "en",
                "en",
                "fr",
                "en",
                "ja",
                "pt",
                "fr",
                "en",
                "hi",
                "es",
                "es",
                "en",
                "en",
            ],
            "signup_date": ["2026-01-15"] * 16,
            "lifecycle": [
                "lead",
                "prospect",
                "customer",
                "churned",
                "lead",
                "prospect",
                "customer",
                "lead",
                "prospect",
                "customer",
                "lead",
                "prospect",
                "customer",
                "lead",
                "prospect",
                "customer",
            ],
            "lead_source": [
                "web",
                "referral",
                "apple",
                "partner",
                "web",
                "event",
                "apple",
                "search",
                "web",
                "partner",
                "event",
                "web",
                "referral",
                "search",
                "web",
                "event",
            ],
            "employer": [
                "Acme",
                "Globex",
                "Apple",
                "Northstar",
                "Acme",
                "Cedar",
                "Apple",
                "Delta",
                "Evergreen",
                "Futura",
                "Globex",
                "Helios",
                "Apple",
                "Ion",
                "Juniper",
                "Kappa",
            ],
            "notes": ["ordinary"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "crm",
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
            "supported_countries": ["US", "DE", "IN", "GB"],
            "supported_languages": ["en", "de", "hi", "ja"],
            "protected_customer_id_policy": "preserve",
        },
        protected_columns=("customer_id",),
    )
    builder.inject(
        "crm-02",
        "first_name",
        "Jose\u0301",
        Disposition.REPAIR,
        expected="José",
        family="combining-unicode-name",
    )
    builder.inject(
        "crm-03",
        "email",
        "miyuki@example.invalid",
        Disposition.PRESERVE,
        family="reserved-invalid-email",
        sensitive=True,
    )
    builder.inject(
        "crm-04",
        "phone",
        "555 0101",
        Disposition.REPAIR,
        expected="555-0101",
        family="spaced-phone",
    )
    builder.inject("crm-05", "country", "US/CA", Disposition.REVIEW, family="ambiguous-country")
    builder.inject("crm-06", "language", "en/fr", Disposition.REVIEW, family="ambiguous-language")
    builder.inject(
        "crm-07", "signup_date", "01/02/2025", Disposition.REVIEW, family="ambiguous-date"
    )
    builder.inject(
        "crm-08", "lifecycle", "lead|churned", Disposition.REVIEW, family="lifecycle-contradiction"
    )
    builder.inject(
        "crm-09",
        "email",
        "TB-CRM-ZW-EMAIL",
        Disposition.FLAG,
        family="zero-width-email",
        sensitive=True,
    )
    builder.inject(
        "crm-11",
        "email",
        "lead\u200b@example.invalid",
        Disposition.FLAG,
        family="zero-width-email",
    )
    builder.inject(
        "crm-10", "notes", "TB-CRM-SSN-0001", Disposition.FLAG, family="hidden-ssn", sensitive=True
    )
    builder.inject(
        "crm-03", "lead_source", "apple", Disposition.PRESERVE, family="apple-lead-source"
    )
    builder.inject("crm-03", "employer", "Apple", Disposition.PRESERVE, family="apple-employer")
    builder.inject(
        "crm-16",
        "customer_id",
        "TB-CRM-CUSTOMER-TAIL",
        Disposition.REVIEW,
        family="protected-customer-id",
        sensitive=True,
    )
    builder.add_row_case(
        "exact-duplicate-crm-02-crm-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-crm-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case("type-drifted-phone", Disposition.REVIEW, family="type-drifted-column")
    return builder.build()
