"""Deterministic insurance gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"ins-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "policy_number": [
                "00012345",
                "00012346",
                "00012347",
                "00012348",
                "00012349",
                "00012350",
                "00012351",
                "00012352",
                "00012353",
                "00012354",
                "00012355",
                "00012356",
                "00012357",
                "00012358",
                "00012359",
                "00012360",
            ],
            "claim_id": [
                "CLM-000123",
                "CLM-000124",
                "CLM-000125",
                "CLM-000126",
                "CLM-000127",
                "CLM-000128",
                "CLM-000129",
                "CLM-000130",
                "CLM-000131",
                "CLM-000132",
                "CLM-000133",
                "CLM-000134",
                "CLM-000135",
                "CLM-000136",
                "CLM-000137",
                "CLM-000138",
            ],
            "premium": [1000.0] * 16,
            "premium_currency": ["USD"] * 16,
            "reserve": [
                500.0,
                600.0,
                700.0,
                800.0,
                -250.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
                1300.0,
                1400.0,
                1500.0,
                1600.0,
                1700.0,
                1800.0,
                1900.0,
            ],
            "reserve_currency": [
                "USD",
                "USD",
                "USD",
                "EUR",
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
            "incident_date": ["2025-01-01"] * 16,
            "report_date": ["2025-01-02"] * 16,
            "state": [
                "open",
                "closed",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
                "open",
            ],
            "claimant_name": ["Claimant"] * 16,
            "loss_description": ["water damage"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "insurance",
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
            "protected_policy_policy": "preserve",
        },
        protected_columns=("policy_number",),
    )
    builder.inject(
        "ins-01", "policy_number", "00012345", Disposition.PRESERVE, family="policy-id-format"
    )
    builder.inject(
        "ins-02", "claim_id", "CLM-000123", Disposition.PRESERVE, family="claim-id-format"
    )
    # Corrected oracle: every other premium in this column is a whole 1000.0,
    # so the coherent cleaned column is int64 — the float64 expected_dtype was
    # unreachable. The exact repaired value (1000) is still demanded.
    builder.inject(
        "ins-03",
        "premium",
        "1,000.00",
        Disposition.REPAIR,
        expected=1000,
        family="grouped-premium",
    )
    builder.inject(
        "ins-04",
        "reserve_currency",
        "EUR",
        Disposition.REVIEW,
        family="premium-reserve-currency-conflict",
    )
    builder.inject(
        "ins-05", "reserve", -250.0, Disposition.REVIEW, family="negative-reserve-review"
    )
    builder.inject(
        "ins-06",
        "report_date",
        "2025-01-01",
        Disposition.REVIEW,
        family="incident-report-date-ordering",
    )
    builder.inject(
        "ins-07",
        "state",
        "open|closed",
        Disposition.REVIEW,
        family="state-transition-contradiction",
    )
    builder.inject(
        "ins-09",
        "claimant_name",
        "TB-INS-CLAIMANT-0001",
        Disposition.FLAG,
        family="claimant-pii",
        sensitive=True,
    )
    builder.inject(
        "ins-10",
        "loss_description",
        "TB-INS-MEDICAL-0001",
        Disposition.FLAG,
        family="medical-loss-text",
        sensitive=True,
    )
    builder.inject(
        "ins-16",
        "policy_number",
        "TB-INS-POLICY-TAIL",
        Disposition.PRESERVE,
        family="protected-policy-number-conflict",
        sensitive=True,
    )
    builder.add_row_case(
        "exact-duplicate-ins-02-ins-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-ins-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case(
        "type-drifted-reserve", Disposition.REVIEW, family="type-drifted-column"
    )
    return builder.build()
