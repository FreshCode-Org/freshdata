"""Deterministic healthcare gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"hc-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "mrn": [f"TB-HC-{i:04d}" for i in range(1, 17)],
            "patient_name": [
                "Amina Khan",
                "Luca Rossi",
                "Marta Silva",
                "Noah Reed",
                "Iris Chen",
                "Sofia Patel",
                "Eli Jones",
                "Ravi Singh",
                "Nora Stone",
                "Mina Park",
                "Owen Hall",
                "Zoe King",
                "Leo Cruz",
                "Aya Mori",
                "Finn Cole",
                "Uma Roy",
            ],
            "diagnosis_code": [
                "R69",
                "E11.9",
                "I10",
                "G rare",
                "J45.9",
                "M54.5",
                "N39.0",
                "K21.9",
                "L20.9",
                "D50.9",
                "G43.9",
                "H25.1",
                "C rare",
                "B20",
                "A09",
                "Z00.0",
            ],
            "loinc": [
                "LP21258-6",
                "2345-7",
                "718-7",
                "rare-LOINC-123",
                "4548-4",
                "8310-5",
                "2951-2",
                "2160-0",
                "6690-2",
                "26499-4",
                "3094-0",
                "33747-0",
                "rare-LOINC-456",
                "1751-7",
                "2823-3",
                "1975-2",
            ],
            "temperature": [
                36.8,
                37.0,
                36.5,
                98.6,
                37.1,
                36.9,
                37.2,
                36.7,
                36.6,
                37.0,
                36.8,
                36.9,
                37.1,
                36.8,
                36.7,
                36.9,
            ],
            "dose": [
                "5 mg",
                "10 mg",
                "2.5 mg",
                "5000 mcg",
                "1 mg",
                "20 mg",
                "5 mg",
                "2 mg",
                "8 mg",
                "4 mg",
                "3 mg",
                "6 mg",
                "7 mg",
                "9 mg",
                "1 mg",
                "2 mg",
            ],
            "event_date": ["2026-01-15"] * 16,
            "dob": ["1980-01-01"] * 16,
            "notes": ["routine"] * 16,
            "phone": ["555-0101"] * 16,
            "language": ["en"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "healthcare",
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
            "standards": ["FHIR", "ICD-10", "LOINC"],
            "temperature_unit": "C",
            "protected_dob_policy": "preserve",
        },
        protected_columns=("mrn", "dob"),
    )
    builder.inject(
        "hc-04", "diagnosis_code", "G rare", Disposition.PRESERVE, family="rare-icd-valid"
    )
    builder.inject(
        "hc-04", "loinc", "rare-LOINC-123", Disposition.PRESERVE, family="rare-loinc-valid"
    )
    builder.inject(
        "hc-04", "temperature", 98.6, Disposition.REVIEW, family="celsius-fahrenheit-conflict"
    )
    builder.inject("hc-01", "dose", "5 mg", Disposition.PRESERVE, family="dose-unit-valid")
    builder.inject(
        "hc-04",
        "dose",
        "5000 mcg",
        Disposition.REPAIR,
        expected=5.0,
        expected_dtype="float64",
        family="mg-mcg-unit-conversion",
    )
    builder.inject("hc-05", "event_date", "2025-01", Disposition.REVIEW, family="partial-date")
    builder.inject(
        "hc-06",
        "event_date",
        "2025-01-15T12:00:00Z",
        Disposition.REPAIR,
        expected="2025-01-15",
        family="fhir-date",
    )
    builder.inject("hc-07", "event_date", "2025-02-30", Disposition.FLAG, family="impossible-date")
    builder.inject(
        "hc-08",
        "patient_name",
        "Jose\u0301",
        Disposition.REPAIR,
        expected="José",
        family="decomposed-unicode",
    )
    builder.inject(
        "hc-09", "notes", "TB-HC-PHI-0001", Disposition.FLAG, family="phi-in-notes", sensitive=True
    )
    builder.inject(
        "hc-10",
        "phone",
        "555-0107",
        Disposition.PRESERVE,
        family="synthetic-phone",
        sensitive=True,
    )
    builder.inject(
        "hc-11", "notes", "TB-HC-MRN-NOTE", Disposition.FLAG, family="mrn-in-notes", sensitive=True
    )
    builder.inject(
        "hc-12", "notes", "555-0112", Disposition.FLAG, family="phone-in-notes", sensitive=True
    )
    builder.inject(
        "hc-16",
        "mrn",
        "TB-HC-MRN-TAIL",
        Disposition.REVIEW,
        family="mrn-tail-canary",
        sensitive=True,
    )
    builder.inject(
        "hc-15", "dob", "01/01/1980", Disposition.REVIEW, family="protected-dob-repair-conflict"
    )
    builder.add_row_case("exact-duplicate-hc-02-hc-03", Disposition.FLAG, family="exact-duplicate")
    builder.add_row_case("removed-hc-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case(
        "type-drifted-temperature", Disposition.REVIEW, family="type-drifted-column"
    )
    return builder.build()
