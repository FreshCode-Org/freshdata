"""Deterministic education gold fixture for TruthBench."""

from __future__ import annotations

import pandas as pd

from ..models import Disposition
from .base import FixtureBuilder, TruthFixture


def build(seed: int = 1729) -> TruthFixture:
    rows = [f"edu-{i:02d}" for i in range(1, 17)]
    frame = pd.DataFrame(
        {
            "student_id": [
                "000123",
                "000124",
                "000125",
                "000126",
                "000127",
                "000128",
                "000129",
                "000130",
                "000131",
                "000132",
                "000133",
                "000134",
                "000135",
                "000136",
                "000137",
                "000138",
            ],
            "grade_letter": [
                "A",
                "A-",
                "B+",
                "B",
                "C+",
                "C",
                "B-",
                "A",
                "A",
                "B",
                "C",
                "D",
                "A",
                "B+",
                "C",
                "A",
            ],
            "score_percent": [
                "95",
                "90",
                "87.5",
                "82",
                "78",
                "72",
                "68",
                "100",
                "0",
                "88",
                "76",
                "61",
                "93",
                "86",
                "74",
                "96",
            ],
            "gpa": [
                4.0,
                3.7,
                3.3,
                3.0,
                2.7,
                2.3,
                2.0,
                4.0,
                0.0,
                3.3,
                2.7,
                1.7,
                3.9,
                3.3,
                2.3,
                4.0,
            ],
            "school_year": ["2025-2026"] * 16,
            "enrollment_date": ["2025-08-15"] * 16,
            "completion_date": ["2026-05-30"] * 16,
            "guardian_email": ["guardian1@example.invalid"] * 16,
            "guardian_phone": ["555-0101"] * 16,
            "ferpa_notes": ["routine"] * 16,
            "grade_policy": ["A>=90;B>=80;C>=70"] * 16,
            "language": ["en"] * 16,
            "batch": [f"seed-{int(seed)}"] * 16,
        },
        index=rows,
    )
    builder = FixtureBuilder(
        "v1",
        "education",
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
            "grade_scales": ["letter", "percentage", "GPA"],
            "protected_grade_policy": "preserve",
        },
        protected_columns=("student_id", "grade_letter"),
    )
    builder.inject(
        "edu-01", "student_id", "000123", Disposition.PRESERVE, family="leading-zero-student-id"
    )
    builder.inject(
        "edu-02", "grade_letter", "A-", Disposition.PRESERVE, family="letter-grade-scale"
    )
    builder.inject("edu-03", "score_percent", 0, Disposition.PRESERVE, family="zero-score")
    builder.inject(
        "edu-04", "school_year", "2025/26", Disposition.REVIEW, family="school-year-ambiguity"
    )
    builder.inject(
        "edu-05",
        "enrollment_date",
        "2026-02-01",
        Disposition.REVIEW,
        family="enrollment-date-ordering",
    )
    builder.inject(
        "edu-06",
        "completion_date",
        "2025-01-01",
        Disposition.REVIEW,
        family="enrollment-date-ordering",
    )
    builder.inject(
        "edu-07",
        "score_percent",
        "95%",
        Disposition.REPAIR,
        expected=95.0,
        expected_dtype="float64",
        family="percentage-scale",
    )
    builder.inject("edu-08", "gpa", 4.0, Disposition.PRESERVE, family="gpa-scale")
    builder.inject(
        "edu-09",
        "guardian_email",
        "guardian@example.invalid",
        Disposition.FLAG,
        family="guardian-contact-pii",
        sensitive=True,
    )
    builder.inject(
        "edu-10",
        "guardian_phone",
        "555-0110",
        Disposition.FLAG,
        family="guardian-contact-pii",
        sensitive=True,
    )
    builder.inject(
        "edu-10",
        "ferpa_notes",
        "TB-EDU-FERPA-0001",
        Disposition.FLAG,
        family="ferpa-sensitive-notes",
        sensitive=True,
    )
    builder.inject(
        "edu-16", "grade_letter", "A", Disposition.REVIEW, family="protected-grade-policy-conflict"
    )
    builder.add_row_case(
        "exact-duplicate-edu-02-edu-03", Disposition.FLAG, family="exact-duplicate"
    )
    builder.add_row_case("removed-edu-15", Disposition.REVIEW, family="removed-row")
    builder.add_schema_case("added-column", Disposition.REVIEW, family="added-column")
    builder.add_schema_case("removed-column", Disposition.REVIEW, family="removed-column")
    builder.add_schema_case("renamed-column", Disposition.REVIEW, family="renamed-column")
    builder.add_schema_case("reordered-columns", Disposition.REVIEW, family="reordered-columns")
    builder.add_schema_case("type-drifted-score", Disposition.REVIEW, family="type-drifted-column")
    return builder.build()
