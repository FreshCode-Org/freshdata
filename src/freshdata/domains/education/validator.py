"""The education domain pack: K-12 student information system (Ed-Fi) hygiene.

Validates student enrollment, assessment, and staff frames aligned to the Ed-Fi Data
Standard. Standard checks (presence, reference) come from
:class:`~freshdata.domains.base.ConfigDrivenValidator`; the Ed-Fi-specific checks
(school-year validity, enrollment window) live here. Repairs are flag-only and
student/staff identifiers are never imputed; the audit trail masks PHI by default.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..._numeric import safe_to_numeric
from .._common import (
    check_both_present,
    check_ge_date,
    check_iso_date,
    check_nonneg,
    check_numeric,
    check_requires_field,
    redact_phi_actions,
    to_datetime_safe,
)
from ..base import ColumnMapping, ConfigDrivenValidator, RepairLog, Rule, ValidationReport

_PACK_DIR = Path(__file__).resolve().parent
_MIN_YEAR = 1900
_MAX_YEAR_OFFSET = 2
_PLAUSIBLE_YEARS = range(1900, 2101)


@cache
def _ref(name: str) -> dict[str, Any]:
    with open(_PACK_DIR / "reference" / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


class EducationValidator(ConfigDrivenValidator):
    """Validator for Ed-Fi-aligned K-12 student information system frames."""

    domain_name = "education"
    version = "0.1.0"
    schema_version = "ed-fi-2024"

    #: Identifiers redacted as ``[PHI]`` in the audit trail (unless audit_include_phi).
    PHI_FIELDS: tuple[str, ...] = ("student_unique_id", "staff_unique_id")

    canonical_fields = (
        "student_unique_id", "school_id", "school_year", "grade_level",
        "enrollment_date", "exit_withdrawal_date", "exit_withdrawal_type",
        "assessment_title", "assessment_score", "assessment_score_type",
        "performance_level", "staff_unique_id", "staff_classification",
    )
    required_fields = ("student_unique_id", "school_id", "school_year")
    id_fields = ("student_unique_id", "staff_unique_id")
    aliases = {
        "student_unique_id": (r"student_?unique_?id", r"student_?id", r"stu_?id"),
        "school_id": (r"school_?id", r"school_?code"),
        "school_year": (r"school_?year", r"academic_?year", r"sy"),
        "grade_level": (r"grade_?level", r"grade", r"grade_?level_?descriptor"),
        "enrollment_date": (r"enrollment_?date", r"entry_?date", r"enroll_?date"),
        "exit_withdrawal_date": (r"exit_?withdrawal_?date", r"exit_?date", r"withdrawal_?date"),
        "exit_withdrawal_type": (r"exit_?withdrawal_?type", r"exit_?type", r"withdrawal_?type"),
        "assessment_title": (r"assessment_?title", r"assessment_?name", r"test_?name"),
        "assessment_score": (r"assessment_?score", r"score", r"test_?score"),
        "assessment_score_type": (
            r"assessment_?score_?type", r"score_?type", r"reporting_?method",
        ),
        "performance_level": (r"performance_?level", r"proficiency", r"perf_?level"),
        "staff_unique_id": (r"staff_?unique_?id", r"staff_?id", r"teacher_?id"),
        "staff_classification": (r"staff_?classification", r"staff_?type", r"staff_?role"),
    }
    rules_path = str(_PACK_DIR / "rules.yaml")

    def __init__(
        self,
        *,
        column_map: Any = None,
        audit_include_phi: bool = False,
        **_kwargs: Any,
    ) -> None:
        self._audit_include_phi = bool(audit_include_phi)
        super().__init__(column_map=column_map)

    def register_extensions(self) -> None:
        self.register_check("school_year_valid", self._check_school_year)
        self.register_check("iso8601_date", check_iso_date)
        self.register_check("ge_date", check_ge_date)
        self.register_check("enrollment_in_year", self._check_enrollment_in_year)
        self.register_check("requires_field", check_requires_field)
        self.register_check("both_present", check_both_present)
        self.register_check("numeric", check_numeric)
        self.register_check("nonneg_numeric", check_nonneg)

    def load_reference_values(self, name: str) -> Any:
        if name in ("grade_levels", "exit_withdrawal_codes", "assessment_categories"):
            return _ref(name)["codes"]
        return super().load_reference_values(name)

    def reference_sources(self) -> list[dict[str, Any]]:
        names = ("grade_levels", "exit_withdrawal_codes", "assessment_categories",
                 "enrollment_type_codes", "school_year_format")
        return [{"name": name, **_ref(name)["_meta"]} for name in names]

    def repair(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> tuple[pd.DataFrame, RepairLog]:
        out, log = super().repair(df, report)
        redact_phi_actions(df, log, report.mapping, self.PHI_FIELDS, self._audit_include_phi)
        return out, log

    # -- custom checks ------------------------------------------------------

    def _check_school_year(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        series = df[mapping.actual("school_year")]
        present = series.notna()
        numeric = safe_to_numeric(series, errors="coerce")
        max_year = pd.Timestamp.now().year + _MAX_YEAR_OFFSET
        valid = (
            numeric.notna()
            & (numeric == numeric.round())
            & (numeric >= _MIN_YEAR)
            & (numeric <= max_year)
        )
        return df.index[present & ~valid].tolist()

    def _check_enrollment_in_year(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        enroll = to_datetime_safe(df[mapping.actual("enrollment_date")])
        year = safe_to_numeric(df[mapping.actual("school_year")], errors="coerce")
        rows: list[Any] = []
        for idx in df.index[enroll.notna() & year.notna()]:
            school_year = int(year.at[idx])
            if school_year not in _PLAUSIBLE_YEARS:
                continue  # an implausible year is ED-003's finding, not this one
            window_start = pd.Timestamp(year=school_year - 1, month=7, day=1)
            window_end = pd.Timestamp(year=school_year, month=6, day=30)
            if not window_start <= enroll.at[idx].normalize() <= window_end:
                rows.append(idx)
        return rows
