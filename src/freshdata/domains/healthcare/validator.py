"""The healthcare domain pack: FHIR/US Core clinical dataset hygiene.

Validates FHIR-sourced clinical data flattened into tabular form — the shape clinical
data engineers work with after extracting from an EHR or FHIR server. One validator
instance handles one resource (``Patient``, ``Observation``, or ``Encounter``): the
resource is taken from ``fhir_resource=`` or, when omitted, auto-detected from the column
signature. Each resource has its own canonical fields and ``rules/<resource>.yaml``.

Privacy: ``patient_id``, ``birth_date``, and ``address_postal_code`` are PHI — the audit
trail masks them as ``[PHI]`` unless the caller passes ``audit_include_phi=True``.
"""

from __future__ import annotations

import json
import re
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .._common import (
    check_at_least_one,
    check_fhir_date,
    check_ge_date,
    check_iso_datetime,
    check_nonneg_number,
    check_not_future,
    check_partial_date,
    check_requires_field,
    redact_phi_actions,
    to_datetime_safe,
)
from ..base import (
    ColumnMapping,
    ConfigDrivenValidator,
    DomainError,
    RepairLog,
    Rule,
    RuleResult,
    ValidationReport,
)
from ..reference import load_reference

_PACK_DIR = Path(__file__).resolve().parent
_BUNDLED_DIR = _PACK_DIR.parent / "bundled"

SUPPORTED_RESOURCES: tuple[str, ...] = (
    "Patient", "Observation", "Encounter", "Condition", "MedicationRequest",
)
_MAX_AGE_YEARS = 150
_MAX_ENCOUNTER_DAYS = 365
_DAYS_PER_YEAR = 365.25

# Distinctive columns used to auto-detect a resource from a flattened frame.
_PATIENT_SIGNAL = ("birth_date", "gender", "marital_status", "deceased", "deceased_date",
                   "address_postal_code")
_OBSERVATION_SIGNAL = ("observation_id", "code_value", "code_system", "value_quantity",
                       "value_string", "interpretation")
_ENCOUNTER_SIGNAL = ("encounter_id", "class_code", "period_start", "period_end", "service_type")
_CONDITION_SIGNAL = ("condition_id", "clinical_status", "verification_status", "onset_date",
                     "recorded_date")
_MEDICATIONREQUEST_SIGNAL = ("medication_request_id", "intent", "medication_code",
                             "medication_display", "authored_on")


@cache
def _ref(name: str) -> dict[str, Any]:
    with open(_PACK_DIR / "reference" / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _iso3166_alpha2() -> list[str]:
    with open(_BUNDLED_DIR / "iso3166.json", encoding="utf-8") as handle:
        data = json.load(handle)
    return sorted(code for code in data if code != "_meta")


def _gender_codes() -> list[str]:
    return list(_ref("fhir_gender_codes")["codes"])


class UnsupportedFHIRResourceError(DomainError):
    """Raised when a requested ``fhir_resource`` is outside this pack's coverage."""

    def __init__(self, requested: Any, supported: list[str]) -> None:
        self.requested = requested
        self.supported = list(supported)
        super().__init__(
            f"unsupported FHIR resource {requested!r}; supported resources: "
            f"{', '.join(supported)}"
        )


class AmbiguousFHIRResourceError(DomainError):
    """Raised when the FHIR resource cannot be confidently detected from columns."""

    def __init__(self, candidates: list[str], reason: str = "") -> None:
        self.candidates = list(candidates)
        listed = ", ".join(self.candidates) if self.candidates else "(none)"
        message = f"could not determine the FHIR resource from columns; candidates: {listed}"
        if reason:
            message += f" ({reason})"
        super().__init__(message + ". Pass fhir_resource= to disambiguate.")


def _field_present(columns: list[Any], aliases: dict[str, Any], canonical: str) -> bool:
    """True if *canonical* maps to a real column (exact, case-insensitive, or alias)."""
    lowered = {str(c).casefold() for c in columns}
    if canonical in columns or canonical.casefold() in lowered:
        return True
    return any(
        re.fullmatch(pattern, str(column), flags=re.IGNORECASE)
        for pattern in aliases.get(canonical, ())
        for column in columns
    )


# Per-resource field configuration. Aliases are regex, matched case-insensitively.
_RESOURCE_CONFIG: dict[str, dict[str, Any]] = {
    "Patient": {
        "schema_version": "fhir-r4-patient",
        "rules_file": "patient.yaml",
        "canonical_fields": (
            "patient_id", "birth_date", "gender", "deceased", "deceased_date",
            "marital_status", "address_postal_code", "address_country",
        ),
        "required_fields": ("patient_id", "birth_date", "gender"),
        "id_fields": ("patient_id",),
        "aliases": {
            "patient_id": (r"patient_?id", r"pat_?id", r"subject_?id", r"mrn"),
            "birth_date": (r"birth_?date", r"birthdate", r"dob", r"date_?of_?birth"),
            "gender": (r"gender", r"sex", r"administrative_?gender"),
            "deceased": (r"deceased", r"deceased_?boolean", r"is_?deceased"),
            "deceased_date": (r"deceased_?date", r"deceased_?datetime", r"death_?date",
                              r"date_?of_?death"),
            "marital_status": (r"marital_?status", r"marital"),
            "address_postal_code": (r"address_?postal_?code", r"postal_?code", r"zip",
                                    r"zip_?code", r"postcode"),
            "address_country": (r"address_?country", r"country"),
        },
    },
    "Observation": {
        "schema_version": "fhir-r4-observation",
        "rules_file": "observation.yaml",
        "canonical_fields": (
            "observation_id", "patient_id", "status", "code_system", "code_value",
            "display", "effective_date", "value_quantity", "value_unit", "value_string",
            "interpretation",
        ),
        "required_fields": ("observation_id", "patient_id", "status", "code_value"),
        "id_fields": ("observation_id", "patient_id"),
        "aliases": {
            "observation_id": (r"observation_?id", r"obs_?id"),
            "patient_id": (r"patient_?id", r"subject_?id", r"subject_?reference"),
            "status": (r"status", r"observation_?status"),
            "code_system": (r"code_?system", r"system", r"code_?coding_?system"),
            "code_value": (r"code_?value", r"code", r"loinc_?code", r"observation_?code"),
            "display": (r"display", r"code_?display", r"label"),
            "effective_date": (r"effective_?date", r"effective_?datetime", r"effective",
                               r"observation_?date"),
            "value_quantity": (r"value_?quantity", r"value_?number", r"numeric_?value"),
            "value_unit": (r"value_?unit", r"unit"),
            "value_string": (r"value_?string", r"value_?text", r"text_?value"),
            "interpretation": (r"interpretation", r"interpretation_?code", r"abnormal_?flag"),
        },
    },
    "Encounter": {
        "schema_version": "fhir-r4-encounter",
        "rules_file": "encounter.yaml",
        "canonical_fields": (
            "encounter_id", "patient_id", "status", "class_code", "period_start",
            "period_end", "reason_code", "reason_code_system",
            "hospitalization_admit_source", "service_type",
        ),
        "required_fields": ("encounter_id", "patient_id", "status"),
        "id_fields": ("encounter_id", "patient_id"),
        "aliases": {
            "encounter_id": (r"encounter_?id", r"enc_?id", r"visit_?id"),
            "patient_id": (r"patient_?id", r"subject_?id", r"subject_?reference"),
            "status": (r"status", r"encounter_?status"),
            "class_code": (r"class_?code", r"class", r"encounter_?class"),
            "period_start": (r"period_?start", r"start_?date", r"admit_?date", r"start"),
            "period_end": (r"period_?end", r"end_?date", r"discharge_?date", r"end"),
            "reason_code": (r"reason_?code", r"reason"),
            "reason_code_system": (r"reason_?code_?system", r"reason_?system"),
            "hospitalization_admit_source": (r"hospitalization_?admit_?source", r"admit_?source"),
            "service_type": (r"service_?type", r"service"),
        },
    },
    "Condition": {
        "schema_version": "fhir-r4-condition",
        "rules_file": "condition.yaml",
        "canonical_fields": (
            "condition_id", "patient_id", "clinical_status", "verification_status",
            "category", "code_system", "code_value", "display", "onset_date", "recorded_date",
        ),
        "required_fields": ("condition_id", "patient_id", "code_value"),
        "id_fields": ("condition_id", "patient_id"),
        "aliases": {
            "condition_id": (r"condition_?id", r"problem_?id", r"diagnosis_?id"),
            "patient_id": (r"patient_?id", r"subject_?id", r"subject_?reference"),
            "clinical_status": (r"clinical_?status", r"clinicalstatus"),
            "verification_status": (r"verification_?status", r"verificationstatus"),
            "category": (r"category",),
            "code_system": (r"code_?system", r"system"),
            "code_value": (r"code_?value", r"code", r"diagnosis_?code", r"icd_?10"),
            "display": (r"display", r"description", r"text"),
            "onset_date": (r"onset_?date", r"onset", r"onset_?datetime"),
            "recorded_date": (r"recorded_?date", r"recorded", r"asserted_?date"),
        },
    },
    "MedicationRequest": {
        "schema_version": "fhir-r4-medicationrequest",
        "rules_file": "medication_request.yaml",
        "canonical_fields": (
            "medication_request_id", "patient_id", "status", "intent",
            "medication_system", "medication_code", "medication_display",
            "authored_on", "requester",
        ),
        "required_fields": ("medication_request_id", "patient_id", "status", "intent"),
        "id_fields": ("medication_request_id", "patient_id"),
        "aliases": {
            "medication_request_id": (r"medication_?request_?id", r"med_?request_?id",
                                      r"prescription_?id", r"rx_?id"),
            "patient_id": (r"patient_?id", r"subject_?id", r"subject_?reference"),
            "status": (r"status", r"request_?status"),
            "intent": (r"intent",),
            "medication_system": (r"medication_?system", r"med_?system"),
            "medication_code": (r"medication_?code", r"med_?code", r"rxnorm", r"drug_?code"),
            "medication_display": (r"medication_?display", r"medication", r"drug_?name"),
            "authored_on": (r"authored_?on", r"authored", r"prescribed_?date", r"written_?date"),
            "requester": (r"requester", r"prescriber", r"provider_?id"),
        },
    },
}


class HealthcareValidator(ConfigDrivenValidator):
    """Validator for flattened FHIR Patient / Observation / Encounter frames."""

    domain_name = "healthcare"
    version = "0.1.0"
    schema_version = "fhir-r4"

    #: PHI columns masked as ``[PHI]`` in the audit trail unless audit_include_phi.
    PHI_FIELDS: tuple[str, ...] = ("patient_id", "birth_date", "address_postal_code")

    def __init__(
        self,
        *,
        column_map: Any = None,
        fhir_resource: str | None = None,
        audit_include_phi: bool = False,
        **_kwargs: Any,
    ) -> None:
        self._audit_include_phi = bool(audit_include_phi)
        self._resource: str | None = None
        super().__init__(column_map=column_map)
        if fhir_resource is not None:
            self._activate_resource(self._normalize_resource(fhir_resource))

    # -- resource resolution -----------------------------------------------

    @property
    def fhir_resource(self) -> str | None:
        return self._resource

    def _normalize_resource(self, value: Any) -> str:
        for resource in SUPPORTED_RESOURCES:
            if str(value).strip().casefold() == resource.casefold():
                return resource
        raise UnsupportedFHIRResourceError(value, list(SUPPORTED_RESOURCES))

    def _activate_resource(self, resource: str) -> None:
        config = _RESOURCE_CONFIG[resource]
        self._resource = resource
        self.canonical_fields = config["canonical_fields"]
        self.required_fields = config["required_fields"]
        self.id_fields = config["id_fields"]
        self.aliases = config["aliases"]
        self.schema_version = config["schema_version"]
        self.rules_path = str(_PACK_DIR / "rules" / config["rules_file"])
        self._rules = None  # reload rules for the active resource

    def _detect_resource(self, df: pd.DataFrame) -> str:
        columns = list(df.columns)

        def has_signal(resource: str, fields: tuple[str, ...]) -> bool:
            aliases = _RESOURCE_CONFIG[resource]["aliases"]
            return any(_field_present(columns, aliases, field) for field in fields)

        is_obs = has_signal("Observation", _OBSERVATION_SIGNAL)
        is_enc = has_signal("Encounter", _ENCOUNTER_SIGNAL)
        is_pat = has_signal("Patient", _PATIENT_SIGNAL)
        is_cond = has_signal("Condition", _CONDITION_SIGNAL)
        is_med = has_signal("MedicationRequest", _MEDICATIONREQUEST_SIGNAL)
        # Condition / MedicationRequest carry distinctive id/status columns that
        # Observation lacks; their shared code_* columns must not also count them as
        # Observations (or Patients via patient_id).
        distinct = is_cond or is_med

        candidates: list[str] = []
        if is_pat and not is_obs and not is_enc and not distinct:
            candidates.append("Patient")
        if is_obs and not distinct:
            candidates.append("Observation")
        if is_enc and not distinct:
            candidates.append("Encounter")
        if is_cond:
            candidates.append("Condition")
        if is_med:
            candidates.append("MedicationRequest")
        if len(candidates) == 1:
            return candidates[0]
        reason = "multiple resource signatures" if candidates else "no resource-specific columns"
        raise AmbiguousFHIRResourceError(candidates or list(SUPPORTED_RESOURCES), reason=reason)

    # -- engine hooks -------------------------------------------------------

    def detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        if self._resource is None:
            self._activate_resource(self._detect_resource(df))
        return super().detect_columns(df)

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """Validate a flattened FHIR frame for the active (or detected) resource.

        The ``domain_trust_score`` uses the shared severity weighting (error=1.0,
        warning=0.25, info=0.05). Because every schema and reference rule here is
        error-severity while the semantic checks (age range, encounter duration) are
        warning/info-severity, schema and reference violations dominate the score and
        semantic findings move it only slightly — the weighting clinical data-quality
        practice expects, where a missing identifier or an invalid coded value matters
        far more than an unusual-but-possible age or a long admission.
        """
        return super().validate(df)

    def register_extensions(self) -> None:
        self.register_check("fhir_date", check_fhir_date)
        self.register_check("partial_date", check_partial_date)
        self.register_check("not_future_date", check_not_future)
        self.register_check("iso8601_datetime", check_iso_datetime)
        self.register_check("ge_date", check_ge_date)
        self.register_check("at_least_one", check_at_least_one)
        self.register_check("requires_field", check_requires_field)
        self.register_check("nonneg_number", check_nonneg_number)
        self.register_check("gender_canonical", self._check_gender)
        self.register_check("deceased_after_birth", self._check_deceased_after_birth)
        self.register_check("age_range", self._check_age_range)
        self.register_check("loinc_when_loinc", self._check_loinc)
        self.register_check("not_future_when_finished", self._check_not_future_finished)
        self.register_check("duration_lt_when_finished", self._check_duration)
        self.register_check("icd10_when_icd10", self._check_icd10)
        self.register_check("ucum_when_present", self._check_ucum)
        self.register_repair("coerce_gender_case", self._repair_gender_case)

    def load_reference_values(self, name: str) -> Any:
        if name in ("fhir_obs_status", "fhir_enc_status", "fhir_gender_codes",
                    "condition_clinical_status", "medicationrequest_status",
                    "medicationrequest_intent", "icd10_common"):
            return _ref(name)["codes"]
        if name == "iso3166":
            return _iso3166_alpha2()
        return super().load_reference_values(name)

    def reference_sources(self) -> list[dict[str, Any]]:
        names = ("fhir_gender_codes", "fhir_obs_status", "fhir_enc_status",
                 "loinc_common", "snomed_common", "icd10_chapters")
        sources = [{"name": name, **_ref(name)["_meta"]} for name in names]
        with open(_BUNDLED_DIR / "iso3166.json", encoding="utf-8") as handle:
            sources.append({"name": "iso3166", **json.load(handle).get("_meta", {})})
        return sources

    def describe(self) -> dict[str, Any]:
        if self._resource is None:
            return {
                "domain": self.domain_name,
                "version": self.version,
                "fhir_resource": None,
                "supported_resources": list(SUPPORTED_RESOURCES),
                "note": "resource is auto-detected per frame at validate() time",
            }
        description = super().describe()
        description["fhir_resource"] = self._resource
        return description

    def repair(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> tuple[pd.DataFrame, RepairLog]:
        out, log = super().repair(df, report)
        redact_phi_actions(df, log, report.mapping, self.PHI_FIELDS, self._audit_include_phi)
        return out, log

    # -- custom checks ------------------------------------------------------

    def _check_gender(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        series = df[mapping.actual("gender")]
        present = series.notna()
        allowed = set(_gender_codes())
        bad = present & ~series.isin(allowed)  # exact match; case fixes are repaired
        return df.index[bad].tolist()

    def _check_deceased_after_birth(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        deceased_date = to_datetime_safe(df[mapping.actual("deceased_date")])
        birth_date = to_datetime_safe(df[mapping.actual("birth_date")])
        deceased_col = mapping.actual("deceased")
        truthy = (
            self._truthy(df[deceased_col]) if deceased_col is not None
            else pd.Series(True, index=df.index)
        )
        both = deceased_date.notna() & birth_date.notna() & truthy
        return df.index[both & (deceased_date < birth_date)].tolist()

    def _check_age_range(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        birth_date = to_datetime_safe(df[mapping.actual("birth_date")])
        age_years = (pd.Timestamp.now() - birth_date).dt.days / _DAYS_PER_YEAR
        bad = birth_date.notna() & ((age_years < 0) | (age_years > _MAX_AGE_YEARS))
        return df.index[bad].tolist()

    def _check_loinc(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        code_value = mapping.actual("code_value")
        system_col = mapping.actual("code_system")
        if system_col is None:
            return []  # without a coding system we cannot assume LOINC
        is_loinc = df[system_col].astype("string").str.strip() == "http://loinc.org"
        loinc_codes = set(_ref("loinc_common")["codes"])
        values = df[code_value].astype("string").str.strip()
        bad = is_loinc.fillna(False) & values.notna() & ~values.isin(loinc_codes)
        return df.index[bad].tolist()

    def _check_icd10(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        """Flag ICD-10-coded values outside the documented common subset (system-gated)."""
        code_value = mapping.actual("code_value")
        system_col = mapping.actual("code_system")
        if code_value is None or system_col is None:
            return []
        systems = df[system_col].astype("string").str.strip().str.lower()
        is_icd10 = (systems.str.contains("icd-10", na=False)
                    | systems.str.contains("icd10", na=False))
        icd10_codes = set(_ref("icd10_common")["codes"])
        values = df[code_value].astype("string").str.strip()
        bad = is_icd10.fillna(False) & values.notna() & ~values.isin(icd10_codes)
        return df.index[bad].tolist()

    def _check_ucum(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        """Flag observation units outside the UCUM common set (via the reference layer)."""
        unit_col = mapping.actual("value_unit")
        if unit_col is None:
            return []
        ref = load_reference("ucum_common", normalizer="exact")  # UCUM is case-sensitive
        return df.index[ref.invalid_mask(df[unit_col])].tolist()

    def _check_not_future_finished(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        finished = self._is_finished(df, mapping)
        today = pd.Timestamp.now(tz="UTC").normalize()
        bad = pd.Series(False, index=df.index)
        for field in ("period_start", "period_end"):
            col = mapping.actual(field)
            if col is None:
                continue
            parsed = to_datetime_safe(df[col], utc=True)
            bad = bad | (finished & parsed.notna() & (parsed.dt.normalize() > today))
        return df.index[bad].tolist()

    def _check_duration(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        start = to_datetime_safe(df[mapping.actual("period_start")])
        end = to_datetime_safe(df[mapping.actual("period_end")])
        finished = self._is_finished(df, mapping)
        both = start.notna() & end.notna() & finished
        return df.index[both & ((end - start).dt.days >= _MAX_ENCOUNTER_DAYS)].tolist()

    # -- custom repair ------------------------------------------------------

    def _repair_gender_case(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule, result: RuleResult
    ) -> dict[Any, Any]:
        col = mapping.actual("gender")
        canonical = {code.casefold(): code for code in _gender_codes()}
        fixes: dict[Any, Any] = {}
        for row in result.violation_rows:
            if row not in df.index:
                continue
            value = df.at[row, col]
            if pd.isna(value):
                continue
            key = str(value).strip().casefold()
            if key in canonical:
                fixes[row] = canonical[key]
        return fixes

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _truthy(series: pd.Series) -> pd.Series:
        def one(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value == 1
            if isinstance(value, str):
                return value.strip().casefold() in ("true", "1", "yes", "y")
            return False

        return series.map(one)

    def _is_finished(self, df: pd.DataFrame, mapping: ColumnMapping) -> pd.Series:
        status_col = mapping.actual("status")
        if status_col is None:
            return pd.Series(False, index=df.index)
        return df[status_col].astype("string").str.casefold() == "finished"
