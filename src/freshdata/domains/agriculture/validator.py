"""The agriculture domain pack: ADAPT field-operation and crop dataset hygiene.

Validates field operations, crop production, soil sampling, and harvest frames aligned
to the ADAPT vocabulary. Standard checks (presence, reference, numeric range) come from
:class:`~freshdata.domains.base.ConfigDrivenValidator`; the ADAPT-specific checks and the
unit-coercion repairs live here. The only active repairs normalize informal area/yield
unit spellings to canonical UN/CEFACT codes; identifiers and crop codes are never
imputed or inferred.
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
    check_iso_date,
    check_not_future,
    check_positive,
    to_datetime_safe,
)
from ..base import ColumnMapping, ConfigDrivenValidator, Rule, RuleResult

_PACK_DIR = Path(__file__).resolve().parent
_MIN_YEAR = 1900
_MAX_YEAR_OFFSET = 1


@cache
def _ref(name: str) -> dict[str, Any]:
    with open(_PACK_DIR / "reference" / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


class AgricultureValidator(ConfigDrivenValidator):
    """Validator for ADAPT-aligned agricultural field and crop frames."""

    domain_name = "agriculture"
    version = "0.1.0"
    schema_version = "adapt-2024"

    canonical_fields = (
        "field_id", "operation_id", "operation_type", "operation_date",
        "crop_type", "crop_code", "area", "area_unit", "yield_value", "yield_unit",
        "soil_texture_class", "soil_ph", "soil_om_pct", "equipment_id",
        "operator_id", "season_year",
    )
    required_fields = ("field_id",)
    id_fields = ("field_id", "operation_id", "equipment_id", "operator_id")
    aliases = {
        "field_id": (r"field_?id", r"field_?identifier"),
        "operation_id": (r"operation_?id", r"op_?id"),
        "operation_type": (r"operation_?type", r"op_?type", r"activity_?type"),
        "operation_date": (r"operation_?date", r"op_?date", r"activity_?date", r"date"),
        "crop_type": (r"crop_?type", r"crop_?name", r"crop"),
        "crop_code": (r"crop_?code", r"fao_?code", r"crop_?id"),
        "area": (r"area", r"field_?area", r"size"),
        "area_unit": (r"area_?unit", r"area_?uom"),
        "yield_value": (r"yield_?value", r"yield", r"yield_?amount"),
        "yield_unit": (r"yield_?unit", r"yield_?uom"),
        "soil_texture_class": (r"soil_?texture_?class", r"soil_?texture", r"texture_?class"),
        "soil_ph": (r"soil_?ph", r"ph"),
        "soil_om_pct": (r"soil_?om_?pct", r"soil_?om", r"organic_?matter_?pct", r"om_?pct"),
        "equipment_id": (r"equipment_?id", r"machine_?id", r"implement_?id"),
        "operator_id": (r"operator_?id", r"driver_?id"),
        "season_year": (r"season_?year", r"crop_?year", r"harvest_?year"),
    }
    rules_path = str(_PACK_DIR / "rules.yaml")

    def __init__(self, *, column_map: Any = None, **_kwargs: Any) -> None:
        super().__init__(column_map=column_map)

    def register_extensions(self) -> None:
        self.register_check("iso8601_date", check_iso_date)
        self.register_check("not_future_date", check_not_future)
        self.register_check("positive", check_positive)
        self.register_check("both_present", check_both_present)
        self.register_check("yield_only_when_harvest", self._check_yield_only_harvest)
        self.register_check("season_year_valid", self._check_season_year)
        self.register_check("date_year_matches_season", self._check_date_year_matches_season)
        self.register_repair("coerce_unit", self._repair_coerce_unit)

    def load_reference_values(self, name: str) -> Any:
        if name in ("operation_types", "fao_crop_codes", "area_units", "yield_units",
                    "soil_texture_classes"):
            return _ref(name)["codes"]
        return super().load_reference_values(name)

    def reference_sources(self) -> list[dict[str, Any]]:
        names = ("operation_types", "fao_crop_codes", "area_units", "yield_units",
                 "soil_texture_classes")
        return [{"name": name, **_ref(name)["_meta"]} for name in names]

    # -- custom checks ------------------------------------------------------

    def _check_yield_only_harvest(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        present = df[mapping.actual("yield_value")].notna()
        op_col = mapping.actual("operation_type")
        if op_col is None:
            return df.index[present].tolist()
        is_harvest = df[op_col].astype("string").str.casefold() == "harvesting"
        return df.index[present & ~is_harvest.fillna(False)].tolist()

    def _check_season_year(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        series = df[mapping.actual("season_year")]
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

    def _check_date_year_matches_season(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        parsed = to_datetime_safe(df[mapping.actual("operation_date")])
        season = safe_to_numeric(df[mapping.actual("season_year")], errors="coerce")
        both = parsed.notna() & season.notna()
        bad = both & (parsed.dt.year != season)
        return df.index[bad].tolist()

    # -- custom repair ------------------------------------------------------

    def _repair_coerce_unit(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule, result: RuleResult
    ) -> dict[Any, Any]:
        ref = _ref(str(rule.repair_params["reference"]))
        codes = {str(code).casefold(): code for code in ref["codes"]}
        coerce_map = {str(k).casefold(): v for k, v in ref.get("coerce", {}).items()}
        col = mapping.actual(rule.fields[0])
        fixes: dict[Any, Any] = {}
        for row in result.violation_rows:
            if row not in df.index:
                continue
            value = df.at[row, col]
            if pd.isna(value):
                continue
            key = str(value).strip().casefold()
            if key in codes:
                fixes[row] = codes[key]       # case fix, e.g. "acr" -> "ACR"
            elif key in coerce_map:
                fixes[row] = coerce_map[key]  # informal spelling, e.g. "acres" -> "ACR"
        return fixes
