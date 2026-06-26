"""freshdata energy (SCADA / Modbus telemetry) domain pack.

Validates point-level telemetry — one row per ``(asset, register, timestamp)`` reading —
against common Modbus/SCADA conventions: Modbus function codes and the 16-bit register
address range, OPC/SCADA point quality, engineering units, and non-future timestamps.
Asset identifiers are treated as IDs and never imputed. The reference data ships with a
``_meta`` block documenting that these are common public conventions, not exhaustive or
vendor-specific specifications.

The validator is **stateless per frame**, so it composes with micro-batch / streaming
cleaning: ``fd.clean(batch, domain="energy")`` can be applied to each batch independently.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from .._common import check_iso_datetime, check_not_future, check_numeric
from ..base import ColumnMapping, ConfigDrivenValidator, Rule, RuleResult

_PACK_DIR = Path(__file__).resolve().parent


@cache
def _ref(name: str) -> dict[str, Any]:
    """Load and cache a bundled reference JSON (``reference/<name>.json``)."""
    with (_PACK_DIR / "reference" / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


class EnergyValidator(ConfigDrivenValidator):
    """SCADA / Modbus telemetry validator (one row per register reading)."""

    domain_name = "energy"
    version = "0.1.0"
    schema_version = "energy/2025.06"

    canonical_fields = (
        "timestamp",
        "asset_id",
        "register_address",
        "function_code",
        "value",
        "quality",
        "unit",
        "object_type",
    )
    required_fields = ("timestamp", "asset_id")
    id_fields = ("asset_id",)

    aliases = {
        "timestamp": (r"ts", r"time", r"timestamp", r"datetime", r"scan_?time", r"sample_?time"),
        "asset_id": (r"asset_?id", r"device_?id", r"tag(_?name)?", r"point_?id", r"rtu_?id",
                     r"plc_?id", r"unit_?id"),
        "register_address": (r"register(_?address)?", r"reg(_?addr(ess)?)?", r"address",
                             r"modbus_?addr(ess)?", r"data_?address"),
        "function_code": (r"function_?code", r"fc", r"func_?code", r"modbus_?function"),
        "value": (r"value", r"reading", r"measurement", r"val", r"raw_?value"),
        "quality": (r"quality", r"qual", r"qc", r"point_?quality", r"opc_?quality"),
        "unit": (r"unit", r"uom", r"units", r"eng_?unit", r"engineering_?unit"),
        "object_type": (r"object_?type", r"register_?type", r"reg_?type", r"modbus_?object"),
    }

    rules_path = str(_PACK_DIR / "rules.yaml")

    # -- extension registration ---------------------------------------------

    def register_extensions(self) -> None:
        self.register_check("iso_datetime", check_iso_datetime)
        self.register_check("not_future", check_not_future)
        self.register_check("numeric", check_numeric)
        self.register_check("valid_function_code", self._check_function_code)
        self.register_check("unique_reading", self._check_unique_reading)
        self.register_check("function_object_consistency", self._check_function_object)
        self.register_check("low_quality_value", self._check_low_quality_value)
        self.register_repair("coerce_quality", self._coerce_quality)

    def load_reference_values(self, name: str) -> list[Any]:
        return list(_ref(name)["codes"])

    def reference_sources(self) -> list[dict[str, Any]]:
        return [dict(_ref(name)["_meta"])
                for name in ("quality_codes", "modbus_function_codes", "units_common")]

    # -- custom checks ------------------------------------------------------

    def _check_function_code(self, df: pd.DataFrame, mapping: ColumnMapping,
                             rule: Rule) -> list[Any]:
        """Flag present function codes outside the documented public set."""
        col = mapping.actual(rule.fields[0])
        if col is None:
            return []
        allowed = set(_ref("modbus_function_codes")["codes"])
        codes = pd.to_numeric(df[col], errors="coerce")
        present = df[col].notna()
        return list(df.index[present & ~codes.isin(allowed)])

    def _check_unique_reading(self, df: pd.DataFrame, mapping: ColumnMapping,
                              rule: Rule) -> list[Any]:
        """Flag duplicate (asset, register, timestamp) readings (audit, never dropped)."""
        cols = [mapping.actual(f) for f in rule.fields]
        if any(c is None for c in cols):
            return []
        keyed = df[cols]
        dup = keyed.duplicated(keep="first") & keyed.notna().all(axis=1)
        return list(df.index[dup])

    def _check_function_object(self, df: pd.DataFrame, mapping: ColumnMapping,
                               rule: Rule) -> list[Any]:
        """Flag rows whose declared object/register type disagrees with the function code."""
        fc_col = mapping.actual("function_code")
        obj_col = mapping.actual("object_type")
        if fc_col is None or obj_col is None:
            return []
        klass = {int(k): v for k, v in _ref("modbus_function_codes")["register_class"].items()}
        codes = pd.to_numeric(df[fc_col], errors="coerce")
        declared = df[obj_col].astype("string").str.strip().str.casefold()
        bad: list[Any] = []
        for idx in df.index:
            code, obj = codes.at[idx], declared.at[idx]
            if pd.isna(code) or pd.isna(obj):
                continue
            expected = klass.get(int(code))
            if expected is not None and obj != expected:
                bad.append(idx)
        return bad

    def _check_low_quality_value(self, df: pd.DataFrame, mapping: ColumnMapping,
                                 rule: Rule) -> list[Any]:
        """Flag readings whose value is present but whose quality is bad/stale/null."""
        q_col = mapping.actual("quality")
        v_col = mapping.actual("value")
        if q_col is None or v_col is None:
            return []
        quality = df[q_col].astype("string").str.strip().str.casefold()
        suspect = quality.isin({"bad", "stale", "null"})
        return list(df.index[suspect & df[v_col].notna()])

    # -- repairs ------------------------------------------------------------

    def _coerce_quality(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule,
                        result: RuleResult) -> dict[Any, Any]:
        """Coerce known quality synonyms (e.g. ``ok`` -> ``good``) to canonical codes."""
        col = mapping.actual(rule.fields[0])
        if col is None:
            return {}
        ref = _ref(str(rule.repair_params["reference"]))
        canonical = {str(c).casefold(): c for c in ref["codes"]}
        synonyms = {str(k).casefold(): v for k, v in ref.get("coerce", {}).items()}
        fixes: dict[Any, Any] = {}
        for idx in result.violation_rows:
            raw = df.at[idx, col]
            if pd.isna(raw):
                continue
            key = str(raw).strip().casefold()
            target = canonical.get(key) or synonyms.get(key)
            if target is not None:
                fixes[idx] = target
        return fixes
