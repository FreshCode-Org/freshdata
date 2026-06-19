"""The retail domain pack: GS1 product-catalog and supply-chain hygiene.

Validates product catalogs against GS1 standards — GTIN/GLN check digits, ISO
3166-1 origin codes, UN/CEFACT units of measure, and GPC brick codes. Standard
checks (presence, regex, reference) come from
:class:`~freshdata.domains.base.ConfigDrivenValidator`; the GS1-specific checks
live here.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import ColumnMapping, ConfigDrivenValidator, Rule, RuleResult

_PACK_DIR = Path(__file__).resolve().parent
_BUNDLED_DIR = _PACK_DIR.parent / "bundled"
_GTIN_LENGTHS = (8, 12, 13, 14)
_NONDIGIT = re.compile(r"\D")


@lru_cache(maxsize=1)
def _iso3166() -> dict[str, Any]:
    with open(_BUNDLED_DIR / "iso3166.json", encoding="utf-8") as handle:
        data = json.load(handle)
    codes: set[str] = set()
    for code, value in data.items():
        if code == "_meta":
            continue
        codes.add(code)
        if isinstance(value, dict) and value.get("alpha-3"):
            codes.add(value["alpha-3"])
    return {"codes": sorted(codes), "_meta": data.get("_meta", {})}


@lru_cache(maxsize=1)
def _uom() -> dict[str, Any]:
    with open(_BUNDLED_DIR / "uom_codes.json", encoding="utf-8") as handle:
        return json.load(handle)


def _mod10_valid(code: str) -> bool:
    """True if *code* (all digits) ends in a valid GS1 mod-10 check digit."""
    body, check = code[:-1], int(code[-1])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


def _gtin_well_formed(text: str) -> bool:
    return text.isdigit() and len(text) in _GTIN_LENGTHS


class RetailValidator(ConfigDrivenValidator):
    """Validator for GS1-aligned product catalog frames."""

    domain_name = "retail"
    version = "0.1.0"
    schema_version = "2024-01"

    canonical_fields = (
        "gtin", "gln", "product_description", "brand_name", "net_content",
        "net_content_uom", "country_of_origin", "gpc_brick_code",
    )
    required_fields = ("gtin",)
    id_fields = ("gtin", "gln")
    aliases = {
        "gtin": (r"gtin", r"gtin_?\d{0,2}", r"upc", r"ean", r"barcode", r"item_?gtin"),
        "gln": (r"gln", r"global_?location_?number"),
        "product_description": (r"product_?description", r"description", r"prod_?desc",
                                r"item_?description"),
        "brand_name": (r"brand_?name", r"brand", r"manufacturer"),
        "net_content": (r"net_?content", r"content", r"net_?weight", r"quantity"),
        "net_content_uom": (r"net_?content_?uom", r"uom", r"unit", r"unit_?of_?measure",
                            r"content_?unit"),
        "country_of_origin": (r"country_?of_?origin", r"country", r"origin", r"made_?in",
                              r"coo"),
        "gpc_brick_code": (r"gpc_?brick_?code", r"gpc", r"brick_?code"),
    }
    rules_path = str(_PACK_DIR / "rules.yaml")

    def register_extensions(self) -> None:
        self.register_check("gtin_length", self._check_gtin_length)
        self.register_check("gtin_check_digit", self._check_gtin_check_digit)
        self.register_check("gln_check", self._check_gln_check)
        self.register_check("nonnull_maxlen", self._check_nonnull_maxlen)
        self.register_check("content_uom_consistency", self._check_content_uom)
        self.register_repair("strip_nondigits", self._repair_strip_nondigits)

    def load_reference_values(self, name: str):
        if name == "iso3166":
            return _iso3166()["codes"]
        if name == "uom":
            return _uom()["codes"]
        return super().load_reference_values(name)

    def reference_sources(self) -> list[dict[str, Any]]:
        return [
            {"name": "iso3166", **_iso3166()["_meta"]},
            {"name": "uom", **_uom()["_meta"]},
        ]

    # -- custom checks ------------------------------------------------------

    def _gtin_text(self, df: pd.DataFrame, mapping: ColumnMapping) -> tuple[pd.Series, pd.Series]:
        series = df[mapping.actual("gtin")]
        return series, series.astype("string").str.strip()

    def _check_gtin_length(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        series, text = self._gtin_text(df, mapping)
        present = series.notna()
        well_formed = text.map(lambda v: isinstance(v, str) and _gtin_well_formed(v))
        return df.index[present & ~well_formed.fillna(False)].tolist()

    def _check_gtin_check_digit(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        series, text = self._gtin_text(df, mapping)
        present = series.notna()
        # Only judge the check digit of well-formed GTINs; malformed ones are GS1-002's job.
        well_formed = text.map(lambda v: isinstance(v, str) and _gtin_well_formed(v))
        bad = text.map(
            lambda v: isinstance(v, str) and _gtin_well_formed(v) and not _mod10_valid(v)
        )
        return df.index[present & well_formed.fillna(False) & bad.fillna(False)].tolist()

    def _check_gln_check(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        series = df[mapping.actual("gln")]
        present = series.notna()
        text = series.astype("string").str.strip()
        ok = text.map(lambda v: isinstance(v, str) and v.isdigit() and len(v) == 13
                      and _mod10_valid(v))
        return df.index[present & ~ok.fillna(False)].tolist()

    def _check_nonnull_maxlen(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        col = mapping.actual("product_description")
        series = df[col]
        max_length = int(rule.params.get("max_length", 200))
        too_long = series.astype("string").str.len() > max_length
        bad = series.isna() | too_long.fillna(False)
        return df.index[bad].tolist()

    def _check_content_uom(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        content = pd.to_numeric(df[mapping.actual("net_content")], errors="coerce")
        uom = df[mapping.actual("net_content_uom")]
        has_uom = uom.notna() & (uom.astype("string").str.strip() != "")
        has_pos_content = content.notna() & (content > 0)
        bad = (has_uom & ~has_pos_content) | (has_pos_content & ~has_uom)
        return df.index[bad].tolist()

    # -- custom repair ------------------------------------------------------

    def _repair_strip_nondigits(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule, result: RuleResult
    ) -> dict[Any, Any]:
        col = mapping.actual("gtin")
        fixes: dict[Any, Any] = {}
        for row in result.violation_rows:
            if row not in df.index:
                continue
            value = df.at[row, col]
            if pd.isna(value):
                continue
            digits = _NONDIGIT.sub("", str(value))
            if _gtin_well_formed(digits) and _mod10_valid(digits):
                fixes[row] = digits
        return fixes
