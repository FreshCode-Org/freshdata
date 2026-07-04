"""Representation corruptors: formatting dirt with unambiguous repairs."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from .base import CorruptionLabel, Corruptor

_SENTINELS = ("N/A", "null", "-", "NONE", "?", "missing")
_NA_VARIANTS = ("", "NA", "n/a", "None", "  ")
_BOOL_SYNONYMS = {
    "true": ("Y", "yes", "TRUE", "1", "y"),
    "false": ("N", "no", "FALSE", "0", "n"),
}
_UNITS = ("kg", "g", "pcs", "ltr")


def _is_number(text: str) -> bool:
    try:
        float(text.replace(",", ""))
    except ValueError:
        return False
    return True


def _whitespace(v: str, rng: random.Random, params: dict[str, Any]) -> str:
    return rng.choice(("  {}", "{} ", " {}  ", "\t{}", "{}\t")).format(v)


def _casing(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not any(ch.isalpha() for ch in v):
        return None
    choices = [c for c in (v.upper(), v.lower(), v.title(), v.swapcase()) if c != v]
    return rng.choice(choices) if choices else None


def _sentinel(v: str, rng: random.Random, params: dict[str, Any]) -> str:
    return rng.choice(params.get("sentinels", _SENTINELS))


def _na_variant(v: str, rng: random.Random, params: dict[str, Any]) -> str:
    return rng.choice(_NA_VARIANTS)


def _mixed_dtype(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _is_number(v):
        return None
    number = float(v.replace(",", ""))
    return rng.choice((f"{number:.1f}", f"{number:.3f}", f" {number:g} ", f"'{v}'"))


def _currency(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _is_number(v):
        return None
    number = float(v.replace(",", ""))
    return rng.choice((
        f"₹{number:,.2f}", f"Rs. {number:.2f}", f"INR {number:,.2f}", f"{number:.2f} INR",
    ))


def _thousands(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _is_number(v) or "," in v or float(v) < 1000:
        return None
    return f"{float(v):,.10g}"


def _unit_suffix(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _is_number(v):
        return None
    unit = rng.choice(params.get("units", _UNITS))
    return rng.choice((f"{v}{unit}", f"{v} {unit}"))


def _bool_synonym(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    synonyms = _BOOL_SYNONYMS.get(v.strip().lower())
    return rng.choice(synonyms) if synonyms else None


def _duplicate_rows(
    df: pd.DataFrame, rng: random.Random, params: dict[str, Any]
) -> tuple[pd.DataFrame, list[CorruptionLabel]]:
    n = int(params.get("n_duplicates", 2))
    if df.empty or n <= 0:
        return df, []
    picks = [rng.randrange(len(df)) for _ in range(n)]
    extra = df.iloc[picks]
    out = pd.concat([df, extra], ignore_index=True)
    labels = [
        CorruptionLabel(
            raw_value=f"duplicate_of_row_{p}", clean_value=None, column=None,
            transform_family="row_structure", params={"source_row": p},
            should_repair=True, should_auto_apply=True, risk="low",
            corruptor="duplicate_row_injection", row=len(df) + i,
        )
        for i, p in enumerate(picks)
    ]
    return out, labels


whitespace_insertion = Corruptor(
    name="whitespace_insertion", family="representation", fn=_whitespace)
casing_change = Corruptor(
    name="casing_change", family="representation", fn=_casing)
sentinel_injection = Corruptor(
    name="sentinel_injection", family="representation", fn=_sentinel,
    params={"target": "missing"})
empty_na_variants = Corruptor(
    name="empty_na_variants", family="representation", fn=_na_variant,
    params={"target": "missing"})
mixed_dtype_stringification = Corruptor(
    name="mixed_dtype_stringification", family="representation", fn=_mixed_dtype)
currency_formatting = Corruptor(
    name="currency_formatting", family="representation", fn=_currency)
thousands_separators = Corruptor(
    name="thousands_separators", family="representation", fn=_thousands)
unit_suffix_insertion = Corruptor(
    name="unit_suffix_insertion", family="representation", fn=_unit_suffix,
    risk="medium", should_auto_apply=False)
boolean_synonym_replacement = Corruptor(
    name="boolean_synonym_replacement", family="representation", fn=_bool_synonym)
duplicate_row_injection = Corruptor(
    name="duplicate_row_injection", family="row_structure", kind="row",
    frame_fn=_duplicate_rows)

REPRESENTATION_CORRUPTORS = (
    whitespace_insertion,
    casing_change,
    sentinel_injection,
    empty_na_variants,
    mixed_dtype_stringification,
    currency_formatting,
    thousands_separators,
    unit_suffix_insertion,
    boolean_synonym_replacement,
    duplicate_row_injection,
)
