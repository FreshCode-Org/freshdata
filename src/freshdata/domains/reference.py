"""Centralized, cached access to the bundled reference code sets.

Historically each domain pack loaded its own ``reference/*.json`` inline. This module
gives every pack (and the format parsers) one cached, normalizer-aware way to load a
reference set — bundled cross-pack sets live in ``domains/bundled/`` — without each
caller re-implementing case folding, synonym coercion, or ``_meta`` bookkeeping.

Every set carries a ``_meta`` block (version / source / disclaimer). These are
**documented common subsets**, not exhaustive code systems; the ``_meta`` says so.

Two reference-file shapes are supported transparently:

- ``{"_meta": {...}, "codes": [...], "coerce": {...}}`` — a flat code list (ISO-4217,
  UCUM, UN/CEFACT units, ...).
- ``{"_meta": {...}, "AF": {...}, "AL": {...}}`` — a key -> payload mapping (ISO-3166);
  its keys are the codes and the payloads stay available via :attr:`ReferenceSet.mapping`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

_BUNDLED_DIR = Path(__file__).resolve().parent / "bundled"

_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "exact": lambda s: s,
    "casefold": lambda s: s.casefold(),
    "upper": lambda s: s.upper(),
}


@dataclass(frozen=True)
class ReferenceSet:
    """An immutable, normalized view over one bundled reference code set."""

    name: str
    codes: frozenset[str]
    coerce_map: dict[str, str]
    meta: dict[str, Any]
    normalizer: str = "exact"
    mapping: dict[str, Any] | None = None

    def _norm(self, value: Any) -> str:
        return _NORMALIZERS[self.normalizer](str(value))

    def __post_init__(self) -> None:
        if self.normalizer not in _NORMALIZERS:
            raise ValueError(f"unknown normalizer {self.normalizer!r}; "
                             f"choose from {sorted(_NORMALIZERS)}")

    @property
    def normalized_codes(self) -> frozenset[str]:
        return frozenset(self._norm(c) for c in self.codes)

    def contains(self, value: Any) -> bool:
        """True if *value* is a recognized code (after normalization/coercion)."""
        if pd.isna(value):
            return False
        key = self._norm(value)
        return key in self.normalized_codes or key in {self._norm(k) for k in self.coerce_map}

    def invalid_mask(self, series: pd.Series) -> pd.Series:
        """Boolean mask of *present* values that are neither a code nor a known synonym."""
        valid = self.normalized_codes | {self._norm(k) for k in self.coerce_map}
        normalized = series.map(lambda v: self._norm(v) if not pd.isna(v) else v)
        return series.notna() & ~normalized.isin(valid)

    def coerce(self, series: pd.Series) -> pd.Series:
        """Map known synonyms (and case variants) to their canonical code."""
        canonical = {self._norm(c): c for c in self.codes}
        synonyms = {self._norm(k): v for k, v in self.coerce_map.items()}

        def _fix(value: Any) -> Any:
            if pd.isna(value):
                return value
            key = self._norm(value)
            if key in canonical and canonical[key] == value:
                return value
            return canonical.get(key) or synonyms.get(key) or value

        return series.map(_fix)


@cache
def load_reference(name: str, *, normalizer: str = "exact") -> ReferenceSet:
    """Load and cache the bundled reference set *name* (without the ``.json`` suffix)."""
    path = _BUNDLED_DIR / f"{name}.json"
    if not path.exists():
        raise KeyError(f"unknown reference set {name!r}; "
                       f"available: {available_references()}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta = dict(raw.get("_meta", {}))
    if "codes" in raw:
        codes = frozenset(str(c) for c in raw["codes"])
        mapping = None
    else:
        mapping = {k: v for k, v in raw.items() if k != "_meta"}
        codes = frozenset(mapping)
    coerce_map = {str(k): str(v) for k, v in raw.get("coerce", {}).items()}
    return ReferenceSet(name, codes, coerce_map, meta, normalizer, mapping)


def available_references() -> list[str]:
    """Names of the bundled reference sets that :func:`load_reference` can load."""
    return sorted(p.stem for p in _BUNDLED_DIR.glob("*.json"))
