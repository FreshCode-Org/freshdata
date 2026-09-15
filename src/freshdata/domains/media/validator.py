"""The media domain pack: EIDR content and DDEX release metadata hygiene.

Validates entertainment metadata in two sub-schemas selected by ``media_type``:
``content`` (EIDR — film, TV, series) and ``release`` (DDEX — music releases, tracks).
When ``media_type`` is omitted it is auto-detected from the column signature; an
indeterminate signature raises :class:`AmbiguousMediaTypeError`.

The EIDR DOI check character (ISO 7064 hybrid MOD 37,36) and the ICPN (UPC/EAN) GS1 mod-10 check
digit are implemented here as pure, unit-tested functions.
"""

from __future__ import annotations

import json
import re
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .._common import (
    check_fhir_date,
    check_iso_date,
    check_positive_integer,
    check_requires_field,
    check_requires_when_value,
)
from ..base import (
    ColumnMapping,
    ConfigDrivenValidator,
    DomainError,
    Rule,
    RuleResult,
    ValidationReport,
)

_PACK_DIR = Path(__file__).resolve().parent
_BUNDLED_DIR = _PACK_DIR.parent / "bundled"
_NONDIGIT = re.compile(r"\D")
_ICPN_LENGTHS = (12, 13)


def _sort_key(value: Any) -> tuple[int, Any]:
    """Order row labels of mixed type deterministically (numbers first, then strings)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))
_EIDR_RE = re.compile(
    r"10\.5240/([0-9A-Z]{4})-([0-9A-Z]{4})-([0-9A-Z]{4})-([0-9A-Z]{4})-([0-9A-Z]{4})-([0-9A-Z])"
)
# ISO/IEC 7064 hybrid system MOD 37,36: modulus M = 36 over the alphabet 0-9A-Z,
# with M + 1 = 37 as the second modulus. The EIDR ID Format specification names
# "ISO 7064 Mod 37,36"; the pure MOD 37-2 system (with the supplementary ``*``) is
# a different algorithm and rejects real EIDR IDs.
_ISO7064_M = 36


# -- pure check-digit functions (unit-tested in tests/domains/test_media.py) --

def _char_value(char: str) -> int:
    """Map an alphanumeric to its ISO 7064 value (0-9 -> 0-9, A-Z -> 10-35)."""
    return ord(char) - 48 if char.isdigit() else ord(char) - 55


def _value_char(value: int) -> str:
    """Inverse of :func:`_char_value` for values 0-35."""
    return chr(48 + value) if value < 10 else chr(55 + value)


def eidr_check_char(payload: str) -> str:
    """Return the EIDR check character for *payload* via ISO 7064 hybrid MOD 37,36.

    *payload* is the 20-character DOI suffix (hyphens removed, check char excluded).
    """
    product = _ISO7064_M
    for char in payload:
        total = (product + _char_value(char)) % _ISO7064_M or _ISO7064_M
        product = total * 2 % (_ISO7064_M + 1)
    return _value_char((_ISO7064_M + 1 - product) % _ISO7064_M)


def is_valid_eidr(value: Any) -> bool:
    """True if *value* is a well-formed EIDR DOI with a valid MOD 37,36 check character."""
    if not isinstance(value, str):
        return False
    match = _EIDR_RE.fullmatch(value.strip())
    if match is None:
        return False
    payload = "".join(match.group(i) for i in range(1, 6))
    return eidr_check_char(payload) == match.group(6)


def is_valid_icpn(value: Any) -> bool:
    """True if *value* is a 12-digit UPC or 13-digit EAN with a valid GS1 mod-10 digit."""
    if value is None:
        return False
    digits = _NONDIGIT.sub("", str(value))
    if len(digits) not in _ICPN_LENGTHS:
        return False
    body, check = digits[:-1], int(digits[-1])
    total = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(body)))
    return (10 - total % 10) % 10 == check


@cache
def _ref(name: str) -> dict[str, Any]:
    with open(_PACK_DIR / "reference" / f"{name}.json", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _iso3166_alpha2() -> list[str]:
    with open(_BUNDLED_DIR / "iso3166.json", encoding="utf-8") as handle:
        data = json.load(handle)
    return sorted(code for code in data if code != "_meta")


def _field_present(columns: list[Any], aliases: dict[str, Any], canonical: str) -> bool:
    lowered = {str(c).casefold() for c in columns}
    if canonical in columns or canonical.casefold() in lowered:
        return True
    return any(
        re.fullmatch(pattern, str(column), flags=re.IGNORECASE)
        for pattern in aliases.get(canonical, ())
        for column in columns
    )


class AmbiguousMediaTypeError(DomainError):
    """Raised when the media sub-schema cannot be detected from columns."""

    def __init__(self, candidates: list[str]) -> None:
        self.candidates = list(candidates)
        listed = ", ".join(self.candidates) if self.candidates else "(none)"
        super().__init__(
            f"could not determine media_type from columns; candidates: {listed}. "
            "Pass media_type='content' or media_type='release'."
        )


# Distinctive (non-shared) columns used for sub-schema auto-detection.
_CONTENT_SIGNAL = ("eidr_id", "runtime_seconds", "series_eidr_id", "season_number",
                   "episode_number")
_RELEASE_SIGNAL = ("release_id", "icpn", "track_count", "party_role", "label_name",
                   "artist_name")

_SUBSCHEMA_CONFIG: dict[str, dict[str, Any]] = {
    "content": {
        "canonical_fields": (
            "eidr_id", "title", "content_type", "release_date", "country_of_origin",
            "language", "runtime_seconds", "distributor_id", "series_eidr_id",
            "season_number", "episode_number",
        ),
        "required_fields": ("eidr_id",),
        "id_fields": ("eidr_id", "distributor_id"),
        "aliases": {
            "eidr_id": (r"eidr_?id", r"eidr", r"content_?id"),
            "title": (r"title", r"name"),
            "content_type": (r"content_?type", r"type", r"referent_?type"),
            "release_date": (r"release_?date", r"date"),
            "country_of_origin": (r"country_?of_?origin", r"country", r"origin_?country"),
            "language": (r"language", r"lang", r"original_?language"),
            "runtime_seconds": (r"runtime_?seconds", r"runtime", r"duration_?seconds"),
            "distributor_id": (r"distributor_?id", r"distributor"),
            "series_eidr_id": (r"series_?eidr_?id", r"series_?id", r"parent_?eidr_?id"),
            "season_number": (r"season_?number", r"season"),
            "episode_number": (r"episode_?number", r"episode"),
        },
    },
    "release": {
        "canonical_fields": (
            "release_id", "icpn", "release_type", "title", "language", "label_name",
            "artist_name", "party_id", "party_role", "territory", "release_date",
            "price_tier", "track_count",
        ),
        "required_fields": ("release_id",),
        "id_fields": ("release_id", "party_id"),
        "aliases": {
            "release_id": (r"release_?id", r"ddex_?release_?id"),
            "icpn": (r"icpn", r"upc", r"ean", r"barcode"),
            "release_type": (r"release_?type", r"type"),
            "title": (r"title", r"release_?title", r"name"),
            "language": (r"language", r"lang"),
            "label_name": (r"label_?name", r"label", r"record_?label"),
            "artist_name": (r"artist_?name", r"artist", r"main_?artist"),
            "party_id": (r"party_?id", r"contributor_?id"),
            "party_role": (r"party_?role", r"role", r"contributor_?role"),
            "territory": (r"territory", r"territory_?code", r"region"),
            "release_date": (r"release_?date", r"date"),
            "price_tier": (r"price_?tier", r"price"),
            "track_count": (r"track_?count", r"tracks", r"number_?of_?tracks"),
        },
    },
}


class MediaValidator(ConfigDrivenValidator):
    """Validator for EIDR content / DDEX release metadata frames."""

    domain_name = "media"
    version = "0.1.0"
    schema_version = "eidr-ddex-2024"

    rules_path = str(_PACK_DIR / "rules.yaml")

    def __init__(
        self,
        *,
        column_map: Any = None,
        media_type: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self._media_type: str | None = (
            self._normalize_media_type(media_type) if media_type is not None else None
        )
        super().__init__(column_map=column_map)
        if self._media_type is not None:
            self._activate(self._media_type)

    # -- sub-schema resolution ---------------------------------------------

    @property
    def media_type(self) -> str | None:
        return self._media_type

    def _normalize_media_type(self, value: Any) -> str:
        text = str(value).strip().casefold()
        if text not in _SUBSCHEMA_CONFIG:
            raise DomainError(
                f"unknown media_type {value!r}; expected 'content' or 'release'"
            )
        return text

    def _activate(self, media_type: str) -> None:
        config = _SUBSCHEMA_CONFIG[media_type]
        self.canonical_fields = config["canonical_fields"]
        self.required_fields = config["required_fields"]
        self.id_fields = config["id_fields"]
        self.aliases = config["aliases"]

    def _detect_media_type(self, df: pd.DataFrame) -> str:
        columns = list(df.columns)
        is_content = any(
            _field_present(columns, _SUBSCHEMA_CONFIG["content"]["aliases"], field)
            for field in _CONTENT_SIGNAL
        )
        is_release = any(
            _field_present(columns, _SUBSCHEMA_CONFIG["release"]["aliases"], field)
            for field in _RELEASE_SIGNAL
        )
        if is_content and not is_release:
            return "content"
        if is_release and not is_content:
            return "release"
        candidates = [m for m, flag in (("content", is_content), ("release", is_release)) if flag]
        raise AmbiguousMediaTypeError(candidates or ["content", "release"])

    # -- engine hooks -------------------------------------------------------

    def detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        if self._media_type is None:
            self._media_type = self._detect_media_type(df)
            self._activate(self._media_type)
        return super().detect_columns(df)

    def _run_rule(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> RuleResult:
        rule_type = rule.params.get("media_type")
        if rule_type and self._media_type and rule_type != self._media_type:
            return RuleResult(
                rule_id=rule.id, name=rule.name, layer=rule.layer, severity=rule.severity,
                fields=rule.fields, check=rule.check, status="skipped", repair=rule.repair,
                message=f"skipped: applies to {rule_type}, not {self._media_type}",
            )
        return super()._run_rule(df, mapping, rule)

    def register_extensions(self) -> None:
        self.register_check("eidr_valid", self._check_eidr)
        self.register_check("icpn_valid", self._check_icpn)
        self.register_check("iso_or_partial_date", check_fhir_date)
        self.register_check("iso8601_date", check_iso_date)
        self.register_check("positive_integer", check_positive_integer)
        self.register_check("positive_int_fields", self._check_positive_int_fields)
        self.register_check("requires_when_value", check_requires_when_value)
        self.register_check("requires_field", check_requires_field)
        self.register_check("movie_no_episode", self._check_movie_no_episode)
        self.register_check("single_track_count", self._check_single_track_count)

    def load_reference_values(self, name: str) -> Any:
        if name in ("eidr_content_types", "ddex_release_types", "ddex_party_roles",
                    "iso639_languages", "territory_codes"):
            return _ref(name)["codes"]
        if name == "iso3166":
            return _iso3166_alpha2()
        return super().load_reference_values(name)

    def reference_sources(self) -> list[dict[str, Any]]:
        names = ("eidr_content_types", "ddex_release_types", "ddex_party_roles",
                 "iso639_languages", "territory_codes")
        sources = [{"name": name, **_ref(name)["_meta"]} for name in names]
        with open(_BUNDLED_DIR / "iso3166.json", encoding="utf-8") as handle:
            sources.append({"name": "iso3166", **json.load(handle).get("_meta", {})})
        return sources

    def describe(self) -> dict[str, Any]:
        description = super().describe()
        description["media_type"] = self._media_type
        return description

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        return super().validate(df)

    # -- custom checks ------------------------------------------------------

    def _check_eidr(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        series = df[mapping.actual(rule.fields[0])]
        present = series.notna()
        ok = series.map(is_valid_eidr)
        return df.index[present & ~ok.fillna(False)].tolist()

    def _check_icpn(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        series = df[mapping.actual("icpn")]
        present = series.notna()
        ok = series.map(is_valid_icpn)
        return df.index[present & ~ok.fillna(False)].tolist()

    def _check_positive_int_fields(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        rows: set[Any] = set()
        for field in rule.params.get("targets", ()):
            col = mapping.actual(field)
            if col is None:
                continue
            series = df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            is_pos_int = numeric.notna() & (numeric > 0) & (numeric == numeric.round())
            rows.update(df.index[series.notna() & ~is_pos_int].tolist())
        return sorted(rows, key=_sort_key)

    def _check_movie_no_episode(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        content_type = df[mapping.actual("content_type")].astype("string").str.casefold()
        is_film = content_type.isin(["movie", "short"])
        rows: set[Any] = set()
        for field in ("season_number", "episode_number"):
            col = mapping.actual(field)
            if col is None:
                continue
            rows.update(df.index[is_film.fillna(False) & df[col].notna()].tolist())
        return sorted(rows, key=_sort_key)

    def _check_single_track_count(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        release_type_col = mapping.actual("release_type")
        if release_type_col is None:
            return []
        is_single = df[release_type_col].astype("string").str.casefold() == "single"
        track_count = pd.to_numeric(df[mapping.actual("track_count")], errors="coerce")
        bad = (
            is_single.fillna(False)
            & track_count.notna()
            & ((track_count < 1) | (track_count > 3))
        )
        return df.index[bad].tolist()
