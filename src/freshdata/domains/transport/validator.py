"""The transport domain pack: static GTFS feed hygiene.

A GTFS feed is a set of related frames (stops, routes, trips, stop_times). This
validator checks one file at a time — :func:`freshdata.clean` drives it once per
file in full-feed mode and supplies the sibling frames as ``feed`` so cross-file
references (trip -> route) can be resolved. Repairs are flag-only: IDs,
coordinates, and times are never silently mutated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from ..base import ColumnMapping, ConfigDrivenValidator, DomainError, Rule, RuleResult, SkipCheck

_PACK_DIR = Path(__file__).resolve().parent
_GTFS_TIME = re.compile(r"(\d{1,3}):([0-5]\d):([0-5]\d)")


def _normalize_file_name(name: str) -> str:
    normalized = str(name).strip().casefold()
    return normalized[:-4] if normalized.endswith(".txt") else normalized


def _to_seconds(value: Any) -> int | None:
    """Parse a GTFS ``H:MM:SS`` time (hours may exceed 23) into seconds."""
    if not isinstance(value, str):
        return None
    match = _GTFS_TIME.fullmatch(value.strip())
    if not match:
        return None
    h, m, s = (int(g) for g in match.groups())
    return h * 3600 + m * 60 + s


def _sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, value)
    return (1, str(value))


class TransportValidator(ConfigDrivenValidator):
    """Validator for public-transit GTFS feeds (one file per instance)."""

    domain_name = "transport"
    version = "0.1.0"
    schema_version = "gtfs-2024"
    multi_frame = True
    supported_files = ("stops", "routes", "trips", "stop_times")

    canonical_fields = (
        "stop_id", "stop_name", "stop_lat", "stop_lon",
        "route_id", "route_short_name", "route_long_name", "route_type",
        "trip_id", "service_id",
        "arrival_time", "departure_time", "stop_sequence",
    )
    required_fields = ()  # per-file requirements come from the `required` rules
    id_fields = ("stop_id", "route_id", "trip_id")
    aliases: Mapping[str, Any] = {}
    rules_path = str(_PACK_DIR / "rules.yaml")

    @classmethod
    def normalize_file_name(cls, name: str) -> str:
        return _normalize_file_name(name)

    @classmethod
    def supports_file(cls, name: str) -> bool:
        return cls.normalize_file_name(name) in cls.supported_files

    def __init__(
        self,
        *,
        column_map: Mapping[str, str] | None = None,
        gtfs_file: str | None = None,
        feed: Mapping[str, pd.DataFrame] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.gtfs_file = self.normalize_file_name(gtfs_file) if gtfs_file is not None else None
        if self.gtfs_file is not None and self.gtfs_file not in self.supported_files:
            listed = ", ".join(f"{name}.txt" for name in self.supported_files)
            raise DomainError(
                f"unsupported GTFS file {gtfs_file!r}; supported files: {listed}"
            )
        self.feed: dict[str, pd.DataFrame] = {}
        for name, frame in (feed or {}).items():
            normalized = self.normalize_file_name(name)
            if normalized in self.feed:
                raise DomainError(
                    f"duplicate GTFS file keys resolve to {normalized!r}; use one name per file"
                )
            self.feed[normalized] = frame
        super().__init__(column_map=column_map)

    def validate(self, df: pd.DataFrame):
        if self.gtfs_file is None:
            raise DomainError("transport validation requires gtfs_file= or a feed dict")
        return super().validate(df)

    def register_extensions(self) -> None:
        self.register_check("route_type_valid", self._check_route_type)
        self.register_check("gtfs_time", self._check_gtfs_time)
        self.register_check("departure_ge_arrival", self._check_departure_ge_arrival)
        self.register_check("monotonic_sequence", self._check_monotonic_sequence)
        self.register_check("cross_file_reference", self._check_cross_file_reference)

    def _run_rule(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> RuleResult:
        rule_file = rule.params.get("gtfs_file")
        if rule_file and self.gtfs_file and rule_file != self.gtfs_file:
            return RuleResult(
                rule_id=rule.id, name=rule.name, layer=rule.layer, severity=rule.severity,
                fields=rule.fields, check=rule.check, status="skipped", repair=rule.repair,
                message=f"skipped: applies to {rule_file}, not {self.gtfs_file}",
            )
        return super()._run_rule(df, mapping, rule)

    def describe(self) -> dict[str, Any]:
        desc = super().describe()
        desc["gtfs_file"] = self.gtfs_file
        desc["feed_files"] = sorted(self.feed)
        return desc

    # -- custom checks ------------------------------------------------------

    def _check_route_type(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        col = mapping.actual("route_type")
        series = df[col]
        present = series.notna()
        numeric = pd.to_numeric(series, errors="coerce")
        allowed = {int(v) for v in rule.params.get("values", ())}
        bad = present & ~numeric.isin(allowed)
        return df.index[bad].tolist()

    def _check_gtfs_time(self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule) -> list[Any]:
        rows: set[Any] = set()
        for field_name in rule.fields:
            col = mapping.actual(field_name)
            if col is None:
                continue
            series = df[col]
            present = series.notna()
            ok = series.astype("string").str.fullmatch(_GTFS_TIME.pattern)
            rows.update(df.index[present & ~ok.fillna(False)].tolist())
        return sorted(rows, key=_sort_key)

    def _check_departure_ge_arrival(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        arrival = df[mapping.actual("arrival_time")].map(_to_seconds)
        departure = df[mapping.actual("departure_time")].map(_to_seconds)
        both = arrival.notna() & departure.notna()
        bad = both & (departure < arrival)
        return df.index[bad].tolist()

    def _check_monotonic_sequence(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        trip = df[mapping.actual("trip_id")]
        seq = pd.to_numeric(df[mapping.actual("stop_sequence")], errors="coerce")
        work = pd.DataFrame({"_trip": trip, "_seq": seq}, index=df.index)
        bad: list[Any] = []
        for _, group in work.groupby("_trip", sort=False):
            prev: float | None = None
            for idx, value in group["_seq"].items():
                if pd.isna(value):
                    continue
                if prev is not None and value <= prev:
                    bad.append(idx)
                prev = value
        return bad

    def _check_cross_file_reference(
        self, df: pd.DataFrame, mapping: ColumnMapping, rule: Rule
    ) -> list[Any]:
        target_file = rule.params.get("target_file")
        target_field = rule.params.get("target_field")
        if not target_file or not target_field:
            raise SkipCheck("rule is missing target_file/target_field")
        if target_file not in self.feed:
            raise SkipCheck(f"{target_file} not in the feed (single-file run)")
        target_df = self.feed[target_file]
        target_column = self._target_column(str(target_field), target_df)
        if target_column is None:
            raise SkipCheck(f"{target_field} not present in {target_file}")
        valid = set(target_df[target_column].dropna())
        series = df[mapping.actual(rule.fields[0])]
        present = series.notna()
        bad = present & ~series.isin(valid)
        return df.index[bad].tolist()

    def _target_column(self, canonical: str, target_df: pd.DataFrame) -> str | None:
        override = next(
            (
                actual
                for actual, mapped_canonical in self._column_map.items()
                if mapped_canonical == canonical and actual in target_df.columns
            ),
            None,
        )
        if override is not None:
            return override
        if canonical in target_df.columns:
            return canonical
        folded = {str(column).casefold(): str(column) for column in target_df.columns}
        return folded.get(canonical.casefold())
