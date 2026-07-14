"""Independent, deterministic fixture construction primitives."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..exact import canonical_json, encode_typed
from ..models import UNSET, CaseExpectation, Disposition, GoldCell


class FixtureError(ValueError):
    """A fixture violates the TruthBench construction contract."""


_HASH_KEY = b"truthbench-fixture-hash-v1"
_SYNTHETIC_EMAIL = re.compile(r"@[A-Za-z0-9.-]+\.invalid(?:$|[^A-Za-z0-9])")
_SYNTHETIC_ID = re.compile(r"(?:^|[^A-Za-z0-9])TB-[A-Za-z0-9-]+(?:$|[^A-Za-z0-9])")
_SYNTHETIC_PHONE = re.compile(r"(?:^|[^0-9])555[- .]?01[0-9]{2}(?:$|[^0-9])")


def _synthetic_pii(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(
        _SYNTHETIC_EMAIL.search(value)
        or _SYNTHETIC_ID.search(value)
        or _SYNTHETIC_PHONE.search(value)
    )


def _typed(value: Any, dtype: Any = None, *, sensitive: bool = False) -> dict[str, Any]:
    return encode_typed(
        value,
        dtype=dtype,
        sensitive=sensitive,
        digest_key=_HASH_KEY if sensitive else None,
    ).to_dict()


@dataclass(frozen=True)
class TruthFixture:
    """A pristine/adversarial pair and complete cell-level oracle."""

    version: str
    domain: str
    seed: int
    pristine: pd.DataFrame
    frame: pd.DataFrame
    cells: tuple[GoldCell, ...]
    schema: Mapping[str, Any]
    policy: Mapping[str, Any]
    protected_columns: tuple[str, ...]
    pii_canaries: Mapping[str, str]
    row_cases: tuple[CaseExpectation, ...]
    schema_cases: tuple[CaseExpectation, ...]
    fixture_hash: str

    @property
    def adversarial(self) -> pd.DataFrame:
        return self.frame

    @property
    def pristine_frame(self) -> pd.DataFrame:
        return self.pristine

    @property
    def adversarial_frame(self) -> pd.DataFrame:
        return self.frame

    def validate(self) -> None:
        if not self.version or not self.domain:
            raise FixtureError("fixture version and domain are required")
        if not self.frame.index.is_unique or self.frame.index.hasnans:
            raise FixtureError("frame row IDs must be unique and non-missing")
        keys = [
            (str(row), str(column)) for row in self.frame.index for column in self.frame.columns
        ]
        labels = [(cell.row_id, cell.column) for cell in self.cells]
        if len(labels) != len(set(labels)) or set(labels) != set(keys):
            raise FixtureError("fixture must have exactly one label per physical cell")
        for cell in self.cells:
            if cell.fixture_version != self.version or cell.domain != self.domain:
                raise FixtureError("cell identity does not match fixture")
            if cell.disposition is Disposition.REPAIR and cell.expected_output is None:
                raise FixtureError(f"repair cell {cell.cell_id} has no expected output")
            if cell.sensitive and (not cell.canary_id or cell.canary_id not in self.pii_canaries):
                raise FixtureError(f"sensitive cell {cell.cell_id} has no canary")
        for case in (*self.row_cases, *self.schema_cases):
            if case.domain != self.domain or case.fixture_version != self.version:
                raise FixtureError("case identity does not match fixture")
        if any(case.kind != "row" for case in self.row_cases):
            raise FixtureError("row_cases must contain row expectations")
        if any(case.kind != "schema" for case in self.schema_cases):
            raise FixtureError("schema_cases must contain schema expectations")
        if self.fixture_hash:
            payload = self.to_dict()
            payload.pop("fixture_hash", None)
            expected_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
            if self.fixture_hash != expected_hash:
                raise FixtureError("fixture hash does not match fixture contents")

    def to_dict(self) -> dict[str, Any]:
        """Return a privacy-safe, JSON-ready representation."""

        def frame_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
            by_key = {(c.row_id, c.column): c for c in self.cells}
            rows: list[dict[str, Any]] = []
            for row in frame.index:
                row_id = str(row)
                values = {
                    str(column): _typed(
                        frame.at[row, column],
                        frame[column].dtype,
                        sensitive=by_key[(row_id, str(column))].sensitive,
                    )
                    for column in frame.columns
                }
                rows.append({"row_id": row_id, "values": values})
            return rows

        canaries = {
            key: _typed(value, sensitive=True) for key, value in sorted(self.pii_canaries.items())
        }
        return {
            "version": self.version,
            "domain": self.domain,
            "seed": self.seed,
            "pristine": frame_payload(self.pristine),
            "frame": frame_payload(self.frame),
            "cells": [cell.to_dict() for cell in self.cells],
            "schema": dict(self.schema),
            "policy": dict(self.policy),
            "protected_columns": list(self.protected_columns),
            "pii_canaries": canaries,
            "row_cases": [case.to_dict() for case in self.row_cases],
            "schema_cases": [case.to_dict() for case in self.schema_cases],
            "fixture_hash": self.fixture_hash,
        }


class FixtureBuilder:
    """Build a fixture while maintaining a complete, atomic label matrix."""

    def __init__(
        self,
        version: str,
        domain: str,
        frame: pd.DataFrame,
        *,
        seed: int = 1729,
        schema: Mapping[str, Any] | None = None,
        policy: Mapping[str, Any] | None = None,
        protected_columns: tuple[str, ...] | list[str] = (),
        row_cases: tuple[CaseExpectation, ...] | list[CaseExpectation] = (),
        schema_cases: tuple[CaseExpectation, ...] | list[CaseExpectation] = (),
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty or frame.columns.empty:
            raise FixtureError("frame must contain at least one row and column")
        if not version or not domain:
            raise FixtureError("version and domain are required")
        if frame.index.hasnans or not frame.index.is_unique:
            raise FixtureError("row IDs must be unique and non-missing")
        row_ids = [str(row) for row in frame.index]
        if any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
            raise FixtureError("row IDs must be unique and non-missing")
        if any(not isinstance(column, str) or not column for column in frame.columns):
            raise FixtureError("column names must be non-empty strings")
        if len(set(frame.columns)) != len(frame.columns):
            raise FixtureError("column names must be unique")

        self.version = version
        self.domain = domain
        self.seed = seed
        self.pristine = frame.copy(deep=True)
        self.pristine.index = row_ids
        self.frame = frame.copy(deep=True)
        self.frame.index = row_ids
        self.schema = dict(
            schema
            or {
                "columns": list(frame.columns),
                "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
            }
        )
        self.policy = dict(policy or {})
        self.protected_columns = tuple(protected_columns)
        unknown_protected = set(self.protected_columns) - set(frame.columns)
        if unknown_protected:
            raise FixtureError(f"unknown protected columns: {sorted(unknown_protected)}")
        self._labels = {
            (row_id, str(column)): GoldCell.create(
                version, domain, row_id, str(column), Disposition.PRESERVE, family="background"
            )
            for row_id in row_ids
            for column in frame.columns
        }
        self._canaries: dict[str, str] = {}
        self._row_cases = list(row_cases)
        self._schema_cases = list(schema_cases)

    def inject(
        self,
        row_id: str,
        column: str,
        value: Any,
        disposition: Disposition | str,
        *,
        expected: Any = UNSET,
        expected_dtype: Any = None,
        family: str,
        sensitive: bool = False,
    ) -> None:
        key = (str(row_id), str(column))
        if key not in self._labels:
            raise FixtureError(f"unknown cell {key}")
        if not isinstance(family, str) or not family:
            raise FixtureError("family is required")
        try:
            parsed = Disposition(disposition)
        except ValueError as exc:
            raise FixtureError(f"unknown disposition: {disposition!r}") from exc
        if parsed is Disposition.REPAIR and expected is UNSET:
            raise FixtureError("repair cells require an expected output")
        if sensitive and not _synthetic_pii(value):
            raise FixtureError("sensitive values must use synthetic PII domains")

        canary_id = None
        if sensitive:
            canary_id = f"{self.domain}-{key[0]}-{key[1]}"
        if expected_dtype is None and expected is not UNSET:
            expected_dtype = self.pristine[key[1]].dtype
        candidate = GoldCell.create(
            self.version,
            self.domain,
            key[0],
            key[1],
            parsed,
            expected_output=expected,
            expected_dtype=expected_dtype,
            family=family,
            sensitive=sensitive,
            canary_id=canary_id,
        )
        previous = self._labels[key]
        if (
            previous.disposition is Disposition.REPAIR
            and candidate.disposition is Disposition.REPAIR
            and previous.expected_output != candidate.expected_output
        ):
            raise FixtureError(f"contradictory repair output for cell {key}")

        # All validation above happens before either mutable structure changes.
        if previous.canary_id and previous.canary_id != canary_id:
            self._canaries.pop(previous.canary_id, None)
        if canary_id:
            self._canaries[canary_id] = value
        # A corruption may intentionally change a column's scalar type.  Cast
        # to object first so pandas does not coerce or emit a FutureWarning.
        self.frame[key[1]] = self.frame[key[1]].astype(object)
        self.frame.at[key[0], key[1]] = value
        self._labels[key] = candidate

    def add_row_case(self, name: str, disposition: Disposition | str, **kwargs: Any) -> None:
        self._row_cases.append(
            CaseExpectation.create(self.version, self.domain, "row", name, disposition, **kwargs)
        )

    def add_schema_case(self, name: str, disposition: Disposition | str, **kwargs: Any) -> None:
        self._schema_cases.append(
            CaseExpectation.create(
                self.version, self.domain, "schema", name, disposition, **kwargs
            )
        )

    def build(self) -> TruthFixture:
        cells = tuple(
            self._labels[(str(row), str(column))]
            for row in self.frame.index
            for column in self.frame.columns
        )
        fixture = TruthFixture(
            version=self.version,
            domain=self.domain,
            seed=self.seed,
            pristine=self.pristine.copy(deep=True),
            frame=self.frame.copy(deep=True),
            cells=cells,
            schema=dict(self.schema),
            policy=dict(self.policy),
            protected_columns=self.protected_columns,
            pii_canaries=dict(self._canaries),
            row_cases=tuple(self._row_cases),
            schema_cases=tuple(self._schema_cases),
            fixture_hash="",
        )
        fixture.validate()
        payload = fixture.to_dict()
        payload.pop("fixture_hash", None)
        digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        result = TruthFixture(**{**fixture.__dict__, "fixture_hash": digest})
        result.validate()
        return result


BuilderFactory = Callable[[int], TruthFixture]
