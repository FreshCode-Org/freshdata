"""Fixture registry for the independent TruthBench oracle."""

from __future__ import annotations

from typing import Callable

from . import crm, education, finance, government, healthcare, insurance, logistics, retail
from .base import FixtureBuilder, FixtureError, TruthFixture

_BUILDERS: dict[str, Callable[[int], TruthFixture]] = {}
DOMAINS: tuple[str, ...] = (
    "crm",
    "education",
    "finance",
    "government",
    "healthcare",
    "insurance",
    "logistics",
    "retail",
)


def register_fixture(domain: str, builder: Callable[[int], TruthFixture]) -> None:
    if not isinstance(domain, str) or not domain:
        raise ValueError("fixture domain must be a non-empty string")
    if domain in _BUILDERS:
        raise ValueError(f"fixture domain already registered: {domain}")
    _BUILDERS[domain] = builder


register_fixture("crm", crm.build)
register_fixture("education", education.build)
register_fixture("finance", finance.build)
register_fixture("government", government.build)
register_fixture("healthcare", healthcare.build)
register_fixture("insurance", insurance.build)
register_fixture("logistics", logistics.build)
register_fixture("retail", retail.build)


def build_fixture(domain: str, seed: int = 1729) -> TruthFixture:
    try:
        builder = _BUILDERS[domain]
    except KeyError as exc:
        raise FixtureError(f"unknown fixture domain: {domain!r}") from exc
    fixture = builder(int(seed))
    fixture.validate()
    return fixture


__all__ = [
    "DOMAINS",
    "FixtureBuilder",
    "FixtureError",
    "TruthFixture",
    "build_fixture",
    "register_fixture",
]
