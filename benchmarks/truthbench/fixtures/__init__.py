"""Fixture registry for the independent TruthBench oracle."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .base import FixtureBuilder, FixtureError, TruthFixture

_BUILDERS: dict[str, Callable[[int], TruthFixture]] = {}
DOMAINS: tuple[str, ...] = ("minimal",)


def register_fixture(domain: str, builder: Callable[[int], TruthFixture]) -> None:
    if not isinstance(domain, str) or not domain:
        raise ValueError("fixture domain must be a non-empty string")
    if domain in _BUILDERS:
        raise ValueError(f"fixture domain already registered: {domain}")
    _BUILDERS[domain] = builder


def _minimal(seed: int) -> TruthFixture:
    frame = pd.DataFrame(
        {
            "name": ["alpha", "beta"],
            "amount": [float(seed % 10), float((seed % 10) + 1)],
        },
        index=["r1", "r2"],
    )
    builder = FixtureBuilder(
        "v1",
        "minimal",
        frame,
        seed=seed,
        schema={"columns": ["name", "amount"]},
        policy={"locale": "en_US"},
        protected_columns=("name",),
    )
    builder.inject("r2", "amount", "2.50", "repair", expected=2.5, family="numeric-format")
    return builder.build()


register_fixture("minimal", _minimal)


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
