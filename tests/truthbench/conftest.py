"""Focused fixtures for the initial TruthBench fixture infrastructure tests."""

from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.truthbench.fixtures.base import FixtureBuilder


@pytest.fixture
def minimal_fixture():
    frame = pd.DataFrame(
        {"name": ["alpha", "beta"], "amount": [1.0, 2.0]},
        index=["r1", "r2"],
    )
    builder = FixtureBuilder(
        "v1",
        "minimal",
        frame,
        schema={"columns": ["name", "amount"]},
        policy={"locale": "en_US"},
        protected_columns=("name",),
    )
    builder.inject(
        "r2",
        "amount",
        "2.50",
        "repair",
        expected=2.5,
        family="numeric-format",
    )
    return builder.build()
