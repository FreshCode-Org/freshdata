"""Polars adapter round-trip tests."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_ONLINE_TIER1, load_online_fixture

pytest.importorskip("polars")
import polars as pl  # noqa: E402


@pytest.mark.parametrize("name", ALL_ONLINE_TIER1[:5])
def test_polars_round_trip_parity(name):
    pdf = load_online_fixture(name)
    pl_df = pl.from_pandas(pdf)
    out_pd = fd.clean(pdf, strategy="balanced", verbose=False)
    out_pl = fd.clean(pl_df, strategy="balanced", verbose=False)
    out_pl_pd = out_pl.to_pandas()
    assert out_pd.shape == out_pl_pd.shape
    assert list(out_pd.columns) == list(out_pl_pd.columns)


def test_polars_return_type():
    pdf = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    pl_df = pl.from_pandas(pdf)
    result = fd.clean(pl_df, verbose=False)
    assert isinstance(result, pl.DataFrame)


def test_infer_roles_accepts_polars():
    pdf = pd.DataFrame({"customer_id": [1, 2, 3], "amount": [1.0, 2.0, 3.0]})
    pl_df = pl.from_pandas(pdf)
    roles = fd.infer_roles(pl_df)
    assert len(roles) == 2
