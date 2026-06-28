"""Tests for baseline-free contract schema diff (F1c: diff_schema / ContractViolation)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from freshdata.api import clean
from freshdata.enterprise.contracts import (
    ColumnContract,
    ContractViolation,
    DataContract,
    DriftReport,
    diff_schema,
)


@pytest.fixture()
def contract() -> DataContract:
    return DataContract(
        name="customers",
        columns=(
            ColumnContract(name="id", dtype="int64", nullable=False),
            ColumnContract(name="email", dtype="object", nullable=False),
            ColumnContract(name="age", dtype="float64", nullable=True),
        ),
    )


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "email": ["a@b.com", "c@d.com"], "age": [25.0, None]})


# ── passing gate ────────────────────────────────────────────────────────────────


def test_diff_schema_returns_drift_report(clean_df, contract):
    rep = diff_schema(clean_df, contract=contract)
    assert isinstance(rep, DriftReport)


def test_clean_frame_passes(clean_df, contract):
    rep = diff_schema(clean_df, contract=contract)
    assert rep.passed


def test_to_frame_has_expected_columns(clean_df, contract):
    frame = diff_schema(clean_df, contract=contract).to_frame()
    for col in ("check_id", "category", "level", "status", "column", "message"):
        assert col in frame.columns


# ── unexpected column ────────────────────────────────────────────────────────────


def test_unexpected_column_warn(contract):
    df = pd.DataFrame({"id": [1], "email": ["x@y.com"], "age": [30.0], "extra": ["??"]})
    rep = diff_schema(df, contract=contract, on_unexpected="warn")
    assert rep.passed  # warn → no error
    msgs = [f.check_id for f in rep.findings]
    assert any("contract.unexpected" in m for m in msgs)


def test_unexpected_column_fail(contract):
    df = pd.DataFrame({"id": [1], "email": ["x@y.com"], "age": [30.0], "extra": ["??"]})
    rep = diff_schema(df, contract=contract, on_unexpected="fail")
    assert not rep.passed
    assert rep.n_errors > 0


# ── missing column ───────────────────────────────────────────────────────────────


def test_missing_column_fail_raises_on_gate(contract):
    df = pd.DataFrame({"id": [1], "email": ["x@y.com"]})  # age missing
    with pytest.raises(ContractViolation) as exc_info:
        clean(df, contract=contract, on_missing="fail")
    assert isinstance(exc_info.value.report, DriftReport)


def test_missing_column_warn_does_not_raise(contract):
    df = pd.DataFrame({"id": [1], "email": ["x@y.com"]})  # age missing
    rep = diff_schema(df, contract=contract, on_missing="warn")
    assert rep.passed  # warn → no error, still passes


# ── dtype drift ──────────────────────────────────────────────────────────────────


def test_dtype_drift_detected(contract):
    # id declared as int64 but supplied as object
    df = pd.DataFrame({"id": ["A", "B"], "email": ["a@b.com", "c@d.com"], "age": [25.0, None]})
    rep = diff_schema(df, contract=contract)
    ids = [f.check_id for f in rep.findings]
    assert any("dtype" in fid for fid in ids)


# ── to_dict round-trip ───────────────────────────────────────────────────────────


def test_to_dict_is_json_friendly(clean_df, contract):
    d = diff_schema(clean_df, contract=contract).to_dict()
    json.dumps(d)  # must not raise


def test_to_json_returns_string(clean_df, contract):
    j = diff_schema(clean_df, contract=contract).to_json()
    assert isinstance(j, str)
    assert "passed" in j
