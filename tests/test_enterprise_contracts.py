"""Tests for schema-drift & data-contract monitoring (Feature 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.config import DriftConfig
from freshdata.enterprise.contracts import (
    ColumnContract,
    DataContract,
    build_baseline,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)


@pytest.fixture
def trusted_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(300),
            "age": rng.normal(40, 8, 300).round(1),
            "country": rng.choice(["US", "GB", "FR"], 300, p=[0.6, 0.3, 0.1]),
            "email": [f"user{i}@example.com" for i in range(300)],
            "signup": pd.date_range("2020-01-01", periods=300, freq="D"),
        }
    )


def test_build_save_load_round_trip(trusted_df, tmp_path):
    base = build_baseline(trusted_df, name="customers", version="2.0.0")
    path = tmp_path / "customers.baseline.json"
    save_baseline(base, path)
    loaded = load_baseline(path)

    assert loaded.name == "customers"
    assert loaded.version == "2.0.0"
    assert loaded.row_count == 300
    assert set(loaded.columns) == set(trusted_df.columns)
    assert loaded.column_order == tuple(trusted_df.columns)
    assert loaded.columns["age"].kind == "numeric"
    assert loaded.columns["country"].kind == "categorical"
    assert loaded.columns["signup"].kind == "datetime"
    # schema version is embedded
    raw = path.read_text()
    assert "freshdata-baseline-v1" in raw


def test_baseline_does_not_store_raw_samples_by_default(trusted_df, tmp_path):
    base = build_baseline(trusted_df, name="customers")
    path = tmp_path / "b.json"
    save_baseline(base, path)
    raw = path.read_text()
    # No raw emails / sample values leak into the persisted baseline.
    assert "user0@example.com" not in raw
    assert base.columns["email"].sample_values == ()
    assert base.columns["country"].metadata.get("labels_hashed") is True


def test_include_samples_opt_in_stores_values(trusted_df):
    base = build_baseline(trusted_df, name="c", include_samples=True)
    assert base.columns["country"].sample_values  # raw values now present
    assert base.columns["country"].metadata.get("labels_hashed") is False
    assert any(v in ("US", "GB", "FR") for v in base.columns["country"].top_values)


def test_no_input_mutation(trusted_df):
    before = trusted_df.copy()
    base = build_baseline(trusted_df, name="c")
    compare_to_baseline(trusted_df, base)
    pd.testing.assert_frame_equal(trusted_df, before)


def test_identical_frame_passes(trusted_df):
    base = build_baseline(trusted_df, name="c")
    report = compare_to_baseline(trusted_df, base)
    assert report.passed
    assert report.n_errors == 0


def test_missing_required_column_fails(trusted_df):
    base = build_baseline(trusted_df, name="c")
    report = compare_to_baseline(trusted_df.drop(columns=["age"]), base)
    assert not report.passed
    assert any(f.check_id == "schema.removed_column" for f in report.errors)


def test_extra_column_warns_by_default(trusted_df):
    base = build_baseline(trusted_df, name="c")
    df2 = trusted_df.copy()
    df2["extra"] = 1
    report = compare_to_baseline(df2, base)
    assert report.passed  # warning, not error
    assert any(f.check_id == "schema.new_column" for f in report.warnings)


def test_dtype_change_fails(trusted_df):
    base = build_baseline(trusted_df, name="c")
    df2 = trusted_df.copy()
    df2["age"] = df2["age"].astype(str)
    report = compare_to_baseline(df2, base)
    assert not report.passed
    assert any(f.check_id == "schema.dtype_change" for f in report.errors)


def test_missing_ratio_drift_warns_then_fails(trusted_df):
    base = build_baseline(trusted_df, name="c")
    warn_df = trusted_df.copy()
    warn_df.loc[: int(0.10 * len(warn_df)), "email"] = None
    warn_report = compare_to_baseline(warn_df, base)
    assert any(
        f.check_id == "stats.missing_ratio" and f.status == "warned"
        for f in warn_report.findings
    )

    fail_df = trusted_df.copy()
    fail_df.loc[: int(0.30 * len(fail_df)), "email"] = None
    fail_report = compare_to_baseline(fail_df, base)
    assert not fail_report.passed
    assert any(
        f.check_id == "stats.missing_ratio" and f.status == "failed"
        for f in fail_report.errors
    )


def test_cardinality_drift_warns(trusted_df):
    base = build_baseline(trusted_df, name="c")
    df2 = trusted_df.copy()
    df2["country"] = "US"  # collapse cardinality 3 -> 1
    report = compare_to_baseline(df2, base)
    assert any(f.check_id == "stats.cardinality" for f in report.warnings)


def test_numeric_ks_detects_shift(trusted_df):
    base = build_baseline(trusted_df, name="c")
    df2 = trusted_df.copy()
    rng = np.random.default_rng(1)
    df2["age"] = rng.normal(70, 8, len(df2))  # large mean shift
    report = compare_to_baseline(df2, base)
    ks = [f for f in report.findings if f.metric == "ks"]
    assert ks and ks[0].current_value >= DriftConfig().numeric_ks_warn


def test_psi_detects_categorical_drift(trusted_df):
    base = build_baseline(trusted_df, name="c")
    df2 = trusted_df.copy()
    rng = np.random.default_rng(2)
    df2["country"] = rng.choice(["US", "GB", "FR"], len(df2), p=[0.1, 0.2, 0.7])
    report = compare_to_baseline(df2, base)
    assert any(f.metric == "psi" and f.column == "country" for f in report.findings)


def test_trust_score_gate_fails_below_threshold(trusted_df):
    base = build_baseline(trusted_df, name="c")
    cfg = DriftConfig(trust_score_min=99.9)
    # Inject a low trust score directly to exercise the gate deterministically.
    report = compare_to_baseline(trusted_df, base, drift_config=cfg, trust_score=50.0)
    assert not report.passed
    assert any(f.check_id == "quality.trust_score" for f in report.errors)


def test_trust_score_gate_passes_when_above(trusted_df):
    base = build_baseline(trusted_df, name="c")
    cfg = DriftConfig(trust_score_min=10.0)
    report = compare_to_baseline(trusted_df, base, drift_config=cfg, trust_score=80.0)
    assert report.passed


def test_contract_nullable_violation():
    df = pd.DataFrame({"x": [1, None, 3]})
    base = build_baseline(df, name="c")
    contract = DataContract(name="c", columns=(ColumnContract(name="x", nullable=False),))
    report = compare_to_baseline(df, base, contract=contract)
    assert not report.passed
    assert any(f.check_id == "contract.nullable" for f in report.errors)


def test_contract_unique_violation():
    df = pd.DataFrame({"x": [1, 1, 2]})
    base = build_baseline(df, name="c")
    contract = DataContract(name="c", columns=(ColumnContract(name="x", unique=True),))
    report = compare_to_baseline(df, base, contract=contract)
    assert any(f.check_id == "contract.unique" for f in report.errors)


def test_contract_allowed_values_violation():
    df = pd.DataFrame({"status": ["a", "b", "z"]})
    base = build_baseline(df, name="c")
    contract = DataContract(
        name="c", columns=(ColumnContract(name="status", allowed_values=("a", "b")),)
    )
    report = compare_to_baseline(df, base, contract=contract)
    bad = [f for f in report.errors if f.check_id == "contract.allowed_values"]
    assert bad and "z" in bad[0].details["offending_sample"]


def test_contract_min_max_value_violation():
    df = pd.DataFrame({"score": [0.1, 0.5, 1.5]})
    base = build_baseline(df, name="c")
    contract = DataContract(
        name="c",
        columns=(ColumnContract(name="score", min_value=0.0, max_value=1.0),),
    )
    report = compare_to_baseline(df, base, contract=contract)
    assert any(f.check_id == "contract.max_value" for f in report.errors)


def test_contract_regex_violation():
    df = pd.DataFrame({"code": ["AB12", "CD34", "bad"]})
    base = build_baseline(df, name="c")
    contract = DataContract(
        name="c", columns=(ColumnContract(name="code", regex=r"[A-Z]{2}\d{2}"),)
    )
    report = compare_to_baseline(df, base, contract=contract)
    assert any(f.check_id == "contract.regex" for f in report.errors)


def test_contract_missing_required_and_extra_forbidden():
    df = pd.DataFrame({"a": [1], "surprise": [2]})
    base = build_baseline(df, name="c")
    contract = DataContract(
        name="c",
        columns=(ColumnContract(name="a"), ColumnContract(name="b")),
        allow_extra_columns=False,
    )
    report = compare_to_baseline(df, base, contract=contract)
    ids = {f.check_id for f in report.errors}
    assert "contract.missing_required" in ids
    assert "contract.unexpected_column" in ids


def test_monitor_contract_from_path(trusted_df, tmp_path):
    base = build_baseline(trusted_df, name="c")
    path = tmp_path / "b.json"
    save_baseline(base, path)
    assert fd.monitor_contract(trusted_df, baseline_path=path, return_report=False) is True
    report = fd.monitor_contract(trusted_df, baseline_path=path)
    assert report.passed


def test_monitor_contract_requires_a_baseline(trusted_df):
    with pytest.raises(ValueError, match="baseline"):
        fd.monitor_contract(trusted_df)


def test_report_is_json_serializable(trusted_df):
    base = build_baseline(trusted_df, name="c")
    report = compare_to_baseline(trusted_df.drop(columns=["age"]), base)
    text = report.to_json()
    assert "baseline_name" in text
    assert isinstance(report.to_dict()["findings"], list)


def test_load_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": "nope", "name": "x"}')
    with pytest.raises(ValueError, match="schema_version"):
        load_baseline(path)


def test_polars_input_supported(trusted_df):
    pl = pytest.importorskip("polars")
    pdf = pl.from_pandas(trusted_df)
    base = build_baseline(pdf, name="c")
    assert base.row_count == 300
    report = compare_to_baseline(pdf, base)
    assert report.passed
