"""End-to-end tests for the unified enterprise interface."""

import json

import pandas as pd
import pytest

from freshdata.adapters.polars import is_polars_frame
from freshdata.enterprise import (
    ClusterConfig,
    EnterpriseConfig,
    FreshDataEnterprise,
    MaskingRule,
    SemanticValidatorConfig,
    clean_enterprise,
)


def _full_config(**overrides):
    base = {
        "enable_clustering": True,
        "clustering": ClusterConfig(columns=("vendor",)),
        "masking": (MaskingRule(name="pii", columns=("email",), strategy="hash", salt="s"),),
        "semantic": (
            SemanticValidatorConfig(
                name="iso", kind="reference", columns=("country",), reference=("US", "CA")
            ),
        ),
    }
    base.update(overrides)
    return EnterpriseConfig(**base)


@pytest.fixture
def raw():
    return pd.DataFrame({
        "vendor": ["Acme Inc", "ACME  inc", "acme inc", "Globex", "globex"],
        "email": ["a@x.com", "b@y.io", "c@z.org", "d@w.net", "e@e.com"],
        "country": ["US", "us", "XX", "CA", "CA"],
    })


def test_pandas_in_pandas_out(raw):
    result = clean_enterprise(raw, enterprise=_full_config(), verbose=False)
    assert isinstance(result.data, pd.DataFrame)
    assert result.data["vendor"].nunique() == 2          # clustered
    assert result.data["email"].iloc[0] != "a@x.com"     # masked
    assert result.validation_report.columns["country"].n_invalid == 1
    assert result.cells_merged == 3


def test_polars_in_polars_out(raw):
    pl = pytest.importorskip("polars")
    result = clean_enterprise(pl.from_pandas(raw), enterprise=_full_config(), verbose=False)
    assert is_polars_frame(result.data)
    assert len(set(result.data["vendor"].to_list())) == 2


def test_gate_pass_and_fail():
    # Two columns so the missing cell's row is not all-empty (which would be dropped).
    df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
    passing = clean_enterprise(
        df, enterprise=EnterpriseConfig(fail_under_trust=10.0), verbose=False
    )
    assert passing.passed_gate is True
    failing = clean_enterprise(
        df,
        enterprise=EnterpriseConfig(fail_under_trust=100.0),
        strategy="conservative",
        verbose=False,
    )
    assert failing.passed_gate is False  # conservative leaves the NaN -> trust < 100


def test_result_serialization_and_render(raw):
    result = clean_enterprise(raw, enterprise=_full_config(fail_under_trust=10.0), verbose=False)
    payload = result.to_dict()
    assert set(payload) >= {"trust_before", "trust_after", "clusters", "masking", "validation"}
    assert json.loads(result.to_json())
    md = result.to_markdown()
    assert "# freshdata Quality Report" in md
    assert "## Clustering" in md and "## PII masking" in md and "## Semantic validation" in md
    summary = result.summary()
    assert "trust" in summary and "gate: PASS" in summary
    assert repr(result).startswith("<EnterpriseResult")
    assert str(result) == summary


def test_lineage_records_each_stage(raw):
    result = clean_enterprise(raw, enterprise=_full_config(), actor="alice", verbose=False)
    rules = [e.rule_applied for e in result.lineage.events]
    assert rules == ["core_clean", "cluster_merge", "pii_mask"]
    assert all(e.who == "alice" for e in result.lineage.events)


def test_lineage_can_be_disabled(raw):
    result = clean_enterprise(raw, enterprise=_full_config(enable_lineage=False), verbose=False)
    assert result.lineage.events == []


def test_toggles_default_off_for_optional_stages(raw):
    # Default EnterpriseConfig: masking/clustering empty, validation list empty.
    result = clean_enterprise(raw, enterprise=EnterpriseConfig(), verbose=False)
    assert result.cluster_results == []
    assert result.mask_report is None
    assert result.validation_report is None
    # core clean still ran and was tracked
    assert any(e.rule_applied == "core_clean" for e in result.lineage.events)


def test_clean_options_forwarded_and_validated(raw):
    # strategy forwarded to CleanConfig
    result = clean_enterprise(raw, enterprise=EnterpriseConfig(), strategy="conservative",
                              verbose=False)
    assert result.trust_after.overall >= 0
    # unknown option rejected by CleanConfig.merge_options
    with pytest.raises(TypeError):
        clean_enterprise(raw, totally_unknown_option=1)


def test_freshdata_enterprise_reuse(raw):
    pipe = FreshDataEnterprise(enterprise=_full_config(), strategy="balanced")
    result = pipe.run(raw, actor="bob")
    assert pipe.result_ is result
    assert result.lineage.events[0].who == "bob"
    assert repr(pipe).startswith("<FreshDataEnterprise")
