"""Tests for the framework-agnostic trust-gate core."""

from __future__ import annotations

import logging
import sys

import pytest

from freshdata import CleanConfig
from freshdata.integrations import TrustGateResult, evaluate_trust_gate


def test_gate_passes_at_zero_threshold(sample_df):
    cleaned, result = evaluate_trust_gate(sample_df, trust_score_threshold=0.0)
    assert isinstance(result, TrustGateResult)
    assert result.passed is True
    assert 0.0 <= result.trust_score <= 100.0
    assert isinstance(result.high_risk_count, int)
    assert result.row_count_in == len(sample_df)
    assert result.row_count_out == len(cleaned)
    assert result.should_fail is False
    assert result.should_skip is False


def test_gate_fails_high_threshold_with_fail(sample_df):
    _, result = evaluate_trust_gate(sample_df, trust_score_threshold=999.0, on_low_score="fail")
    assert result.passed is False
    assert result.should_fail is True
    assert result.should_skip is False
    assert "failed" in result.message


def test_skip_flag_set(sample_df):
    _, result = evaluate_trust_gate(sample_df, trust_score_threshold=999.0, on_low_score="skip")
    assert result.should_skip is True
    assert result.should_fail is False


def test_warn_does_not_raise(sample_df):
    # The core never raises; warn just logs and returns.
    _, result = evaluate_trust_gate(sample_df, trust_score_threshold=999.0, on_low_score="warn")
    assert result.passed is False
    assert result.should_fail is False
    assert result.should_skip is False


@pytest.mark.parametrize("invalid_policy", ["Fail", "error", ""])
def test_invalid_low_score_policy_raises_before_cleaning(sample_df, invalid_policy):
    with pytest.raises(ValueError, match="on_low_score must be one of"):
        evaluate_trust_gate(sample_df, on_low_score=invalid_policy)


def test_as_metadata_keys(sample_df):
    _, result = evaluate_trust_gate(sample_df, trust_score_threshold=0.0)
    meta = result.as_metadata()
    assert meta["freshdata/passed"] is True
    for key in (
        "freshdata/trust_score",
        "freshdata/grade",
        "freshdata/threshold",
        "freshdata/high_risk_count",
        "freshdata/row_count_in",
        "freshdata/row_count_out",
        "freshdata/generated_utc",
    ):
        assert key in meta


def test_publish_full_report_attaches_clean_report(sample_df):
    _, result = evaluate_trust_gate(
        sample_df, trust_score_threshold=0.0, publish_full_report=True
    )
    assert result.report_dict is not None
    assert "clean_report" in result.report_dict
    assert result.to_dict()["report"]["clean_report"]


def test_publish_includes_compliance_when_available(sample_df):
    pytest.importorskip("freshdata.compliance")
    _, result = evaluate_trust_gate(
        sample_df, trust_score_threshold=0.0, publish_full_report=True
    )
    assert "compliance" in result.report_dict


def test_compliance_is_optional_noop_when_absent(sample_df, monkeypatch):
    # Simulate compliance not being installed: the import fails -> graceful no-op.
    monkeypatch.setitem(sys.modules, "freshdata.compliance", None)
    _, result = evaluate_trust_gate(
        sample_df, trust_score_threshold=0.0, publish_full_report=True
    )
    assert "clean_report" in result.report_dict
    assert "compliance" not in result.report_dict


def test_no_report_dict_without_publish(sample_df):
    _, result = evaluate_trust_gate(sample_df, trust_score_threshold=0.0)
    assert result.report_dict is None
    assert "report" not in result.to_dict()


def test_gate_without_config_keeps_stdout_clean_and_logs_warnings(sample_df, capsys, caplog):
    """The gate must not print: ``dbt-gate`` promises JSON-only stdout."""
    caplog.set_level(logging.WARNING, logger="freshdata.integrations")
    _, result = evaluate_trust_gate(
        sample_df, trust_score_threshold=0.0, publish_full_report=True
    )
    assert capsys.readouterr().out == ""
    warnings = result.report_dict["clean_report"]["warnings"]
    assert warnings  # sample_df's duplicate row exceeds the duplicate threshold
    for warning in warnings:
        assert f"freshdata: {warning}" in caplog.messages


def test_gate_respects_explicit_verbose_config(sample_df, capsys):
    evaluate_trust_gate(sample_df, clean_config=CleanConfig(verbose=True))
    assert "freshdata: rows" in capsys.readouterr().out
