"""Report export completeness and enterprise lineage validity (Metric 9)."""

from __future__ import annotations

import json
import os
import tempfile

import freshdata as fd
import pytest

import harness_metrics as hm
from fixtures import FRAME_FIXTURES


ALL_FIXTURES = FRAME_FIXTURES + ("gold",)


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_export_completeness_is_full(name):
    df = hm.make_frame(name, 2_000, seed=42)
    config = hm.config_for(name, df)
    _cleaned, report = fd.clean(df, config=config, return_report=True)
    result = hm.metric_export_completeness(name, df, config, report)
    assert result["export_completeness_pct"] == 100.0, (name, result["fields_missing"])
    assert result["fields_missing"] == []


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_report_methods_non_empty(name):
    df = hm.make_frame(name, 2_000, seed=42)
    _cleaned, report = fd.clean(df, config=hm.config_for(name, df), return_report=True)
    assert isinstance(report.summary(), str) and report.summary()
    assert len(report.to_frame()) > 0
    assert isinstance(report.to_dict(), dict) and report.to_dict()


@pytest.mark.parametrize("name", ALL_FIXTURES)
def test_repaired_actions_carry_before_after_risk_confidence(name):
    df = hm.make_frame(name, 2_000, seed=42)
    _cleaned, report = fd.clean(df, config=hm.config_for(name, df), return_report=True)
    repaired = [a for a in report if getattr(a, "count", 0)]
    assert repaired, "expected at least one repair action"
    for a in repaired:
        assert a.description, (name, a.step)
        assert a.risk in {"low", "medium", "high"}
        assert isinstance(a.confidence, float)


def test_enterprise_lineage_emits_valid_json():
    df = hm.make_frame("crm", 2_000, seed=42)
    res = fd.clean_enterprise(df, clean_config=hm.config_for("crm", df))
    assert len(res.quality.to_markdown()) > 0
    path = os.path.join(tempfile.gettempdir(), "fd_test_lineage.json")
    res.lineage.emit(path)
    with open(path) as fh:
        payload = json.load(fh)
    assert isinstance(payload, (dict, list))
    os.remove(path)
