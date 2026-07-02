"""Tests for explain_clean reverse-engineering API."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_ONLINE_TIER1, load_online_fixture


def test_explain_clean_summary(messy):
    explanation = fd.explain_clean(messy, {"strategy": "balanced"})
    text = explanation.summary()
    assert "freshdata explain" in text
    assert explanation.rows_before == len(messy)
    assert explanation.report is not None
    payload = explanation.to_dict()
    assert "before_stats" in payload
    assert "actions_by_step" in payload


def test_explain_clean_narratives_on_missing():
    df = pd.DataFrame({"age": [1, None, 3], "score": [10, 20, 30]})
    explanation = fd.explain_clean(df, {"strategy": "balanced"})
    assert explanation.narratives or explanation.report.actions


@pytest.mark.parametrize("name", ALL_ONLINE_TIER1[:3])
def test_explain_clean_on_online_fixtures(name):
    df = load_online_fixture(name)
    explanation = fd.explain_clean(df, {"strategy": "balanced"})
    assert explanation.cols_before == df.shape[1]
    assert explanation.after_stats
