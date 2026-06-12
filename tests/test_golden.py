"""Golden report snapshot tests (balanced strategy).

Update snapshots after intentional behavior changes::

    pytest tests/test_golden.py --update-golden
"""

from __future__ import annotations

import json

import pytest

import freshdata as fd
from expectations import ALL_FIXTURES, load_fixture
from golden_util import load_golden, normalize_report, write_golden


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_balanced_report_golden_snapshot(fixture_name, update_golden):
    df = load_fixture(fixture_name)
    _, report = fd.clean(df, return_report=True, verbose=False)
    actual = normalize_report(report)

    if update_golden:
        path = write_golden(fixture_name, report, strategy="balanced")
        pytest.skip(f"updated golden snapshot: {path}")

    expected = load_golden(fixture_name, strategy="balanced")
    assert actual == expected, (
        f"golden snapshot mismatch for {fixture_name!r}; "
        "run pytest tests/test_golden.py --update-golden to refresh"
    )


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_balanced_report_structure(fixture_name):
    df = load_fixture(fixture_name)
    _, report = fd.clean(df, return_report=True, verbose=False)
    payload = normalize_report(report)
    for key in ("missing_before", "missing_after", "actions", "warnings"):
        assert key in payload
    assert json.dumps(payload)


@pytest.mark.parametrize("fixture_name", ["aqi_sample", "fraud_signals"])
def test_balanced_report_has_engine_actions(fixture_name):
    df = load_fixture(fixture_name)
    _, report = fd.clean(df, return_report=True, verbose=False)
    steps = {a.step for a in report}
    assert steps & {"missing", "outliers", "fix_dtypes", "strip_whitespace"}
