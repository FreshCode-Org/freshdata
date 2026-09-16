"""Tests for explain_clean and infer_roles reverse-engineering APIs."""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_ONLINE_TIER1, load_online_fixture
from freshdata.engine.context import build_context
from freshdata.explain import _cell_changes


def test_infer_roles_returns_dataframe(messy):
    roles = fd.infer_roles(messy)
    assert "column" in roles.columns
    assert "role" in roles.columns
    assert len(roles) == messy.shape[1]


def test_explain_clean_summary(messy):
    explanation = fd.explain_clean(messy, strategy="balanced")
    text = explanation.summary()
    assert "freshdata explain" in text
    assert explanation.rows_before == len(messy)
    assert explanation.report is not None
    payload = explanation.to_dict()
    assert "before_stats" in payload
    assert "actions_by_step" in payload


def test_explain_clean_narratives_on_missing():
    df = pd.DataFrame({"age": [1, None, 3], "score": [10, 20, 30]})
    explanation = fd.explain_clean(df, strategy="balanced")
    assert explanation.narratives or explanation.report.actions


def test_explain_clean_profiles_wide_noop_frame_once():
    """The post-clean narrative pass must not re-profile irrelevant columns."""
    n_columns = 120
    df = pd.DataFrame({
        f"feature_{col}": [f"value-{row}-{col}" for row in range(40)]
        for col in range(n_columns)
    })

    with patch("freshdata.engine.context.build_context", wraps=build_context) as mock:
        explanation = fd.explain_clean(df, strategy="balanced", verbose=False)

    assert explanation.narratives == []
    assert mock.call_count == n_columns


@pytest.mark.parametrize("name", ALL_ONLINE_TIER1[:3])
def test_explain_clean_on_online_fixtures(name):
    df = load_online_fixture(name)
    explanation = fd.explain_clean(df, strategy="balanced")
    assert explanation.cols_before == df.shape[1]
    assert explanation.after_stats


def test_cell_changes_aligns_on_index_after_row_removal():
    """Regression for #30: dropping rows must not mark every surviving cell
    in every column as changed."""
    before = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"]})
    after = before.drop_duplicates()  # keeps index labels 0, 2, 3
    assert _cell_changes(before, after) == {"a": 0, "b": 0}


def test_cell_changes_counts_real_edits_on_surviving_rows():
    before = pd.DataFrame({"a": [1, 1, 2, 3]})
    after = before.drop_duplicates().copy()
    after.loc[2, "a"] = 99
    assert _cell_changes(before, after) == {"a": 1}


def test_cell_changes_missing_on_both_sides_is_unchanged():
    before = pd.DataFrame({"a": [None, 1.0, 2.0]})
    after = pd.DataFrame({"a": [None, 1.0, 5.0]})
    assert _cell_changes(before, after) == {"a": 1}


def test_cell_changes_value_to_missing_transition_counts():
    before = pd.DataFrame({"a": pd.array([1, 2, 3], dtype="Int64")})
    after = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64")})
    assert _cell_changes(before, after) == {"a": 1}


def test_cell_changes_dtype_conversion_counts_value_diffs_only():
    """A dtype change alone (int64 -> Int64) is not a cell change."""
    before = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    after = pd.DataFrame({"a": pd.array([1, 2, 99], dtype="Int64")})
    assert _cell_changes(before, after) == {"a": 1}


def test_cell_changes_duplicate_labels_keep_conservative_count():
    """Alignment is ambiguous with duplicate index labels; the historical
    whole-column count is retained for that case."""
    before = pd.DataFrame({"a": [1, 2, 3]}, index=[0, 0, 1])
    after = pd.DataFrame({"a": [1, 2]}, index=[0, 0])
    assert _cell_changes(before, after) == {"a": 2}


def test_explain_clean_dedupe_reports_zero_cell_changes(messy=None):
    """End-to-end: duplicate removal alone must not inflate cell_changes."""
    df = pd.DataFrame(
        {"a": [1, 1, 2, 3], "b": [10.0, 10.0, 20.0, 30.0]}
    )
    rep = fd.explain_clean(df, drop_duplicates=True)
    assert rep.rows_after < rep.rows_before
    assert all(v == 0 for v in rep.cell_changes.values()), rep.cell_changes


def test_explain_clean_accepts_unhashable_cells():
    # Regression (#450): nunique() hashes every value, so a list cell raised
    # TypeError although fd.clean and fd.profile accept the same frame.
    df = pd.DataFrame({"payload": [[1], [1], {"k": 2}, None], "n": [1, 2, 3, 4]})
    report = fd.explain_clean(df)
    assert report is not None
    stats = report.to_dict()["before_stats"]["payload"]
    assert stats["nunique"] == 2  # [1] twice, {"k": 2} once
    assert stats["null_count"] == 1


def test_explain_clean_to_dict_is_json_serializable_for_empty_and_infinite():
    # Regression (#460): a zero-row column gave null_pct NaN, and ±inf bounds
    # are not JSON either, so the payload could not be serialized.
    empty = fd.explain_clean(pd.DataFrame({"a": pd.Series([], dtype="float64")}))
    payload = empty.to_dict()
    json.dumps(payload)
    assert payload["before_stats"]["a"]["null_pct"] == 0.0

    infinite = fd.explain_clean(pd.DataFrame({"a": [float("inf"), 1.0, 2.0]}))
    json.dumps(infinite.to_dict())


def test_explain_clean_keeps_finite_bounds():
    report = fd.explain_clean(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    stats = report.to_dict()["before_stats"]["a"]
    assert stats["min"] == 1.0 and stats["max"] == 3.0
