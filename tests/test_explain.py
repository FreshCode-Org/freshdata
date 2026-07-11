"""Tests for explain_clean and infer_roles reverse-engineering APIs."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_ONLINE_TIER1, load_online_fixture
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
    rep = fd.explain_clean(df)
    assert rep.rows_after < rep.rows_before
    assert all(v == 0 for v in rep.cell_changes.values()), rep.cell_changes
