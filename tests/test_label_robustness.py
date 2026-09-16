"""Column labels that pandas itself handles awkwardly: missing, coerced, repeated.

Regressions for #459 (suggest_plan on duplicate labels), #461 (numeric-or-None
labels), #462 (label identity in infer_roles) and #437 (the text entry points).
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import freshdata as fd

warnings.simplefilter("ignore")


# ── #461: a NaN column label must not break duplicate detection ────────────────


def _nan_label_frame() -> pd.DataFrame:
    # pandas coerces Index([0, None]) to float64, so the second label is NaN and
    # cannot be looked up by value — DataFrame.duplicated() raised KeyError.
    return pd.DataFrame({0: [1, 2, 1], None: [3, 4, 3]})


@pytest.mark.parametrize("call", [
    lambda df: fd.clean(df, verbose=False),
    fd.profile,
    fd.infer_roles,
    fd.explain_clean,
])
def test_numeric_or_none_labels_are_accepted(call):
    assert call(_nan_label_frame()) is not None


def test_duplicate_rows_are_still_detected_with_a_nan_label():
    _, report = fd.clean(
        _nan_label_frame(), drop_duplicates=True, return_report=True, verbose=False
    )
    assert any(a.step == "drop_duplicates" and a.count == 1 for a in report.actions)


def test_duplicate_subset_still_applies_with_a_nan_label():
    df = pd.DataFrame({0: [1, 1, 2], None: [9, 8, 7]})
    out = fd.clean(
        df, drop_duplicates=True, duplicate_subset=[0.0], verbose=False
    )
    assert len(out) == 2  # deduplicated on the first column only


# ── #462: infer_roles reports the labels the frame actually has ────────────────


@pytest.mark.parametrize("labels", [[0, None], [-2, 0.78], ["a", 1]])
def test_infer_roles_keeps_label_identity(labels):
    df = pd.DataFrame([[1, 2], [3, 4]])
    df.columns = pd.Index(labels, dtype=object)
    reported = fd.infer_roles(df)["column"].tolist()
    assert reported == sorted(labels, key=str)  # rows are ordered by label text
    for label in reported:
        assert df[label].shape == (2,)  # the documented round-trip


# ── #459 / #437: duplicate labels raise the same error everywhere ──────────────


@pytest.mark.parametrize("call", [
    fd.suggest_plan,
    fd.plan,
    fd.clean_text,
    fd.lint_text_encoding,
    fd.infer_roles,
])
def test_duplicate_labels_raise_value_error(call):
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
    with pytest.raises(ValueError, match="requires unique column labels"):
        call(df)
    assert df.columns.tolist() == ["a", "a"]  # never modified


def test_unique_labels_still_work_on_those_entry_points():
    df = pd.DataFrame({"a": ["  x  ", "y"], "n": [1, 2]})
    assert fd.suggest_plan(df) is not None
    assert fd.plan(df) is not None
    cleaned, _ = fd.clean_text(df)
    assert cleaned["a"].tolist() == ["x", "y"]
    assert fd.lint_text_encoding(df) is not None
