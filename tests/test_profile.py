import json

import pandas as pd
import pytest

import freshdata as fd


def test_profile_include_plan_attaches_clean_plan(messy):
    p = fd.profile(messy, include_plan=True)
    assert hasattr(p, "plan")
    assert isinstance(p.plan, fd.CleanPlan)
    assert p.plan.config.strategy == "balanced"


def test_profile_never_modifies_input(messy):
    snapshot = messy.copy(deep=True)
    fd.profile(messy)
    pd.testing.assert_frame_equal(messy, snapshot)


def test_table_level_stats(messy):
    p = fd.profile(messy)
    assert p.n_rows == 5
    assert p.n_cols == 6
    assert p.duplicate_rows == 1
    assert p.missing_cells == 6
    assert p.memory > 0


def test_suggestions_match_what_clean_does(messy):
    p = fd.profile(messy)
    cleaned = fd.clean(messy)
    suggested = {c.name: c.suggested_dtype for c in p.columns if c.suggested_dtype}
    assert suggested["AGE"] == str(cleaned["age"].dtype)
    assert suggested["Salary($)"] == str(cleaned["salary"].dtype)
    assert suggested["Joined Date"] == str(cleaned["joined_date"].dtype)
    assert suggested["Active"] == str(cleaned["active"].dtype)


def test_issue_detection(messy):
    p = fd.profile(messy)
    by_name = {c.name: c for c in p.columns}
    assert any("whitespace" in i for i in by_name[" First Name "].issues)
    assert any("sentinel" in i for i in by_name["AGE"].issues)
    assert any("constant" in i for i in by_name["empty"].issues)
    assert by_name["empty"].missing == 5


def test_outlier_issue_for_numeric_columns():
    df = pd.DataFrame({"v": [10.0, 11.0, 12.0] * 7 + [9999.0]})
    p = fd.profile(df)
    [col] = p.columns
    assert any("outlier" in i for i in col.issues)


def test_identifier_issue():
    df = pd.DataFrame({"id": [f"u{i}" for i in range(25)]})
    p = fd.profile(df)
    assert any("identifier" in i for i in p.columns[0].issues)


def test_to_frame_shape(messy):
    frame = fd.profile(messy).to_frame()
    assert len(frame) == 6
    assert frame.index.name == "column"
    assert "suggested_dtype" in frame.columns


def test_to_dict_serializable(messy):
    payload = fd.profile(messy).to_dict()
    assert json.dumps(payload)  # all plain Python types
    assert payload["n_rows"] == 5


def test_str_renders_a_table(messy):
    text = str(fd.profile(messy))
    assert "freshdata profile" in text
    assert "AGE" in text
    assert "would convert to" in text


def test_unhashable_rows_give_none_duplicates():
    df = pd.DataFrame({"v": [[1], [2]], "w": [1, 2]})
    p = fd.profile(df)
    assert p.duplicate_rows is None  # multi-column duplicated() cannot hash lists
    assert p.columns[0].unique is None


def test_profile_rejects_non_dataframe():
    with pytest.raises(TypeError):
        fd.profile("not a frame")


def test_empty_frame_profile():
    p = fd.profile(pd.DataFrame())
    assert p.n_rows == 0 and p.n_cols == 0
    assert str(p)  # renders without crashing
