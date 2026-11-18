import pandas as pd
import pytest
from freshdata._core.models import CleanConfig, ColumnRule
from freshdata._core.stages import (
    stage_column_names,
    stage_strings,
    stage_dtypes,
    stage_duplicates,
    stage_missing,
    stage_outliers,
)

def test_stage_column_names_pandas():
    df = pd.DataFrame({" First Name ": ["Alice"], "Spend ($)": [100]})
    cfg = CleanConfig(normalize_names=True)
    out_df, actions = stage_column_names(df, cfg)
    assert list(out_df.columns) == ["first_name", "spend"]
    assert len(actions) == 2

def test_stage_strings_pandas():
    df = pd.DataFrame({"city": [" New York  ", "", "  ", "Chicago"]})
    cfg = CleanConfig(strip_strings=True)
    out_df, actions = stage_strings(df, cfg)
    assert [None if pd.isna(x) else x for x in out_df["city"].tolist()] == ["New York", None, None, "Chicago"]
    assert any(a.action == "strip_strings" for a in actions)

def test_stage_dtypes_currency_pandas():
    df = pd.DataFrame({"price": ["$1,200.50", "350.00", "$45.99"]})
    cfg = CleanConfig(fix_dtypes=True)
    out_df, actions = stage_dtypes(df, cfg)
    assert pd.api.types.is_numeric_dtype(out_df["price"])
    assert out_df["price"].iloc[0] == 1200.50

def test_stage_duplicates_pandas():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    out_drop, act_drop = stage_duplicates(df, CleanConfig(drop_duplicates=True))
    assert len(out_drop) == 2
    out_keep, act_keep = stage_duplicates(df, CleanConfig(drop_duplicates=False))
    assert len(out_keep) == 3

def test_stage_missing_protected():
    df = pd.DataFrame({"target": [1, None, 3], "val": [10.0, None, 30.0]})
    cfg = CleanConfig(
        target_column="target",
        rules={"val": ColumnRule(impute="median")},
    )
    out_df, actions = stage_missing(df, cfg)
    assert out_df["target"].isna().sum() == 1  # target untouched
    assert out_df["val"].isna().sum() == 0     # val imputed to 20.0
    assert out_df["val"].iloc[1] == 20.0

def test_stage_outliers_clip():
    df = pd.DataFrame({"x": [10, 11, 12, 10, 11, 100]})
    cfg = CleanConfig(
        rules={"x": ColumnRule(outliers="clip")},
        iqr_factor=1.5,
    )
    out_df, actions = stage_outliers(df, cfg)
    assert out_df["x"].max() < 100

def test_stage_outliers_short_series():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    cfg = CleanConfig(rules={"x": ColumnRule(outliers="clip")})
    out_df, actions = stage_outliers(df, cfg)
    assert len(actions) == 0
    assert out_df["x"].tolist() == [1.0, 2.0]


def test_row_filters_keep_extension_dtypes():
    import freshdata as fd

    df = pd.DataFrame({
        "n": pd.array([1, 1, None, 2, 3, 4], dtype="Int64"),
        "s": pd.array(["a", "a", "b", None, "c", "d"], dtype="string"),
        "c": pd.Categorical(["x", "x", "y", "y", "x", "y"]),
        "t": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "v": [1.0, 1.0, 2.0, 3.0, 4.0, 1000.0],
    })
    cfg = CleanConfig(
        normalize_names=False, strip_strings=False, fix_dtypes=False, drop_duplicates=True,
        default_impute="preserve",
        rules={"n": ColumnRule(impute="drop"), "v": ColumnRule(outliers="drop")},
    )
    out, report = fd.clean(df, cfg, return_report=True)
    assert {a.action for a in report.actions} == {"drop_duplicates", "drop_nulls", "drop_outliers"}
    assert out.dtypes.to_dict() == df.dtypes.to_dict()
    assert out["n"].tolist() == [1, 2, 3]
    assert list(out.index) == [0, 1, 2]

NAN = float("nan")

@pytest.mark.parametrize(("strategy", "values", "expected", "actions"), [
    ("median", [1.0, 2.0, 3.0, 10.0], [1.0, 2.0, 3.0, 10.0], []),
    ("median", [1.0, None, 3.0, 10.0], [1.0, 3.0, 3.0, 10.0], ["impute_median"]),
    ("median", [None] * 4, [NAN] * 4, []),
    ("mean", [1.0, None, 3.0, 10.0], [1.0, 14 / 3, 3.0, 10.0], ["impute_mean"]),
    ("mean", [None] * 4, [NAN] * 4, []),
    ("mode", [1.0, None, 3.0, 10.0], [1.0, 1.0, 3.0, 10.0], ["impute_mode"]),
    ("mode", [None] * 4, [NAN] * 4, []),
    ("drop", [1.0, 2.0, 3.0, 10.0], [1.0, 2.0, 3.0, 10.0], []),
    ("drop", [1.0, None, 3.0, 10.0], [1.0, 3.0, 10.0], ["drop_nulls"]),
    ("drop", [None] * 4, [], ["drop_nulls"]),
    ("drop", [], [], ["drop_nulls"]),
    ("preserve", [1.0, None, 3.0, 10.0], [1.0, NAN, 3.0, 10.0], []),
])
def test_stage_missing_strategies(strategy, values, expected, actions):
    df = pd.DataFrame({"x": pd.Series(values, dtype="float64")})
    out, acts = stage_missing(df, CleanConfig(rules={"x": ColumnRule(impute=strategy)}))
    assert out["x"].tolist() == pytest.approx(expected, nan_ok=True)
    assert [a.action for a in acts] == actions

@pytest.mark.parametrize(("strategy", "values", "expected", "actions"), [
    ("clip", [10.0, 11.0, 12.0, 10.0, 11.0, 100.0], [10.0, 11.0, 12.0, 10.0, 11.0, 14.0], ["clip_iqr"]),
    ("clip", [10.0, 11.0, None, 12.0, 10.0, 11.0, 100.0], [10.0, 11.0, NAN, 12.0, 10.0, 11.0, 14.0], ["clip_iqr"]),
    ("clip", [None] * 6, [NAN] * 6, []),
    ("drop", [10.0, 11.0, 12.0, 10.0, 11.0, 100.0], [10.0, 11.0, 12.0, 10.0, 11.0], ["drop_outliers"]),
    ("drop", [10.0, 11.0, None, 12.0, 10.0, 11.0, 100.0], [10.0, 11.0, NAN, 12.0, 10.0, 11.0], ["drop_outliers"]),
    ("drop", [None] * 6, [NAN] * 6, []),
    ("preserve", [10.0, 11.0, None, 12.0, 10.0, 11.0, 100.0], [10.0, 11.0, NAN, 12.0, 10.0, 11.0, 100.0], []),
])
def test_stage_outliers_strategies(strategy, values, expected, actions):
    df = pd.DataFrame({"x": pd.Series(values, dtype="float64")})
    out, acts = stage_outliers(df, CleanConfig(rules={"x": ColumnRule(outliers=strategy)}))
    assert out["x"].tolist() == pytest.approx(expected, nan_ok=True)
    assert [a.action for a in acts] == actions
