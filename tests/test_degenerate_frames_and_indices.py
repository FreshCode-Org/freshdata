# tests/test_degenerate_frames_and_indices.py
from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.config import CleanConfig


@pytest.mark.parametrize(
    "df_factory",
    [
        lambda: pd.DataFrame(),  # (0, 0)
        lambda: pd.DataFrame(index=pd.RangeIndex(5, name="row_id")),  # (5, 0)
        lambda: pd.DataFrame(columns=["a", "b"], dtype="Int64"),  # (0, 2)
        lambda: pd.DataFrame({"a": pd.Series([10], dtype="Int64"), "b": pd.Series([" x "], dtype="string")}, index=pd.Index([99], name="custom_idx")),  # (1, 2)
        lambda: pd.DataFrame(
            {"val": pd.Series([1, 1, 2, 3], dtype="Int64"), "txt": pd.Series([" a ", " a ", "b", "c"], dtype="string")},
            index=pd.Index([100, 200, 50, 10], name="non_monotonic_id"),
        ),
        lambda: pd.DataFrame(
            {"val": pd.Series([10, 10, 20], dtype="Int64")},
            index=pd.MultiIndex.from_tuples([("eu", 1), ("eu", 2), ("us", 1)], names=["region", "seq"]),
        ),
    ],
)
def test_degenerate_shapes_and_indices_preserved_across_engines(df_factory) -> None:
    """Fixes #463 & #456: Preserve (0,0), (5,0), (0,5), single-row, MultiIndex, and non-monotonic indices."""
    df = df_factory()
    cfg = CleanConfig(strategy="conservative", fix_dtypes=False, drop_duplicates=True, duplicate_keep="first")

    res_pd, rep_pd = fd.clean(df, config=cfg, engine="pandas", return_report=True)
    res_pl, rep_pl = fd.clean(df, config=cfg, engine="polars", return_report=True)
    res_ddb, rep_ddb = fd.clean(df, config=cfg, engine="duckdb", return_report=True)

    # Baseline check against pandas manual strip + drop_duplicates
    expected = df.copy()
    for col in expected.columns:
        if isinstance(expected[col].dtype, pd.StringDtype) or expected[col].dtype == object:
            expected[col] = expected[col].str.strip()
    if expected.shape[1] > 0 and expected.shape[0] > 0:
        expected = expected.drop_duplicates(keep="first")

    pd.testing.assert_frame_equal(res_pd, expected, check_dtype=True, check_index_type=True)
    pd.testing.assert_frame_equal(res_pd, res_pl, check_dtype=True, check_index_type=True)
    pd.testing.assert_frame_equal(res_pd, res_ddb, check_dtype=True, check_index_type=True)
    assert rep_pd.duplicates_removed == rep_pl.duplicates_removed == rep_ddb.duplicates_removed
