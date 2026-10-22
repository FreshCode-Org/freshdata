# tests/test_property_invariants.py
from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, given, settings, strategies as st

import freshdata as fd
from freshdata.config import CleanConfig


@st.composite
def dirty_dataframe_strategy(draw: st.DrawFn) -> pd.DataFrame:
    n_rows = draw(st.integers(min_value=1, max_value=12))
    int_vals = draw(st.lists(st.one_of(st.integers(-50, 50), st.none()), min_size=n_rows, max_size=n_rows))
    str_vals = draw(
        st.lists(
            st.sampled_from([" alpha ", "beta", "N/A", "null", "", "gamma ", None]),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    bool_vals = draw(st.lists(st.one_of(st.booleans(), st.none()), min_size=n_rows, max_size=n_rows))
    id_vals = list(range(1000, 1000 + n_rows))

    return pd.DataFrame({
        "user_id": pd.Series(id_vals, dtype="Int64"),
        "int_col": pd.Series([pd.NA if v is None else v for v in int_vals], dtype="Int64"),
        "str_col": pd.Series([pd.NA if v is None else v for v in str_vals], dtype="string"),
        "bool_col": pd.Series([pd.NA if v is None else v for v in bool_vals], dtype="boolean"),
    })


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(df=dirty_dataframe_strategy())
def test_five_hard_invariants(df: pd.DataFrame) -> None:
    cfg = CleanConfig(
        strategy="conservative",
        fix_dtypes=False,
        drop_duplicates=True,
        impute_missing=True,
        handle_outliers=False,
    )

    # 1. Input Non-Mutation
    df_snapshot = df.copy(deep=True)
    res_pd, rep1 = fd.clean(df, config=cfg, engine="pandas", return_report=True)
    pd.testing.assert_frame_equal(df, df_snapshot, check_dtype=True)

    # 2. Idempotence: clean(clean(df)) == clean(df) & zero actions on 2nd pass
    res_pd_2, rep2 = fd.clean(res_pd, config=cfg, engine="pandas", return_report=True)
    pd.testing.assert_frame_equal(res_pd, res_pd_2, check_dtype=True)
    assert rep2.duplicates_removed == 0
    assert rep2.values_imputed == 0
    assert len(rep2.dtype_changes) == 0

    # 3. Column-Order Invariance
    rev_cols = list(reversed(df.columns))
    res_rev = fd.clean(df[rev_cols], config=cfg, engine="pandas")[df.columns]
    pd.testing.assert_frame_equal(res_pd, res_rev, check_dtype=True)

    # 4. Row-Order Invariance (up to permutation)
    perm_df = df.iloc[::-1].copy()
    res_perm = fd.clean(perm_df, config=cfg, engine="pandas")
    s1 = res_pd.sort_values(by=list(df.columns), na_position="last").reset_index(drop=True)
    s2 = res_perm.sort_values(by=list(df.columns), na_position="last").reset_index(drop=True)
    pd.testing.assert_frame_equal(s1, s2, check_dtype=True)

    # 5. Cross-Engine Equivalence (pandas == polars == duckdb)
    res_pl = fd.clean(df, config=cfg, engine="polars")
    res_ddb = fd.clean(df, config=cfg, engine="duckdb")
    pd.testing.assert_frame_equal(res_pd, res_pl, check_dtype=True)
    pd.testing.assert_frame_equal(res_pd, res_ddb, check_dtype=True)
