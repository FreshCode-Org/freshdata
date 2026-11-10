# tests/test_property_invariants.py
from __future__ import annotations

import pandas as pd
import polars as pl
from hypothesis import HealthCheck, given, settings, strategies as st

import freshdata as fd
from freshdata import CleanConfig, ColumnRule
from freshdata.polars_engine import (
    CleanConfig as PeCleanConfig,
    apply_plan as pe_apply,
    suggest_plan as pe_suggest,
)


@st.composite
def dirty_dataframe_strategy(draw: st.DrawFn) -> pd.DataFrame:
    n_rows = draw(st.integers(min_value=2, max_value=15))
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
        default_impute="preserve",
        default_outliers="preserve",
    )

    # 1. Input Non-Mutation: df must not be modified in place
    df_snapshot = df.copy(deep=True)
    res_pd, rep1 = fd.clean(df, config=cfg, return_report=True)
    pd.testing.assert_frame_equal(df, df_snapshot, check_dtype=True)

    # 2. Idempotence: clean(clean(df)) == clean(df)
    res_pd_2, rep2 = fd.clean(res_pd, config=cfg, return_report=True)
    pd.testing.assert_frame_equal(res_pd, res_pd_2, check_dtype=True)

    # 3. Column-Order Invariance: permuted input columns yield identical cleaned columns
    rev_cols = list(reversed(df.columns))
    res_rev = fd.clean(df[rev_cols], config=cfg)
    assert set(res_rev.columns) == set(res_pd.columns)
    for c in df.columns:
        pd.testing.assert_series_equal(res_pd[c], res_rev[c], check_dtype=True)

    # 4. Duplicate Invariance: no duplicate rows in output
    assert not res_pd.duplicated().any()

    # 5. Cross-Engine Equivalence: row count preservation between pandas and polars
    pl_df = pl.from_pandas(df)
    res_pl = fd.clean(pl_df, config=cfg)
    assert len(res_pl) == len(res_pd)
    assert res_pl.columns == list(res_pd.columns)


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.lists(
        st.one_of(st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False), st.none()),
        min_size=10,
        max_size=30,
    )
)
def test_numeric_imputation_and_outlier_invariants(values: list[float | None]) -> None:
    df = pd.DataFrame({"val": pd.Series(values, dtype="Float64")})
    cfg = CleanConfig(default_impute="median", default_outliers="clip")
    res_pd = fd.clean(df, config=cfg)
    has_non_null = any(v is not None for v in values)
    if has_non_null:
        assert res_pd["val"].isna().sum() == 0
    else:
        assert res_pd["val"].isna().sum() == len(df)

    pldf = pl.from_pandas(df)
    res_pl = fd.clean(pldf, config=cfg)
    if has_non_null:
        assert res_pl["val"].null_count() == 0
    else:
        assert res_pl["val"].null_count() == len(df)

    lazy_pl = pldf.lazy()
    res_lazy = fd.clean(lazy_pl, config=cfg)
    assert isinstance(res_lazy, pl.LazyFrame)
    if has_non_null:
        assert res_lazy.collect()["val"].null_count() == 0


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    values=st.lists(
        st.one_of(st.floats(min_value=-500.0, max_value=500.0, allow_nan=False), st.none()),
        min_size=10,
        max_size=30,
    )
)
def test_polars_engine_property_invariants(values: list[float | None]) -> None:
    lf = pl.LazyFrame({"val": values, "txt": [" foo " for _ in values]})
    cfg = PeCleanConfig(default_impute="median", default_outliers="clip", strip_strings=True)
    plan = pe_suggest(lf, config=cfg)
    assert plan is not None
    cleaned_lf = pe_apply(lf, plan)
    assert isinstance(cleaned_lf, pl.LazyFrame)
    collected = cleaned_lf.collect()
    has_non_null = any(v is not None for v in values)
    if has_non_null:
        assert collected["val"].null_count() == 0
    else:
        assert collected["val"].null_count() == len(values)
    assert collected["txt"].to_list() == ["foo" for _ in values]
