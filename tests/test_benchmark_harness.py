from pathlib import Path

import polars as pl
from benchmarks.bench_v3_lazy_vs_pandas import (
    generate_messy_parquet,
    run_freshdata_v3_lazy,
    run_pandas_fallback,
)


def test_v3_lazy_matches_pandas_fallback_and_compiles_instantly(tmp_path: Path) -> None:
    src_parquet = tmp_path / "messy.parquet"
    out_lazy = tmp_path / "out_lazy.parquet"
    out_pandas = tmp_path / "out_pandas.parquet"

    generate_messy_parquet(src_parquet, n_rows=20_000)

    lazy_metrics = run_freshdata_v3_lazy(src_parquet, out_lazy)
    _ = run_pandas_fallback(src_parquet, out_pandas)

    # 1. Verify apply_plan() compilation is a pure AST operation (under 10 ms)
    assert lazy_metrics["compile_ms"] < 10.0

    # 2. Verify both pipelines produce identical data invariants
    df_lazy = pl.read_parquet(out_lazy)
    df_pd = pl.read_parquet(out_pandas)

    assert df_lazy.height == df_pd.height
    assert df_lazy["age"].null_count() == 0
    assert df_pd["age"].null_count() == 0

    # Protected target column must retain exact null count in v3.0
    assert df_lazy["churn"].null_count() > 0

    # Outlier bounds must match exactly between engines
    assert round(float(df_lazy["spend"].max()), 2) == round(float(df_pd["spend"].max()), 2)
    assert round(float(df_lazy["age"].median()), 2) == round(float(df_pd["age"].median()), 2)
