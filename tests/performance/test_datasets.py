from __future__ import annotations

import pandas as pd
import pytest
from benchmarks.performance.datasets import WIDTHS, DatasetSpec, make_mixed_frame


@pytest.mark.parametrize("width, n_cols", WIDTHS.items())
def test_mixed_frame_has_exact_shape_and_required_roles(width: str, n_cols: int) -> None:
    df = make_mixed_frame(DatasetSpec(rows=1_000, width=width, seed=42))
    assert df.shape == (1_000, n_cols)
    assert {"record_id", "target", "numeric_0", "category_0", "text_0", "event_time_0"} <= set(
        df.columns
    )
    assert df["target"].isna().sum() > 0
    assert df.duplicated().sum() > 0
    assert isinstance(df["category_0"].dtype, pd.CategoricalDtype)
    assert isinstance(df["event_time_0"].dtype, pd.DatetimeTZDtype)


def test_mixed_frame_is_deterministic_and_seed_sensitive() -> None:
    spec = DatasetSpec(rows=2_000, width="medium", seed=7)
    assert make_mixed_frame(spec).equals(make_mixed_frame(spec))
    assert not make_mixed_frame(spec).equals(
        make_mixed_frame(DatasetSpec(rows=2_000, width="medium", seed=8))
    )


def test_mixed_frame_covers_nullable_outlier_and_high_cardinality_cases() -> None:
    df = make_mixed_frame(DatasetSpec(rows=5_000, width="medium", seed=42))
    assert str(df["nullable_int_0"].dtype) == "Int64"
    assert df["nullable_int_0"].isna().any()
    assert df["numeric_0"].isna().any()
    assert df["numeric_0"].max() > 1_000
    assert df["high_cardinality_0"].nunique() > 2_000


@pytest.mark.parametrize(
    "dataset_type, expected_prefix",
    [
        ("numeric", "numeric_"),
        ("categorical", "category_"),
        ("string", "text_"),
        ("nullable", "nullable_int_"),
        ("datetime", "datetime_"),
        ("high_cardinality", "high_cardinality_"),
    ],
)
def test_family_profile_dominates_non_role_columns(
    dataset_type: str, expected_prefix: str
) -> None:
    df = make_mixed_frame(
        DatasetSpec(rows=1_000, width="medium", seed=42, dataset_type=dataset_type)
    )
    family_columns = [name for name in df.columns if name.startswith(expected_prefix)]
    assert len(family_columns) >= 24
