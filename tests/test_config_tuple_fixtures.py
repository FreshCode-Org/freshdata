"""Config string-vs-tuple parity across real-world fixtures."""

import pandas as pd
import pytest

import freshdata as fd
from expectations import ALL_FIXTURES, load_expectations, load_fixture, resolve_column

ISOLATE = {"drop_empty_rows": False, "drop_duplicates": False, "verbose": False}

# Priority fixtures from multi-dataset debug sweep (see scripts/debug_dataset_sweep.py).
PRIORITY_FIXTURES = [
    "mixed_roles",
    "fraud_signals",
    "duplicate_heavy",
    "locale_numbers",
    "survey_responses",
    "wide_sparse",
    "aqi_sample",
]


def _parity(clean_a, clean_b) -> None:
    pd.testing.assert_frame_equal(clean_a, clean_b)


@pytest.mark.parametrize("fixture_name", PRIORITY_FIXTURES)
def test_id_columns_string_matches_tuple(fixture_name):
    df = load_fixture(fixture_name)
    id_cols = [c for c in df.columns if c.lower().endswith("_id") or c.lower() == "id"]
    col = str(id_cols[0] if id_cols else df.columns[0])
    kw = {**ISOLATE, "id_columns": col, "outlier_action": "cap"}
    _parity(fd.clean(df, **kw), fd.clean(df, **{**kw, "id_columns": (col,)}))


@pytest.mark.parametrize("fixture_name", PRIORITY_FIXTURES)
def test_preserve_columns_string_matches_tuple(fixture_name):
    df = load_fixture(fixture_name)
    exp = load_expectations(fixture_name).get("balanced", {})
    cols = exp.get("columns_never_imputed") or exp.get("columns_preserved_as_text") or []
    col = resolve_column(df, cols[0]) if cols else str(df.columns[-1])
    kw = {**ISOLATE, "preserve_columns": col, "outlier_action": "cap"}
    _parity(fd.clean(df, **kw), fd.clean(df, **{**kw, "preserve_columns": (col,)}))


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_extra_sentinels_string_matches_tuple(fixture_name):
    df = load_fixture(fixture_name)
    kw = {**ISOLATE, "extra_sentinels": "missing"}
    _parity(fd.clean(df, **kw), fd.clean(df, **{**kw, "extra_sentinels": ("missing",)}))


@pytest.mark.parametrize("fixture_name", ALL_FIXTURES)
def test_clean_idempotent_balanced(fixture_name):
    df = load_fixture(fixture_name)
    once = fd.clean(df, strategy="balanced", verbose=False)
    twice = fd.clean(once, strategy="balanced", verbose=False)
    pd.testing.assert_frame_equal(once, twice)
