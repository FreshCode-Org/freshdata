"""Native pandas output discloses extension dtype changes without recasting."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from pandas.api.types import is_dtype_equal

import freshdata as fd
from freshdata.execution._report import record_extension_dtype_changes

pytest.importorskip("polars")
pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

ENGINES = ("polars", "duckdb")
KW = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}


def _dtype_changes(report):
    return [event for event in report.backend_differences if event["step"] == "output_dtype"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("dtype", "values"), [
    pytest.param("Int8", [1, 2, 3], id="nullable-small-int"),
    pytest.param("UInt64", [2**63 + 1, 2**63 + 2, 2**63 + 3], id="wide-unsigned"),
    pytest.param("Float64", [1.0, None, 3.0], id="nullable-float64"),
    pytest.param("Float32", [1.0, None, 3.0], id="nullable-float32"),
    pytest.param("string", ["a", None, "b"], id="nullable-text"),
    pytest.param("string[pyarrow]", ["a", None, "b"], id="arrow-text"),
    pytest.param("boolean", [True, None, False], id="nullable-boolean"),
    pytest.param("Int64", [2**53 + 1, None, 2**53 + 3], id="exact-nullable-wide-int"),
])
def test_extension_dtype_change_is_audited_per_column(engine, dtype, values):
    frame = pd.DataFrame({"c": pd.Series(values, dtype=dtype), "row": [0, 1, 2]})
    original = frame.copy(deep=True)
    out, report = fd.clean(frame, engine=engine, return_report=True, **KW)

    assert out["c"].isna().tolist() == frame["c"].isna().tolist()
    assert out.loc[frame["c"].notna(), "c"].tolist() == frame.loc[
        frame["c"].notna(), "c"
    ].tolist()
    pd.testing.assert_frame_equal(frame, original)
    changes = _dtype_changes(report)
    if is_dtype_equal(frame["c"].dtype, out["c"].dtype):
        assert changes == []
    else:
        assert len(changes) == 1
        assert changes[0]["backend"] == engine
        assert changes[0]["column"] == "c"
        assert str(frame["c"].dtype) in changes[0]["detail"]
        assert str(out["c"].dtype) in changes[0]["detail"]
        assert json.loads(report.to_json())["backend_differences"] == report.backend_differences


@pytest.mark.parametrize("engine", ENGINES)
def test_renamed_and_disambiguated_columns_are_audited_under_output_names(engine):
    frame = pd.DataFrame({
        "Customer Value": pd.Series([1, 2, 3], dtype="Int64"),
        "Customer-Value": pd.Series([4, 5, 6], dtype="Int64"),
    })
    out, report = fd.clean(frame, engine=engine, return_report=True, **KW)
    assert out.columns.tolist() == ["customer_value", "customer_value_2"]
    assert {event["column"] for event in _dtype_changes(report)} == set(out.columns)


@pytest.mark.parametrize("engine", ENGINES)
def test_non_string_column_label_maps_to_the_native_output_label(engine):
    frame = pd.DataFrame({7: pd.Series([1, 2, 3], dtype="Int64")})
    out, report = fd.clean(frame, engine=engine, column_names=False, return_report=True, **KW)
    assert out.columns.tolist() == ["7"]
    assert [event["column"] for event in _dtype_changes(report)] == ["7"]


@pytest.mark.parametrize("engine", ENGINES)
def test_dropped_extension_column_is_not_a_dtype_change(engine):
    frame = pd.DataFrame({"empty": pd.Series([None, None], dtype="Int64"), "row": [0, 1]})
    out, report = fd.clean(frame, engine=engine, return_report=True, **KW)
    assert out.columns.tolist() == ["row"]
    assert _dtype_changes(report) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_regular_numpy_dtypes_do_not_create_extension_dtype_events(engine):
    frame = pd.DataFrame({"n": [1, 2, 3], "s": ["a", "b", "c"]})
    _, report = fd.clean(frame, engine=engine, return_report=True, **KW)
    assert _dtype_changes(report) == []


def test_pandas_fallback_is_not_reported_as_a_native_dtype_change():
    frame = pd.DataFrame({"c": pd.Series([1, 2, 3], dtype="Int64")})
    _, report = fd.clean(frame, engine="polars", strategy="balanced", return_report=True,
                         fix_dtypes=False, verbose=False)
    assert report.backend == "pandas"
    assert report.fallback_events
    assert _dtype_changes(report) == []


@pytest.mark.parametrize(("engine", "output_format"), [
    ("polars", "polars-lazy"), ("duckdb", "duckdb"),
])
def test_native_handles_are_not_materialized_for_pandas_dtype_auditing(
    engine, output_format, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("dtype auditing must not convert a native handle to pandas")

    monkeypatch.setattr("freshdata.adapters.polars.to_pandas", forbidden)
    frame = pd.DataFrame({"c": pd.Series([1, 2, 3], dtype="Int64")})
    _, report = fd.clean(frame, engine=engine, output_format=output_format,
                         return_report=True, **KW)
    assert report.materialized is False
    assert report.rows_materialized is None
    assert _dtype_changes(report) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_arrow_output_does_not_claim_a_pandas_dtype_change(engine):
    frame = pd.DataFrame({"c": pd.Series([1, 2, 3], dtype="Int64")})
    _, report = fd.clean(frame, engine=engine, output_format="arrow", return_report=True, **KW)
    assert _dtype_changes(report) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_auditing_does_not_truncate_an_explicit_fractional_imputation(engine):
    frame = pd.DataFrame({"c": pd.Series([1, 2, None], dtype="Int64"), "row": [0, 1, 2]})
    out, report = fd.clean(frame, engine=engine, impute="mean", return_report=True, **KW)
    assert out["c"].tolist() == [1.0, 2.0, 1.5]
    assert len(_dtype_changes(report)) == 1
    assert any(event["step"] == "impute" for event in report.backend_differences)


@pytest.mark.parametrize("engine", ENGINES)
def test_empty_frame_schema_changes_are_still_audited(engine):
    frame = pd.DataFrame({"c": pd.Series([], dtype="Int64")})
    out, report = fd.clean(frame, engine=engine, return_report=True, **KW)
    assert out.shape == (0, 1)
    changes = _dtype_changes(report)
    assert len(changes) == (not is_dtype_equal(out["c"].dtype, frame["c"].dtype))


@pytest.mark.parametrize("engine", ENGINES)
def test_dtype_details_do_not_include_category_domain_values(engine):
    frame = pd.DataFrame({
        "c": pd.Categorical(["a", "b", "a"],
                            categories=["a", "b", "unused-private-category"], ordered=True),
        "row": [0, 1, 2],
    })
    _, report = fd.clean(frame, engine=engine, return_report=True, **KW)
    assert "unused-private-category" not in report.to_json()


@pytest.mark.parametrize("drop_second", [False, True])
def test_duckdb_registration_names_are_used_for_case_insensitive_headers(drop_second):
    frame = pd.DataFrame({
        "Customer Value": pd.Series([1, 2, 3], dtype="Int64"),
        "customer value": pd.Series([None] * 3 if drop_second else [4, 5, 6], dtype="Int64"),
    })
    out, report = fd.clean(frame, engine="duckdb", return_report=True, **KW)
    assert len(out.columns) == (1 if drop_second else 2)
    assert {event["column"] for event in _dtype_changes(report)} == set(out.columns)


@pytest.mark.parametrize("drop_unrelated", [False, True])
def test_duckdb_final_projection_suffix_uses_the_actual_output_column(drop_unrelated):
    frame = pd.DataFrame({
        "A": [1, 2, 3], "a": ["x", "y", "z"],
        "A 1": pd.Series([4, 5, 6], dtype="Int64"),
    })
    if drop_unrelated:
        frame["Unused"] = None
    out, report = fd.clean(frame, engine="duckdb", return_report=True, **KW)
    assert out.columns.tolist() == ["a", "a_1", "a_1_1"]
    (event,) = _dtype_changes(report)
    assert event["column"] == "a_1_1"
    assert str(out.iloc[:, 2].dtype) in event["detail"]


def test_duplicate_output_labels_are_compared_by_position_without_raising():
    source = pd.DataFrame({
        "A": [1, 2, 3], "a": ["x", "y", "z"],
        "A 1": pd.Series([4, 5, 6], dtype="Int64"),
    })
    result = pd.DataFrame([[1, "x", 4], [2, "y", 5], [3, "z", 6]],
                          columns=["a", "a_1", "a_1"])
    report = fd.CleanReport(backend="duckdb")
    record_extension_dtype_changes(
        source, result, fd.CleanConfig(strategy="conservative", fix_dtypes=False), report
    )
    (event,) = _dtype_changes(report)
    assert event["column"] == "a_1"
    assert "Int64" in event["detail"] and "int64" in event["detail"]
