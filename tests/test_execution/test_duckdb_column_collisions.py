"""DuckDB cannot safely carry case-insensitive pandas header collisions."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution import FallbackError, FallbackWarning
from freshdata.execution.backends._duckdb import DuckDBEngine

pytest.importorskip("duckdb")
KW = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}


def _frame(empty_second=False, unrelated_empty=False):
    frame = pd.DataFrame({
        "A": [1, 2, 3], "a": [None] * 3 if empty_second else ["x", "y", "z"],
        "A 1": pd.Series([4, 5, 6], dtype="Int64"),
    })
    if unrelated_empty:
        frame["Unused"] = None
    return frame


@pytest.mark.parametrize("column_names", [True, False])
@pytest.mark.parametrize(("empty_second", "unrelated_empty"), [
    (False, False), (True, False), (False, True),
])
def test_case_colliding_headers_preserve_pandas_values_and_schema(
    column_names, empty_second, unrelated_empty
):
    frame = _frame(empty_second, unrelated_empty)
    original = frame.copy(deep=True)
    expected, reference = fd.clean(
        frame, engine="pandas", column_names=column_names, return_report=True, **KW
    )
    out, report = fd.clean(
        frame, engine="duckdb", column_names=column_names, return_report=True, **KW
    )
    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected))
    pd.testing.assert_frame_equal(frame, original)
    assert report.backend == "pandas"
    assert report.requested_backend == "duckdb"
    assert "case-insensitive" in report.fallback_events[0]["fallback_reason"]
    assert [(a.step, a.count) for a in report.actions] == [
        (a.step, a.count) for a in reference.actions
    ]


@pytest.mark.parametrize("column_names", [True, False])
def test_case_collision_warn_policy_is_disclosed(column_names):
    with pytest.warns(FallbackWarning, match="case-insensitive"):
        out = fd.clean(_frame(True), engine="duckdb", column_names=column_names,
                       fallback_policy="warn", **KW)
    assert out.shape == (3, 2)


def test_strict_policy_rejects_before_native_ingestion_or_pandas_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("strict policy must reject before cleaning")

    monkeypatch.setattr(DuckDBEngine, "_register_source", forbidden)
    monkeypatch.setattr(DuckDBEngine, "_fallback", forbidden)
    with pytest.raises(FallbackError, match="case-insensitive"):
        fd.clean(_frame(True), engine="duckdb", fallback_policy="error", **KW)


def test_native_handle_request_reports_the_materialized_preservation_fallback():
    out, report = fd.clean(_frame(True), engine="duckdb", output_format="duckdb",
                           return_report=True, **KW)
    assert isinstance(out, pd.DataFrame)
    assert report.materialized is True
    assert out.shape == (3, 2)
    assert report.fallback_events


@pytest.mark.parametrize("names", [("A B", "A-B"), ("Ä", "ä"), ("ß", "ss")])
def test_non_colliding_duckdb_identifiers_remain_native(names):
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=names)
    out, report = fd.clean(frame, engine="duckdb", return_report=True, **KW)
    assert out.shape == (2, 2)
    assert report.backend == "duckdb"
    assert report.fallback_events == []


def test_non_string_labels_that_collide_in_duckdb_are_preserved_by_pandas():
    frame = pd.DataFrame({7: [1, 2, 3], "7": [4, 5, 6]})
    expected = fd.clean(frame, engine="pandas", column_names=False, **KW)
    out, report = fd.clean(frame, engine="duckdb", column_names=False,
                           return_report=True, **KW)
    pd.testing.assert_frame_equal(pd.DataFrame(out), pd.DataFrame(expected))
    assert report.backend == "pandas"
