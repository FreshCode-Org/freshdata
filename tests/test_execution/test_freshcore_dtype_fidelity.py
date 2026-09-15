"""FreshCore dtype and column-label fidelity.

- #262: datetime/categorical/timedelta columns came back as strings, integers
  as float64, and integers beyond 2**53 were silently rounded.
- #232 (part 2): labels such as ``1`` and ``"1"`` collapsed into one native
  column, and non-string labels came back stringified.

The Rust extension is optional, so a fake native module stands in for it.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution.backends._freshcore import FreshCoreEngine

KW = {"strategy": "conservative", "fix_dtypes": False, "verbose": False}


class _ForbiddenNative:
    @staticmethod
    def execute_plan(payload):  # pragma: no cover - must not be called
        raise AssertionError("unexpected native call")


class _EchoNative:
    """Applies the rename map, then returns the columns (optionally overridden)."""

    calls = 0
    payloads: list = []
    overrides: dict = {}

    @classmethod
    def execute_plan(cls, payload):
        cls.calls += 1
        cls.payloads.append(payload)
        renames = dict(payload["config"]["rename_map"])
        columns = []
        for column in payload["columns"]:
            name = renames.get(column["name"], column["name"])
            values = cls.overrides.get(name, column["values"])
            columns.append({**column, "name": name, "values": values})
        return {"columns": columns, "actions": []}


def _use_native(monkeypatch, module) -> None:
    monkeypatch.setattr(FreshCoreEngine, "_load_native", staticmethod(lambda: module))


def _echo(monkeypatch, **overrides) -> type[_EchoNative]:
    native = type(
        "EchoNative", (_EchoNative,), {"calls": 0, "payloads": [], "overrides": overrides}
    )
    _use_native(monkeypatch, native)
    return native


def _dtypes(frame: pd.DataFrame) -> list[str]:
    return [str(t) for t in frame.dtypes]


FALLBACK_CASES = [
    pytest.param(
        pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", None, "2024-01-03"]), "k": [1.0, 2, 3]}),
        "datetime column(s) 'ts'",
        id="datetime",
    ),
    pytest.param(
        pd.DataFrame(
            {"tz": pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC"), "k": [1.0, 2]}
        ),
        "datetime column(s) 'tz'",
        id="datetime-tz",
    ),
    pytest.param(
        pd.DataFrame({"cat": pd.Categorical(["a", "b", "a"]), "k": [1.0, 2, 3]}),
        "categorical column(s) 'cat'",
        id="categorical",
    ),
    pytest.param(
        pd.DataFrame({"td": pd.to_timedelta([1, 2, 3], unit="s"), "k": [1.0, 2, 3]}),
        "timedelta column(s) 'td'",
        id="timedelta",
    ),
    pytest.param(
        pd.DataFrame({"p": pd.period_range("2024-01", periods=3, freq="M"), "k": [1.0, 2, 3]}),
        "period column(s) 'p'",
        id="period",
    ),
    pytest.param(
        pd.DataFrame({"iv": pd.interval_range(0, 3), "k": [1.0, 2, 3]}),
        "interval column(s) 'iv'",
        id="interval",
    ),
    pytest.param(
        pd.DataFrame(
            {
                "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "cat": pd.Categorical(["a", "b"]),
                "n": [1, 2],
            }
        ),
        "datetime/categorical column(s) 'ts', 'cat'",
        id="several-kinds",
    ),
    pytest.param(
        pd.DataFrame({"big": pd.array([2**53 + 1, None, 3], dtype="Int64"), "k": [1.0, 2, 3]}),
        "integer column(s) 'big' hold values beyond ±2**53",
        id="wide-Int64",
    ),
    pytest.param(
        pd.DataFrame({"big": np.array([-(2**53) - 1, 0, 1], dtype="int64")}),
        "integer column(s) 'big' hold values beyond ±2**53",
        id="wide-negative-int64",
    ),
    pytest.param(
        pd.DataFrame({"big": np.array([2**64 - 1, 0, 1], dtype="uint64")}),
        "integer column(s) 'big' hold values beyond ±2**53",
        id="wide-uint64",
    ),
    pytest.param(
        pd.DataFrame({1: [10.0, 20.0, 30.0], "1": ["a", "b", "c"]}),
        "column labels 1, '1' collide once stringified",
        id="232-colliding-labels",
    ),
    pytest.param(
        pd.DataFrame({1: [10.0, 20.0, 30.0], " 1 ": ["a", "b", "c"]}),
        "column labels 1, ' 1 ' collide once stringified",
        id="232-labels-colliding-after-rename",
    ),
]


@pytest.mark.parametrize(("df", "reason"), FALLBACK_CASES)
def test_falls_back_to_pandas_with_identical_output(monkeypatch, df, reason):
    _use_native(monkeypatch, _ForbiddenNative)
    expected, expected_report = fd.clean(df.copy(), engine="pandas", return_report=True, **KW)
    out, report = fd.clean(df.copy(), engine="freshcore", return_report=True, **KW)

    assert report.backend == "pandas"
    (event,) = report.fallback_events
    assert event["backend"] == "freshcore"
    assert event["fallback_step"] == "pipeline"
    assert reason in event["fallback_reason"]
    # engine="pandas" wraps its frame in CleanResult (a DataFrame subclass).
    pd.testing.assert_frame_equal(out, pd.DataFrame(expected))
    assert list(out.columns) == list(expected.columns)
    assert _dtypes(out) == _dtypes(expected)
    assert [(a.step, a.column, a.count) for a in report.actions] == [
        (a.step, a.column, a.count) for a in expected_report.actions
    ]


@pytest.mark.parametrize(("df", "reason"), FALLBACK_CASES)
def test_fallback_policy_error_refuses(monkeypatch, df, reason):
    _use_native(monkeypatch, _ForbiddenNative)
    with pytest.raises(fd.FallbackError, match=re.escape(reason)):
        fd.clean(df, engine="freshcore", fallback_policy="error", **KW)


def test_issue_262_reproduction_matches_pandas(monkeypatch):
    _use_native(monkeypatch, _ForbiddenNative)
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "tz": pd.to_datetime(["2024-01-01", "2024-01-02"]).tz_localize("UTC"),
            "cat": pd.Categorical(["a", "b"]),
            "td": pd.to_timedelta([1, 2], unit="s"),
            "n": [1, 2],
            "big": pd.array([2**53 + 1, None], dtype="Int64"),
        }
    )
    expected = fd.clean(df.copy(), engine="pandas", **KW)
    out, report = fd.clean(df.copy(), engine="freshcore", return_report=True, **KW)

    assert report.backend == "pandas"
    assert _dtypes(out) == _dtypes(expected)
    assert out["big"].iloc[0] == 2**53 + 1


def test_integer_dtypes_round_trip_through_native(monkeypatch):
    df = pd.DataFrame(
        {
            "n": np.array([1, 2, 3], dtype="int64"),
            "big": pd.array([2**53, None, -(2**53)], dtype="Int64"),
            "small": pd.array([1, None, 3], dtype="Int8"),
            "u": np.array([1, 2, 3], dtype="uint16"),
            "x": [1.5, 2.5, 3.5],
        }
    )
    native = _echo(monkeypatch)
    out, report = fd.clean(df.copy(), engine="freshcore", return_report=True, **KW)

    assert native.calls == 1
    assert report.backend == "freshcore"
    assert report.fallback_events == []
    assert report.backend_differences == []
    assert _dtypes(out) == ["int64", "Int64", "Int8", "uint16", "float64"]
    pd.testing.assert_frame_equal(out, df)
    assert out["big"].iloc[0] == 2**53


def test_restored_integer_dtypes_match_pandas_engine(monkeypatch):
    df = pd.DataFrame({"n": [1, 2, 3], "m": pd.array([4, None, 6], dtype="Int64")})
    expected = fd.clean(df.copy(), engine="pandas", **KW)
    _echo(monkeypatch)
    out = fd.clean(df.copy(), engine="freshcore", **KW)
    pd.testing.assert_frame_equal(out, pd.DataFrame(expected))


@pytest.mark.parametrize(
    ("dtype", "native_values", "detail"),
    [
        pytest.param("int64", [1.5, 2.0, 3.0], "non-integral values", id="non-integral"),
        pytest.param(
            "Int64", [1.0, 2.25, None], "non-integral values", id="nullable-non-integral"
        ),
        pytest.param(
            "int64",
            [1.0, None, 3.0],
            "missing values, which a non-nullable integer dtype cannot hold",
            id="missing-in-int64",
        ),
        pytest.param(
            "int8", [1.0, 300.0, 3.0], "values outside the int8 range", id="out-of-range"
        ),
    ],
)
def test_unrestorable_integer_columns_are_reported(monkeypatch, dtype, native_values, detail):
    df = pd.DataFrame({"n": pd.Series([1, 2, 3], dtype=dtype), "k": [1.0, 2.0, 3.0]})
    _echo(monkeypatch, n=native_values)
    out, report = fd.clean(df, engine="freshcore", return_report=True, **KW)

    assert report.backend == "freshcore"
    assert str(out["n"].dtype) == "float64"
    assert out["n"].tolist()[0] == native_values[0]
    (difference,) = report.backend_differences
    assert difference["backend"] == "freshcore"
    assert difference["step"] == "dtypes"
    assert difference["column"] == "n"
    assert detail in difference["detail"]
    assert "float64" in difference["detail"]


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(pd.DataFrame({0: [1.0, 2.0], 1: ["a", "b"]}), id="integer-labels"),
        pytest.param(pd.DataFrame(np.array([[1.0, 2.0], [3.0, 4.0]])), id="range-index-labels"),
        pytest.param(
            pd.DataFrame({" First Name ": ["a", "b"], 2: [1, 2], 2.5: ["x", "y"]}),
            id="mixed-renamed-labels",
        ),
    ],
)
def test_non_string_labels_round_trip(monkeypatch, df):
    expected = fd.clean(df.copy(), engine="pandas", **KW)
    native = _echo(monkeypatch)
    out, report = fd.clean(df.copy(), engine="freshcore", return_report=True, **KW)

    assert native.calls == 1
    assert report.backend == "freshcore"
    assert [c["name"] for c in native.payloads[0]["columns"]] == [str(c) for c in df.columns]
    assert list(out.columns) == list(expected.columns)
    assert [type(c) for c in out.columns] == [type(c) for c in expected.columns]
    pd.testing.assert_frame_equal(out, pd.DataFrame(expected))


@pytest.mark.parametrize(
    "df",
    [
        pytest.param(pd.DataFrame({"s": ["a", None, "b"], "n": [1, 2, 3]}), id="str-and-int"),
        pytest.param(
            pd.DataFrame({"n": np.array([2**53, -(2**53), 0], dtype="int64")}),
            id="int-at-exact-limit",
        ),
        pytest.param(
            pd.DataFrame({"n": pd.array([None, None], dtype="Int64")}), id="all-na-Int64"
        ),
        pytest.param(pd.DataFrame({1: [1.0, 2.0], "2": ["a", "b"]}), id="distinct-mixed-labels"),
    ],
)
def test_supported_shapes_stay_native(monkeypatch, df):
    native = _echo(monkeypatch)
    _, report = fd.clean(df, engine="freshcore", return_report=True, **KW)
    assert native.calls == 1
    assert report.backend == "freshcore"
    assert report.fallback_events == []
