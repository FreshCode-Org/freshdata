"""Copilot sample masking is an allow-list: only numeric and boolean values pass.

Regression tests for Arrow-backed string columns (and every other non-numeric
dtype) reaching the provider prompt raw because the old check recognised only
object, ``string`` and categorical columns as string-like.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable

import pandas as pd
import pytest

from freshdata.experimental.ai_copilot import (
    _mask_sample,
    _passes_through_raw,
    _sample_mask_positions,
    analyze_dataset,
)

NAMES = ["Alice Johnson", "Bob Smith"]


def _pa():
    return pytest.importorskip("pyarrow")


def _run_with_prompt(frame: pd.DataFrame, **kwargs):
    prompts: list[str] = []

    def provider(prompt: str) -> str:
        prompts.append(prompt)
        return "ok"

    with pytest.warns(FutureWarning, match="experimental"):
        report = analyze_dataset(frame, provider=provider, **kwargs)
    assert len(prompts) == 1
    return report, prompts[0]


def _sinks(report, prompt: str) -> dict[str, str]:
    rendered = report._repr_html_()
    return {
        "prompt": prompt,
        "model_context": json.dumps(report.model_context, default=str, ensure_ascii=False),
        "to_json": report.to_json(),
        "html": rendered,
        "html_unescaped": html.unescape(rendered),
        "str": str(report),
    }


def _assert_absent(report, prompt: str, raws: list[str]) -> None:
    for sink, text in _sinks(report, prompt).items():
        leaked = [raw for raw in raws if raw in text]
        assert not leaked, f"raw sample value(s) {leaked} reached {sink}"


def _arrow(type_factory: Callable[[object], object]) -> Callable[[], pd.Series]:
    def build() -> pd.Series:
        pa = _pa()
        return pd.Series(pd.array(NAMES, dtype=pd.ArrowDtype(type_factory(pa))))

    return build


def _string_pyarrow() -> pd.Series:
    _pa()
    return pd.Series(NAMES, dtype="string[pyarrow]")


# Each case builds the column and lists raw renderings that must never escape.
MASKED_CASES: dict[str, tuple[Callable[[], pd.Series], list[str]]] = {
    "arrow-string": (_arrow(lambda pa: pa.string()), NAMES),
    "arrow-large_string": (_arrow(lambda pa: pa.large_string()), NAMES),
    "arrow-dictionary": (_arrow(lambda pa: pa.dictionary(pa.int32(), pa.string())), NAMES),
    "string[pyarrow]": (_string_pyarrow, NAMES),
    "string[python]": (lambda: pd.Series(NAMES, dtype="string"), NAMES),
    "categorical": (lambda: pd.Series(pd.Categorical(NAMES)), NAMES),
    "object-bytes": (lambda: pd.Series([n.encode() for n in NAMES], dtype=object), NAMES),
    "datetime64": (
        lambda: pd.Series(pd.to_datetime(["1980-02-03 04:05:06", "1975-06-07 08:09:10"])),
        ["1980-02-03", "1975-06-07"],
    ),
    "datetime64-tz": (
        lambda: pd.Series(pd.to_datetime(["1980-02-03 04:05:06", "1975-06-07 08:09:10"])).dt
        .tz_localize("UTC"),
        ["1980-02-03", "1975-06-07"],
    ),
    "timedelta64": (
        lambda: pd.Series(pd.to_timedelta([1234567, 7654321], unit="s")),
        ["14 days 06:56:07", "88 days 14:12:01"],
    ),
    "period": (
        lambda: pd.Series(pd.period_range("1980-02-03", periods=2, freq="D")),
        ["1980-02-03", "1980-02-04"],
    ),
    "categorical-of-ints": (
        lambda: pd.Series(pd.Categorical([123456789, 987654321])),
        ["123456789", "987654321"],
    ),
}


def test_advisory_poc_arrow_string_column_is_masked() -> None:
    pa = _pa()
    df = pd.DataFrame({"name": pd.array(NAMES, dtype=pd.ArrowDtype(pa.string())), "v": [1, 2]})
    report, prompt = _run_with_prompt(df)
    assert [n for n in NAMES if n in prompt] == []
    assert "name" in report.audit["sample_masked_columns"]


@pytest.mark.parametrize("case", sorted(MASKED_CASES))
def test_non_numeric_dtype_never_reaches_any_sink_raw(case: str) -> None:
    build, raws = MASKED_CASES[case]
    column = build()
    assert not _passes_through_raw(column.dtype)
    df = pd.DataFrame({"col": column, "v": [1, 2]})
    report, prompt = _run_with_prompt(df)
    assert "col" in report.audit["sample_masked_columns"]
    rows = report.model_context["sample_rows_masked"]
    assert len(rows) == 2
    for row in rows:
        assert isinstance(row["col"], str)
        assert len(row["col"]) == 16 and set(row["col"]) <= set("0123456789abcdef")
    _assert_absent(report, prompt, raws)


def test_arrow_list_of_strings_is_masked_in_sample() -> None:
    # ``analyze_dataset`` cannot profile list columns (pandas cannot
    # factorize them), so check the sample masking helpers directly.
    pa = _pa()
    frame = pd.DataFrame(
        {
            "tags": pd.array([[n] for n in NAMES], dtype=pd.ArrowDtype(pa.list_(pa.string()))),
            "v": [1, 2],
        }
    )
    assert not _passes_through_raw(frame["tags"].dtype)
    positions = _sample_mask_positions(frame, [], [])
    assert positions == [0]
    blob = json.dumps(_mask_sample(frame, positions, 5), default=str)
    assert not [n for n in NAMES if n in blob]


def _passthrough_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i64": pd.Series([123456, 654321], dtype="int64"),
            "f64": [1.25, 2.5],
            "flag": [True, False],
            "nullable_int": pd.Series([7, None], dtype="Int64"),
            "nullable_bool": pd.Series([True, None], dtype="boolean"),
        }
    )


def test_numeric_and_bool_columns_still_pass_through() -> None:
    frame = _passthrough_frame()
    report = analyze_dataset(frame)
    assert report.audit["sample_masked_columns"] == []
    rows = report.model_context["sample_rows_masked"]
    assert [r["i64"] for r in rows] == [123456, 654321]
    assert [r["f64"] for r in rows] == [1.25, 2.5]
    assert [r["flag"] for r in rows] == [True, False]
    assert [r["nullable_int"] for r in rows] == [7, None]
    assert [r["nullable_bool"] for r in rows] == [True, None]


def test_arrow_numeric_and_bool_columns_still_pass_through() -> None:
    pa = _pa()
    frame = pd.DataFrame(
        {
            "a_int": pd.array([123456, 654321], dtype=pd.ArrowDtype(pa.int64())),
            "a_float": pd.array([1.25, 2.5], dtype=pd.ArrowDtype(pa.float64())),
            "a_bool": pd.array([True, False], dtype=pd.ArrowDtype(pa.bool_())),
        }
    )
    for column in frame.columns:
        assert _passes_through_raw(frame[column].dtype), column
    assert _sample_mask_positions(frame, [], []) == []
    rows = _mask_sample(frame, [], 5)
    assert [r["a_int"] for r in rows] == [123456, 654321]
    assert [r["a_bool"] for r in rows] == [True, False]


@pytest.mark.parametrize(
    "dtype",
    [
        object,
        "string",
        pd.CategoricalDtype(["a"]),
        pd.CategoricalDtype([1, 2]),
        "datetime64[ns]",
        pd.DatetimeTZDtype(tz="UTC"),
        "timedelta64[ns]",
        pd.PeriodDtype("D"),
        pd.IntervalDtype("int64"),
    ],
    ids=str,
)
def test_passes_through_raw_rejects_non_numeric_dtypes(dtype) -> None:
    assert not _passes_through_raw(pd.Series([], dtype=dtype).dtype)


@pytest.mark.parametrize(
    "dtype", ["int8", "uint64", "float32", "complex128", "bool", "Int64", "Float64", "boolean"]
)
def test_passes_through_raw_accepts_numeric_and_bool(dtype) -> None:
    assert _passes_through_raw(pd.Series([], dtype=dtype).dtype)


def test_declared_numeric_column_is_still_masked() -> None:
    df = pd.DataFrame({"ssn": [123456789, 987654321], "v": [1, 2]})
    report, prompt = _run_with_prompt(df, sensitive_columns=["ssn"])
    assert report.audit["sample_masked_columns"] == ["ssn"]
    _assert_absent(report, prompt, ["123456789", "987654321"])


def test_allow_unmasked_columns_opts_out_non_string_dtypes() -> None:
    df = pd.DataFrame(
        {"when": pd.to_datetime(["1980-02-03", "1975-06-07"]), "v": [1, 2]}
    )
    masked = analyze_dataset(df)
    assert "1980-02-03" not in json.dumps(masked.model_context, default=str)
    allowed = analyze_dataset(df, allow_unmasked_columns=("when",))
    assert allowed.audit["sample_masked_columns"] == []
    assert "1980-02-03" in json.dumps(allowed.model_context, default=str)
