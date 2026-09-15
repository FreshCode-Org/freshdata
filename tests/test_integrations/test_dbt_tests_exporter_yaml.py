"""YAML round-trip tests for the dbt tests exporter's scalar writer (issue #342).

Every value ``export_dbt_tests`` writes must load back (via PyYAML, which dbt uses)
as the same Python value and type: strings stay strings, numbers stay numbers.
"""

from __future__ import annotations

import itertools
import math

import pytest

import freshdata as fd
from freshdata import QualityFinding
from freshdata.integrations.dbt.tests_exporter import _scalar

yaml = pytest.importorskip("yaml")

TRICKY_STRINGS = [
    # YAML 1.1 dates, timestamps and sexagesimals
    "2024-01-01",
    "2024-01-01T10:00:00",
    "2024-01-01 10:00:00",
    "12:30",
    "1:30:00",
    # int / float literal spellings
    "0x1F",
    "0b101",
    "0o17",
    "017",
    "012",
    "1e3",
    "1_000",
    ".5",
    "+1",
    "-1",
    "1.0",
    "0",
    "1",
    "-0",
    ".inf",
    "-.inf",
    ".NaN",
    "nan",
    "inf",
    "Infinity",
    # bool / null words
    "yes",
    "No",
    "on",
    "OFF",
    "true",
    "False",
    "null",
    "Null",
    "NULL",
    "~",
    "y",
    "n",
    "Y",
    "N",
    "none",
    # indicators and flow characters
    "a: b",
    "# x",
    "a #b",
    "a:b",
    "- x",
    "-",
    "?",
    "? x",
    "|",
    ">",
    "!tag",
    "&anchor",
    "*alias",
    "%dir",
    "@at",
    "`tick`",
    "[a]",
    "{a: 1}",
    "a, b",
    "<<",
    "=",
    "'single'",
    '"double"',
    # whitespace and escaping
    "",
    " ",
    " lead",
    "trail ",
    'with "quote"',
    "back\\slash",
    "tab\tchar",
    "new\nline",
    "cr\rchar",
    "a\n",
    "\nb",
    # control and special Unicode characters
    "nul\x00",
    "bell\x07",
    "esc\x1b",
    "del\x7f",
    "nel\x85",
    "c1\x9f",
    "ls\u2028",
    "ps\u2029",
    "bom\ufeff",
    "nonchar\ufffe",
    "nonchar\uffff",
    "café",
    "中文",
    "emoji\U0001f600",
]

PLAIN_STRINGS = [
    "open",
    "closed",
    "not_null",
    "freshdata_expectation",
    "warn",
    "orders",
    "_private",
    "a-b.c",
    "Col1",
]

NON_STRING_VALUES = [
    None,
    True,
    False,
    0,
    1,
    -5,
    10**20,
    -(10**20),
    1.5,
    -0.0,
    0.1,
    1e16,
    1e-5,
    1e300,
    -2.5e-300,
    float("inf"),
    float("-inf"),
]


def _load_in_contexts(rendered: str) -> list[object]:
    """Load *rendered* as a mapping value and as a sequence item."""
    as_value = yaml.safe_load(f"key: {rendered}\n")["key"]
    as_item = yaml.safe_load(f"items:\n  - {rendered}\n")["items"][0]
    return [as_value, as_item]


def _assert_same(loaded: object, expected: object) -> None:
    assert type(loaded) is type(expected), (loaded, expected)
    assert loaded == expected


@pytest.mark.parametrize("text", TRICKY_STRINGS + PLAIN_STRINGS)
def test_strings_round_trip_as_same_string(text: str) -> None:
    for loaded in _load_in_contexts(_scalar(text)):
        _assert_same(loaded, text)


@pytest.mark.parametrize("value", NON_STRING_VALUES)
def test_numbers_bools_and_none_keep_their_type(value: object) -> None:
    rendered = _scalar(value)
    assert not rendered.startswith('"')
    for loaded in _load_in_contexts(rendered):
        _assert_same(loaded, value)


def test_nan_stays_a_float() -> None:
    rendered = _scalar(float("nan"))
    assert rendered == ".nan"
    for loaded in _load_in_contexts(rendered):
        assert isinstance(loaded, float) and math.isnan(loaded)


def test_numpy_float_stays_a_float() -> None:
    np = pytest.importorskip("numpy")
    for loaded in _load_in_contexts(_scalar(np.float64(1e16))):
        _assert_same(loaded, 1e16)


@pytest.mark.parametrize("text", PLAIN_STRINGS)
def test_ordinary_identifiers_stay_unquoted(text: str) -> None:
    assert _scalar(text) == text


def test_lone_surrogate_is_escaped_to_ascii() -> None:
    rendered = _scalar("x\ud800")
    assert rendered == '"x\\ud800"'
    for loaded in _load_in_contexts(rendered):
        _assert_same(loaded, "x\ud800")


def test_all_short_strings_over_tricky_alphabet_round_trip() -> None:
    alphabet = "aZ_09 .-+:#'\"\\\n\t~ynx!&*,[]{}%@`|>?=<e"
    for length in (1, 2):
        for chars in itertools.product(alphabet, repeat=length):
            text = "".join(chars)
            for loaded in _load_in_contexts(_scalar(text)):
                _assert_same(loaded, text)


def test_issue_342_repro_end_to_end(tmp_path) -> None:
    values = ["2024-01-01", "0x1F", "0b101", "017"]
    names = ["2024-01-01", "a\nb"]
    findings = [
        QualityFinding.create(
            severity="error",
            step="s",
            column=name,
            rule_name="allowed_values",
            message="m",
            extra={"value_set": values},
        )
        for name in names
    ]
    path = tmp_path / "schema.yml"
    text = fd.export_dbt_tests(findings, "orders", str(path))

    assert path.read_text(encoding="utf-8") == text
    doc = yaml.safe_load(text)
    cols = doc["models"][0]["columns"]
    assert [c["name"] for c in cols] == ["2024-01-01", "a\nb"]
    for col in cols:
        accepted = col["tests"][0]["accepted_values"]
        assert accepted["values"] == values
        assert all(type(v) is str for v in accepted["values"])


def test_export_round_trips_mixed_value_sets(tmp_path) -> None:
    values: list[object] = [*TRICKY_STRINGS, *PLAIN_STRINGS, 1, 2.5, True, None]
    finding = QualityFinding.create(
        severity="warning",
        step="s",
        column="yes",
        rule_name="allowed_values",
        message="on: off",
        extra={"value_set": values},
    )
    text = fd.export_dbt_tests([finding], "2024-01-01", str(tmp_path / "s.yml"))
    doc = yaml.safe_load(text)
    model = doc["models"][0]
    assert model["name"] == "2024-01-01"
    col = model["columns"][0]
    assert col["name"] == "yes"
    loaded = col["tests"][0]["accepted_values"]["values"]
    assert len(loaded) == len(values)
    for got, expected in zip(loaded, values):
        _assert_same(got, expected)
