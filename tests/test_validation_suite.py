"""ValidationSuite: build, run, serialize, migrate, CLI."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import cli
from freshdata.enterprise.contracts import ColumnContract, DataContract, enforce_contract
from freshdata.validation_suite import (
    ColumnRule,
    CrossColumnRule,
    ValidationError,
    ValidationSuite,
    run_suite,
)


@pytest.fixture
def customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["c1", "c2", "c3", "c4"],
            "age": [30, 45, 22, 67],
            "email": ["a@x.com", "b@y.org", "c@z.net", "d@w.io"],
            "signup": pd.to_datetime(
                ["2020-01-01", "2020-06-01", "2021-01-01", "2021-06-01"]
            ),
            "last_seen": pd.to_datetime(
                ["2021-01-01", "2021-06-01", "2022-01-01", "2022-06-01"]
            ),
        }
    )


@pytest.fixture
def suite() -> ValidationSuite:
    return ValidationSuite(
        name="customers",
        rules=[
            ColumnRule("customer_id", nullable=False, unique=True),
            ColumnRule("age", min_value=0, max_value=120),
            ColumnRule("email", regex=r"[^@]+@[^@]+\.[^@]+"),
        ],
        cross_column=[CrossColumnRule("signup", "<=", "last_seen")],
        min_rows=1,
    )


# -- basic pass/fail -----------------------------------------------------


def test_passing_frame(customers, suite):
    result = fd.validate(customers, suite=suite)
    assert result.passed
    assert result.n_errors == 0
    result.raise_if_failed()  # no-op on pass


def test_failing_frame_raises(customers, suite):
    bad = customers.copy()
    bad.loc[0, "age"] = -3
    result = fd.validate(bad, suite=suite)
    assert not result.passed
    with pytest.raises(ValidationError) as exc:
        result.raise_if_failed()
    assert "customers" in str(exc.value)
    assert exc.value.result is result


def test_validation_never_mutates(customers, suite):
    bad = customers.copy()
    bad.loc[0, "age"] = -3
    before = bad.copy(deep=True)
    fd.validate(bad, suite=suite)
    pd.testing.assert_frame_equal(bad, before)


def test_suite_and_context_mutually_exclusive(customers, suite):
    with pytest.raises(TypeError, match="not both"):
        fd.validate(customers, suite=suite, context="age is positive")


def test_suite_wrong_type():
    with pytest.raises(TypeError, match="ValidationSuite"):
        fd.validate(pd.DataFrame(), suite=42)


# -- mostly tolerance ----------------------------------------------------


def test_mostly_tolerates_violations_as_warning():
    df = pd.DataFrame({"v": list(range(99)) + [-1]})
    tolerant = ValidationSuite(name="t", rules=[ColumnRule("v", min_value=0, mostly=0.95)])
    result = run_suite(df, tolerant)
    assert result.passed
    assert result.n_warnings == 1
    assert result.report.warnings[0].details["n_violations"] == 1


def test_mostly_still_fails_above_tolerance():
    df = pd.DataFrame({"v": [1, -1, -2, -3]})
    tolerant = ValidationSuite(name="t", rules=[ColumnRule("v", min_value=0, mostly=0.95)])
    result = run_suite(df, tolerant)
    assert not result.passed


def test_mostly_default_fails_single_violation():
    df = pd.DataFrame({"v": list(range(999)) + [-1]})
    strict = ValidationSuite(name="t", rules=[ColumnRule("v", min_value=0)])
    assert not run_suite(df, strict).passed


def test_mostly_out_of_range_rejected():
    with pytest.raises(ValueError, match="mostly"):
        ColumnRule("v", mostly=0.0)
    with pytest.raises(ValueError, match="mostly"):
        ColumnRule("v", mostly=1.5)


# -- new primitives ------------------------------------------------------


def test_string_length_bounds():
    df = pd.DataFrame({"code": ["ab", "abc", "abcd"]})
    r = run_suite(df, ValidationSuite(name="t", rules=[ColumnRule("code", min_length=3)]))
    assert not r.passed
    assert r.report.errors[0].check_id == "contract.min_length"
    r = run_suite(df, ValidationSuite(name="t", rules=[ColumnRule("code", max_length=3)]))
    assert not r.passed
    assert r.report.errors[0].check_id == "contract.max_length"
    r = run_suite(
        df, ValidationSuite(name="t", rules=[ColumnRule("code", min_length=2, max_length=4)])
    )
    assert r.passed


def test_datetime_range():
    df = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01", "2023-06-15", "2019-12-31"])})
    suite = ValidationSuite(
        name="t", rules=[ColumnRule("ts", min_datetime="2020-01-01")]
    )
    r = run_suite(df, suite)
    assert not r.passed
    assert r.report.errors[0].check_id == "contract.min_datetime"
    suite = ValidationSuite(
        name="t",
        rules=[ColumnRule("ts", min_datetime="2019-01-01", max_datetime="2024-01-01")],
    )
    assert run_suite(df, suite).passed


def test_datetime_range_on_string_column():
    df = pd.DataFrame({"ts": ["2020-01-01", "2025-01-01"]})
    suite = ValidationSuite(name="t", rules=[ColumnRule("ts", max_datetime="2024-01-01")])
    assert not run_suite(df, suite).passed


def test_datetime_range_tz_aware():
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2020-01-01T00:00:00Z", "2025-06-01T00:00:00Z"])}
    )
    suite = ValidationSuite(name="t", rules=[ColumnRule("ts", max_datetime="2024-01-01")])
    r = run_suite(df, suite)
    assert not r.passed


def test_numeric_range_on_datetime_column_warns_instead_of_silent_pass():
    # Regression: min_value/max_value used to silently no-op on datetimes.
    df = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01", "2021-01-01"])})
    suite = ValidationSuite(name="t", rules=[ColumnRule("ts", min_value=0)])
    r = run_suite(df, suite)
    findings = [f for f in r.report.findings if "min_datetime" in f.message]
    assert findings, "expected a warning pointing at min_datetime/max_datetime"


def test_row_count_bounds(customers):
    assert not run_suite(customers, ValidationSuite(name="t", min_rows=10)).passed
    assert not run_suite(customers, ValidationSuite(name="t", max_rows=2)).passed
    assert run_suite(customers, ValidationSuite(name="t", min_rows=1, max_rows=100)).passed


def test_compound_unique():
    df = pd.DataFrame({"region": ["eu", "eu", "us"], "sku": ["a", "a", "a"]})
    suite = ValidationSuite(name="t", compound_unique=(("region", "sku"),))
    r = run_suite(df, suite)
    assert not r.passed
    assert r.report.errors[0].check_id == "contract.compound_unique"
    assert run_suite(df.drop_duplicates(), suite).passed


def test_compound_unique_missing_column():
    df = pd.DataFrame({"region": ["eu"]})
    r = run_suite(df, ValidationSuite(name="t", compound_unique=(("region", "sku"),)))
    assert not r.passed
    assert "missing column" in r.report.errors[0].message


def test_strict_columns_exact_schema():
    df = pd.DataFrame({"a": [1], "extra": [2]})
    suite = ValidationSuite(
        name="t", rules=[ColumnRule("a"), ColumnRule("b")], strict_columns=True
    )
    r = run_suite(df, suite)
    messages = " | ".join(f.message for f in r.report.errors)
    assert "extra" in messages  # undeclared column
    assert "'b'" in messages  # missing declared column
    exact = pd.DataFrame({"a": [1], "b": [2]})
    assert run_suite(exact, suite).passed


def test_dtype_exact():
    df = pd.DataFrame({"v": pd.array([1, 2], dtype="int32")})
    family = ValidationSuite(name="t", rules=[ColumnRule("v", dtype="int64")])
    assert run_suite(df, family).passed  # families collapse by default
    exact = ValidationSuite(name="t", rules=[ColumnRule("v", dtype="int64", dtype_exact=True)])
    assert not run_suite(df, exact).passed


# -- cross-column rules --------------------------------------------------


def test_cross_column_ops():
    df = pd.DataFrame({"lo": [1, 2, 3], "hi": [2, 3, 4]})
    for op, holds in [("<", True), ("<=", True), (">", False), ("==", False), ("!=", True)]:
        r = run_suite(
            df, ValidationSuite(name="t", cross_column=[CrossColumnRule("lo", op, "hi")])
        )
        assert r.passed is holds, op


def test_cross_column_mostly():
    df = pd.DataFrame({"lo": [1, 2, 30] + [0] * 97, "hi": [2, 3, 4] + [1] * 97})
    rule = CrossColumnRule("lo", "<", "hi", mostly=0.95)
    r = run_suite(df, ValidationSuite(name="t", cross_column=[rule]))
    assert r.passed
    assert r.n_warnings == 1


def test_cross_column_nulls_skipped():
    df = pd.DataFrame({"lo": [1, None], "hi": [2, 0]})
    r = run_suite(df, ValidationSuite(name="t", cross_column=[CrossColumnRule("lo", "<", "hi")]))
    assert r.passed


def test_cross_column_missing_column():
    df = pd.DataFrame({"lo": [1]})
    r = run_suite(df, ValidationSuite(name="t", cross_column=[CrossColumnRule("lo", "<", "hi")]))
    assert not r.passed
    assert "missing" in r.report.errors[0].message


def test_cross_column_bad_op():
    with pytest.raises(ValueError, match="op"):
        CrossColumnRule("a", "~", "b")


# -- edge cases ----------------------------------------------------------


def test_empty_frame():
    df = pd.DataFrame({"a": pd.Series([], dtype="float64")})
    suite = ValidationSuite(name="t", rules=[ColumnRule("a", min_value=0)])
    assert run_suite(df, suite).passed
    assert not run_suite(df, ValidationSuite(name="t", min_rows=1)).passed


def test_all_null_column():
    df = pd.DataFrame({"a": [None, None]})
    suite = ValidationSuite(
        name="t", rules=[ColumnRule("a", min_value=0, min_length=1, regex="x")]
    )
    assert run_suite(df, suite).passed  # value checks apply to non-null values only
    assert not run_suite(
        df, ValidationSuite(name="t", rules=[ColumnRule("a", nullable=False)])
    ).passed


def test_unicode_values():
    df = pd.DataFrame({"name": ["café", "naïve", "日本語"]})
    suite = ValidationSuite(name="t", rules=[ColumnRule("name", min_length=2, max_length=10)])
    assert run_suite(df, suite).passed


def test_non_default_index():
    df = pd.DataFrame({"a": [1, 2]}, index=[10, 20])
    assert run_suite(df, ValidationSuite(name="t", rules=[ColumnRule("a", min_value=0)])).passed


def test_polars_input_records_materialization():
    pl = pytest.importorskip("polars")
    pdf = pl.DataFrame({"a": [1, 2]})
    result = run_suite(pdf, ValidationSuite(name="t", rules=[ColumnRule("a", min_value=0)]))
    assert result.passed
    assert result.execution["backend"] == "pandas"
    assert any("materialized" in f["reason"] for f in result.execution["fallback"])


def test_pandas_input_no_fallback_recorded(customers, suite):
    result = run_suite(customers, suite)
    assert result.execution["fallback"] == []


# -- serialization + migration -------------------------------------------


def test_json_roundtrip(suite):
    restored = ValidationSuite.from_json(suite.to_json())
    assert restored == suite


def test_save_load(tmp_path, suite):
    path = tmp_path / "suite.json"
    suite.save(path)
    assert ValidationSuite.load(path) == suite


def test_validate_accepts_path(tmp_path, customers, suite):
    path = tmp_path / "suite.json"
    suite.save(path)
    assert fd.validate(customers, suite=str(path)).passed


def test_schema_version_rejected():
    with pytest.raises(ValueError, match="schema_version"):
        ValidationSuite.from_dict({"schema_version": "v999", "name": "t"})


def test_result_serializable(customers, suite):
    bad = customers.copy()
    bad.loc[0, "age"] = -3
    result = run_suite(bad, suite)
    parsed = json.loads(result.to_json())
    assert parsed["passed"] is False
    assert parsed["suite_name"] == "customers"
    assert parsed["findings"]


def test_from_contract_migration(customers):
    contract = DataContract(
        name="legacy",
        columns=(ColumnContract("age", min_value=0, max_value=120),),
        min_rows=1,
    )
    suite = ValidationSuite.from_contract(contract)
    assert suite.name == "legacy"
    assert suite.min_rows == 1
    assert run_suite(customers, suite).passed
    # suite compiles back to an equivalent contract
    assert suite.to_contract().to_dict()["columns"] == contract.to_dict()["columns"]


def test_enforce_contract_direct(customers):
    contract = DataContract(name="c", columns=(ColumnContract("age", max_value=10),))
    report = enforce_contract(customers, contract)
    assert not report.passed
    assert report.baseline_name == "c"


def test_rules_reject_non_rule():
    with pytest.raises(TypeError, match="ColumnRule"):
        ValidationSuite(name="t", rules=["not a rule"])  # type: ignore[list-item]


# -- CLI -----------------------------------------------------------------


def _write_inputs(tmp_path, suite):
    data = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(data, index=False)
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, -5, 3]}).to_csv(bad, index=False)
    spath = tmp_path / "suite.json"
    suite.save(spath)
    return data, bad, spath


def test_cli_validate_pass_fail_usage(tmp_path, capsys):
    suite = ValidationSuite(name="cli", rules=[ColumnRule("a", min_value=0)])
    data, bad, spath = _write_inputs(tmp_path, suite)
    assert cli.main(["validate", str(data), "--suite", str(spath)]) == 0
    assert cli.main(["validate", str(bad), "--suite", str(spath)]) == 1
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL" in out
    assert cli.main(["validate", str(data)]) == 2  # neither --suite nor --contract
    assert cli.main(["validate", str(data), "--suite", str(spath), "--contract", "x"]) == 2


def test_cli_validate_contract_and_json_output(tmp_path):
    contract = DataContract(name="c", columns=(ColumnContract("a", min_value=0),))
    cpath = tmp_path / "contract.json"
    cpath.write_text(json.dumps(contract.to_dict()), encoding="utf-8")
    data = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, -1]}).to_csv(data, index=False)
    out = tmp_path / "result.json"
    code = cli.main(
        ["validate", str(data), "--contract", str(cpath), "--json", str(out), "--quiet"]
    )
    assert code == 1
    parsed = json.loads(out.read_text(encoding="utf-8"))
    assert parsed["passed"] is False


def test_cli_validate_malformed_suite(tmp_path):
    spath = tmp_path / "broken.json"
    spath.write_text("{}", encoding="utf-8")
    data = tmp_path / "data.csv"
    pd.DataFrame({"a": [1]}).to_csv(data, index=False)
    assert cli.main(["validate", str(data), "--suite", str(spath)]) == 2
