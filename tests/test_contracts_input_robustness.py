"""Input-robustness regressions for the contract / baseline entry points.

Covers mixed-format datetime bounds (#242), polars input to ``diff_schema``
(#274), NaN and mixed-type keys in ``compare_to_baseline(key=...)`` (#276),
non-string column labels (#232 part 4), tz-aware vs naive baseline timestamps
(#233 part 5) and duplicate column labels (#265 part 2).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.contracts import _resolve_label


def _contract(*columns: fd.ColumnContract, **kwargs: object) -> fd.DataContract:
    return fd.DataContract("c", tuple(columns), **kwargs)  # type: ignore[arg-type]


def _check_ids(report: fd.DriftReport) -> list[str]:
    return [f.check_id for f in report.findings]


# ── #242: mixed-format datetime strings ─────────────────────────────────────────


def test_max_datetime_catches_mixed_format_strings():
    df = pd.DataFrame({"signup": ["2024-01-15", "12/31/2030"]})
    rep = fd.enforce_contract(
        df, _contract(fd.ColumnContract("signup", max_datetime="2025-01-01"))
    )
    assert not rep.passed
    (finding,) = [f for f in rep.findings if f.check_id == "contract.max_datetime"]
    assert "(1/2 = 50.00%)" in finding.message


def test_min_datetime_catches_mixed_format_strings():
    df = pd.DataFrame({"signup": ["2024-01-15", "01/02/1990"]})
    rep = fd.enforce_contract(
        df, _contract(fd.ColumnContract("signup", min_datetime="2000-01-01"))
    )
    assert "contract.min_datetime" in _check_ids(rep)
    assert not rep.passed


def test_unparseable_datetimes_are_reported_as_warning():
    df = pd.DataFrame({"signup": ["2024-01-15", "not a date", "nope", "nope", None]})
    rep = fd.enforce_contract(
        df, _contract(fd.ColumnContract("signup", max_datetime="2025-01-01"))
    )
    (finding,) = [f for f in rep.findings if f.check_id == "contract.unparseable_datetime"]
    assert finding.level == "warning"
    assert finding.status == "warned"
    assert finding.current_value == 3
    assert finding.details["n_unparseable"] == 3
    assert finding.details["examples"] == ["not a date", "nope"]
    assert rep.passed  # warning only; the parseable value is within bounds


def test_all_unparseable_values_still_warn():
    df = pd.DataFrame({"signup": ["x", "y"]})
    rep = fd.enforce_contract(
        df, _contract(fd.ColumnContract("signup", min_datetime="2000-01-01"))
    )
    assert _check_ids(rep) == ["contract.unparseable_datetime"]
    assert rep.passed


def test_mixed_timezone_strings_still_checked_without_future_warning():
    df = pd.DataFrame({"ts": ["2024-01-01T00:00:00+01:00", "2031-06-01T00:00:00-05:00"]})
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        rep = fd.enforce_contract(
            df, _contract(fd.ColumnContract("ts", max_datetime="2030-01-01"))
        )
    assert "contract.max_datetime" in _check_ids(rep)
    assert "contract.unparseable_datetime" not in _check_ids(rep)


def test_datetime_dtype_column_has_no_unparseable_finding():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=3)})
    rep = fd.enforce_contract(df, _contract(fd.ColumnContract("ts", max_datetime="2024-01-02")))
    assert _check_ids(rep) == ["contract.max_datetime"]


# ── #274: diff_schema on polars ─────────────────────────────────────────────────


def test_diff_schema_accepts_polars_frame():
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    df = pl.DataFrame({"a": [1, None]})
    contract = _contract(fd.ColumnContract("a", nullable=False))
    rep = fd.diff_schema(df, contract=contract)
    assert "schema.nullable_change" in _check_ids(rep)
    assert rep.contract_results["nullable_changed"] == ["a"]


# ── #276: key-level changes with NaN / mixed-type keys ──────────────────────────


def test_nan_key_is_matched_in_identical_frames():
    df = pd.DataFrame({"account": [1.0, np.nan], "balance": [10, 20]})
    kc = fd.compare_to_baseline(df.copy(), df.copy(), key="account").key_changes
    assert kc is not None
    assert (kc["added"], kc["removed"], kc["changed"], kc["unchanged"]) == (0, 0, 0, 2)
    assert kc["baseline_records"] == kc["current_records"] == 2


def test_nan_key_counts_as_one_key_for_changes():
    base = pd.DataFrame({"account": [1.0, np.nan], "balance": [10, 20]})
    cur = pd.DataFrame({"account": [1.0, np.nan, 3.0], "balance": [10, 99, 5]})
    kc = fd.compare_to_baseline(cur, base, key="account").key_changes
    assert (kc["added"], kc["removed"], kc["changed"], kc["unchanged"]) == (1, 0, 1, 1)


def test_mixed_int_str_keys_do_not_crash():
    df = pd.DataFrame({"sku": [1001, "A-17"], "qty": [1, 2]})
    kc = fd.compare_to_baseline(df.copy(), df.copy(), key="sku").key_changes
    assert (kc["added"], kc["removed"], kc["changed"], kc["unchanged"]) == (0, 0, 0, 2)

    cur = pd.DataFrame({"sku": ["A-17", 1001, "B-2"], "qty": [5, 1, 3]})
    kc = fd.compare_to_baseline(cur, df.copy(), key="sku").key_changes
    assert (kc["added"], kc["removed"], kc["changed"], kc["unchanged"]) == (1, 0, 1, 1)


def test_composite_key_with_nan_component():
    base = pd.DataFrame({"a": [1, 1, 2], "b": ["x", np.nan, np.nan], "v": [1, 2, 3]})
    cur = base.iloc[[2, 0, 1]].reset_index(drop=True)
    kc = fd.compare_to_baseline(cur, base, key=["a", "b"]).key_changes
    assert (kc["added"], kc["removed"], kc["changed"], kc["unchanged"]) == (0, 0, 0, 3)


# ── #232 part 4: non-string column labels ───────────────────────────────────────


@pytest.fixture()
def int_labelled() -> pd.DataFrame:
    return pd.DataFrame({0: [1, 2, 3], 1: ["a", "b", "c"]})


@pytest.mark.parametrize("name", [0, "0"])
def test_enforce_contract_finds_integer_labels(int_labelled, name):
    rep = fd.enforce_contract(
        int_labelled, _contract(fd.ColumnContract(name, dtype="int", min_value=0))
    )
    assert [f for f in rep.findings if f.status != "passed"] == []
    assert rep.contract_results == {name: True}


def test_enforce_contract_value_checks_run_on_integer_labels(int_labelled):
    rep = fd.enforce_contract(int_labelled, _contract(fd.ColumnContract("0", min_value=2)))
    assert "contract.min_value" in _check_ids(rep)


def test_enforce_contract_strict_columns_with_integer_labels(int_labelled):
    contract = _contract(fd.ColumnContract(0), fd.ColumnContract("1"), strict_columns=True)
    rep = fd.enforce_contract(int_labelled, contract)
    assert rep.passed, _check_ids(rep)


def test_enforce_contract_missing_non_string_label_is_a_finding(int_labelled):
    rep = fd.enforce_contract(int_labelled, _contract(fd.ColumnContract(7)))
    assert _check_ids(rep) == ["contract.missing_required"]


def test_compound_unique_with_integer_labels():
    df = pd.DataFrame({0: [1, 1], 1: ["a", "a"]})
    rep = fd.enforce_contract(df, _contract(compound_unique=(("0", 1),)))  # type: ignore[arg-type]
    assert _check_ids(rep) == ["contract.compound_unique"]
    assert "duplicate" in rep.findings[0].message


def test_diff_schema_integer_labels(int_labelled):
    rep = fd.diff_schema(int_labelled, contract=_contract(fd.ColumnContract("x")))
    assert rep.contract_results["removed"] == ["x"]
    assert rep.contract_results["unexpected"] == ["0", "1"]

    rep = fd.diff_schema(int_labelled, contract=_contract(fd.ColumnContract("0", nullable=False)))
    assert rep.contract_results["removed"] == []
    assert rep.contract_results["unexpected"] == ["1"]


def test_diff_schema_rename_candidate_with_integer_labels(int_labelled):
    contract = _contract(fd.ColumnContract("col0", dtype="int"))
    rep = fd.diff_schema(int_labelled, contract=contract)  # no crash on df["0"]
    assert rep.contract_results["removed"] == ["col0"]


def test_baseline_distribution_drift_with_integer_labels():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({0: rng.normal(0, 1, 500)})
    cur = pd.DataFrame({0: rng.normal(5, 1, 500)})
    rep = fd.compare_to_baseline(cur, fd.build_baseline(base, name="b"))
    assert "0" in rep.distribution_drift
    assert "drift.ks" in _check_ids(rep)


def test_key_changes_with_integer_key_label():
    df = pd.DataFrame({0: [1, 2], 1: ["a", "b"]})
    kc = fd.compare_to_baseline(df.copy(), df.copy(), key=0).key_changes  # type: ignore[arg-type]
    assert (kc["added"], kc["removed"], kc["changed"]) == (0, 0, 0)


def test_resolve_label_prefers_exact_then_unique_str_match():
    df = pd.DataFrame({0: [1], "b": [2]})
    assert _resolve_label(df, 0) == 0
    assert _resolve_label(df, "0") == 0
    assert _resolve_label(df, "b") == "b"
    assert _resolve_label(df, "zzz") is None


def test_resolve_label_raises_on_ambiguous_string_form():
    class Label:
        def __init__(self, tag: int) -> None:
            self.tag = tag

        def __str__(self) -> str:
            return "same"

    df = pd.DataFrame([[1, 2]], columns=pd.Index([Label(1), Label(2)], dtype=object))
    with pytest.raises(ValueError, match="ambiguous"):
        _resolve_label(df, "same")


def test_labels_that_collide_as_strings_are_rejected():
    df = pd.DataFrame({1: [10.0], "1": ["a"]})
    with pytest.raises(ValueError, match="distinct when converted to str"):
        fd.enforce_contract(df, _contract(fd.ColumnContract("1")))


# ── #233 part 5: tz-aware vs naive baseline timestamps ──────────────────────────


def test_timezone_dropped_upstream_reports_drift_instead_of_crashing():
    base = fd.build_baseline(
        pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, tz="UTC")}), name="b"
    )
    cur = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40)})
    rep = fd.compare_to_baseline(cur, base)
    (finding,) = [f for f in rep.findings if f.check_id == "drift.timezone_change"]
    assert finding.level == "warning"
    assert (finding.baseline_value, finding.current_value) == ("tz-aware", "tz-naive")
    assert "drift.datetime_range" not in _check_ids(rep)


def test_timezone_added_upstream_still_compares_ranges_in_utc():
    base = fd.build_baseline(
        pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40)}), name="b"
    )
    cur = pd.DataFrame({"ts": pd.date_range("2024-01-10", periods=40, tz="Asia/Kolkata")})
    rep = fd.compare_to_baseline(cur, base)
    ids = _check_ids(rep)
    assert "drift.timezone_change" in ids
    (rng,) = [f for f in rep.findings if f.check_id == "drift.datetime_range"]
    assert rng.metric == "max_timestamp"


def test_same_timezone_awareness_emits_no_timezone_finding():
    frame = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, tz="UTC")})
    rep = fd.compare_to_baseline(frame, fd.build_baseline(frame, name="b"))
    assert "drift.timezone_change" not in _check_ids(rep)


# ── #265 part 2: duplicate column labels ────────────────────────────────────────


@pytest.fixture()
def duplicated() -> pd.DataFrame:
    return pd.DataFrame([[1, 2, 3]], columns=["age", "age", "id"])


@pytest.mark.parametrize(
    "call",
    [
        lambda df: fd.build_baseline(df, name="b"),
        lambda df: fd.compare_to_baseline(df, pd.DataFrame({"id": [1]})),
        lambda df: fd.compare_to_baseline(pd.DataFrame({"id": [1]}), df),
        lambda df: fd.enforce_contract(df, _contract(fd.ColumnContract("id"))),
        lambda df: fd.diff_schema(df, contract=_contract(fd.ColumnContract("id"))),
    ],
    ids=["build_baseline", "compare_current", "compare_baseline", "enforce", "diff_schema"],
)
def test_duplicate_labels_raise_value_error(duplicated, call):
    with pytest.raises(ValueError, match=r"duplicated label\(s\): \['age'\]"):
        call(duplicated)


def test_validate_suite_rejects_duplicate_labels(duplicated):
    suite = fd.ValidationSuite(name="s", rules=[fd.ColumnRule("id", min_value=0)])
    with pytest.raises(ValueError, match="duplicated label"):
        fd.validate(duplicated, suite=suite)
