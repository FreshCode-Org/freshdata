"""Validation must detect one bad cell, never mutate, and never lose a row.

Three properties the suite did not assert directly:

**One-value violations.** A dataset containing exactly one bad cell must be
enough to prove a rule fires. A rule tested only against a wholly-broken frame
can pass while detecting something else entirely, and a rule never tested
against a *clean* frame can pass by always failing. Each rule here is asserted
both ways.

**Non-mutation, by digest.** The existing checks use
``pd.testing.assert_frame_equal``, which compares values and dtypes but not
``.attrs`` or the index name -- so an in-place change to either would go
unnoticed. These hash content, labels, dtypes, ``attrs`` and the index name.

**Remediation integrity.** Every input row must end up in exactly one of
accepted / quarantined / rejected / needs_review, the original value must stay
recoverable, and the audit must explain the decision.
"""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

import freshdata as fd
from freshdata import ColumnRule, ValidationSuite


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "b"], "n": [1.0, 2.0, 3.0, 4.0]}
    )


def _suite(**kw) -> ValidationSuite:
    return ValidationSuite(name="t", **kw)


def _run(df: pd.DataFrame, suite: ValidationSuite) -> bool:
    """True when the suite reports a violation."""
    return not fd.run_suite(df, suite).passed


# -- one bad cell must be enough --------------------------------------------

ONE_VALUE_CASES = [
    ("unique", _suite(rules=(ColumnRule(name="id", unique=True),)),
     lambda d: d.__setitem__("id", [1, 2, 3, 3])),
    ("allowed_values", _suite(rules=(ColumnRule(name="cat", allowed_values=("a", "b")),)),
     lambda d: d.__setitem__("cat", ["a", "b", "a", "z"])),
    ("range", _suite(rules=(ColumnRule(name="n", min_value=0, max_value=10),)),
     lambda d: d.__setitem__("n", [1.0, 2.0, 3.0, 99.0])),
    ("nullable", _suite(rules=(ColumnRule(name="n", nullable=False),)),
     lambda d: d.__setitem__("n", [1.0, 2.0, 3.0, None])),
    ("max_missing_ratio", _suite(rules=(ColumnRule(name="n", max_missing_ratio=0.0),)),
     lambda d: d.__setitem__("n", [1.0, 2.0, 3.0, None])),
    ("regex", _suite(rules=(ColumnRule(name="cat", regex=r"^[ab]$"),)),
     lambda d: d.__setitem__("cat", ["a", "b", "a", "zz"])),
    ("compound_unique", _suite(compound_unique=(("id", "cat"),)),
     lambda d: (
         d.__setitem__("id", [1, 2, 3, 3]),
         d.__setitem__("cat", ["a", "b", "a", "a"]),
     )),
    ("min_rows", _suite(min_rows=4), lambda d: d.drop(index=3, inplace=True)),
]


_IDS = [c[0] for c in ONE_VALUE_CASES]


@pytest.mark.parametrize(("name", "suite", "break_it"), ONE_VALUE_CASES, ids=_IDS)
def test_one_violation_is_detected(name, suite, break_it):
    df = _frame()
    break_it(df)
    assert _run(df, suite), f"{name} did not detect its single violation"


@pytest.mark.parametrize(("name", "suite", "break_it"), ONE_VALUE_CASES, ids=_IDS)
def test_the_same_rule_passes_a_clean_frame(name, suite, break_it):
    """Guards the other direction: a rule that always fails detects nothing."""
    assert not _run(_frame(), suite), f"{name} reported a violation on clean data"


def test_a_missing_required_column_is_detected():
    assert _run(_frame(), _suite(rules=(ColumnRule(name="absent", required=True),)))


def test_extra_columns_are_detected_under_strict_columns():
    assert _run(_frame(), _suite(rules=(ColumnRule(name="id"),), strict_columns=True))


def test_max_rows_is_detected():
    df = _frame()
    df.loc[4] = [5, "a", 5.0]
    assert _run(df, _suite(max_rows=4))


# -- non-mutation, by digest ------------------------------------------------


def _digest(df: pd.DataFrame) -> str:
    """Hash content, labels, dtypes, attrs and index name.

    Stricter than ``assert_frame_equal``, which ignores ``attrs`` and the index
    name -- an in-place change to either would otherwise pass unnoticed.
    """
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(repr([str(c) for c in df.columns]).encode())
    h.update(repr([str(t) for t in df.dtypes]).encode())
    h.update(repr(sorted(df.attrs.items())).encode())
    h.update(repr(df.index.name).encode())
    return h.hexdigest()


def _annotated() -> pd.DataFrame:
    df = pd.DataFrame(
        {"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "z"], "n": [1.0, 2.0, 3.0, None]}
    )
    df.attrs["provenance"] = "unit-test"
    df.index.name = "row"
    return df


READ_ONLY_CALLS = {
    "run_suite": lambda df: fd.run_suite(
        df, ValidationSuite(name="t", rules=(ColumnRule(name="cat", allowed_values=("a", "b")),))
    ),
    "validate_fields": lambda df: fd.validate_fields(df, {"n": "numeric"}),
    "profile": fd.profile,
    "suggest_plan": fd.suggest_plan,
    "infer_roles": fd.infer_roles,
    "explain_clean": fd.explain_clean,
    "clean": lambda df: fd.clean(df, verbose=False),
}


@pytest.mark.parametrize("name", sorted(READ_ONLY_CALLS))
def test_the_input_frame_is_not_mutated(name):
    df = _annotated()
    before = _digest(df)
    READ_ONLY_CALLS[name](df)
    assert _digest(df) == before, f"{name} mutated its input"


# -- cross-field ------------------------------------------------------------


def _date_order(row):
    start, end = row.get("start_date"), row.get("end_date")
    if pd.isna(start) or pd.isna(end):
        return None
    return "start_date must be <= end_date" if start > end else None


def _min_max(row):
    low, high = row.get("min_price"), row.get("max_price")
    if pd.isna(low) or pd.isna(high):
        return None
    return "min_price must be <= max_price" if low > high else None


def _cross_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2026-01-01", "2026-05-01", "2026-03-01", None]),
            "end_date": pd.to_datetime(["2026-02-01", "2026-04-01", "2026-04-01", "2026-04-01"]),
            "min_price": [10.0, 50.0, 5.0, 1.0],
            "max_price": [20.0, 40.0, 9.0, None],
        }
    )


def test_both_cross_field_inversions_are_reported_on_the_offending_row():
    report = fd.validate_fields(_cross_frame(), cross_rules=(_date_order, _min_max))
    rows = {issue.row for issue in report.issues}
    assert rows == {1}, "only the inverted row should be reported"
    assert len([i for i in report.issues if i.row == 1]) == 2, "both rules should fire"
    assert all(i.classification == "cross_field_inconsistency" for i in report.issues)


def test_a_missing_half_of_a_pair_is_not_a_violation():
    """Row 3 has a null start_date and max_price; absence is not inversion."""
    report = fd.validate_fields(_cross_frame(), cross_rules=(_date_order, _min_max))
    assert all(issue.row != 3 for issue in report.issues)


def test_a_cross_field_failure_routes_to_review_rather_than_repair():
    report = fd.validate_fields(_cross_frame(), cross_rules=(_date_order, _min_max))
    assert {i.action for i in report.issues} == {"manual_review"}


# -- remediation integrity --------------------------------------------------


def test_every_row_lands_in_exactly_one_bucket_and_stays_recoverable():
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "amount": [10.0, "apple", 30.0, 40.0]})
    report = fd.validate_fields(df, {"amount": "numeric"})
    result = fd.apply_field_policy(df, report)

    total = (
        len(result.accepted)
        + len(result.quarantined)
        + len(result.rejected)
        + len(result.needs_review)
    )
    assert total == len(df), "a row was silently dropped"
    assert "apple" in result.quarantined["amount"].astype(str).tolist()


def test_the_audit_explains_the_decision_and_matches_the_outcome():
    df = pd.DataFrame({"id": ["a", "b", "c", "d"], "amount": [10.0, "apple", 30.0, 40.0]})
    report = fd.validate_fields(df, {"amount": "numeric"})
    result = fd.apply_field_policy(df, report)

    (entry,) = [e for e in result.audit if e["row"] == 1]
    assert entry["original"] == "apple"
    assert entry["action"] == entry["applied"] == "quarantine"
    assert entry["classification"] == "semantic_mismatch"
    assert "not silently converted" in entry["reason"]
    # The report's claim and the frames must agree.
    assert 1 not in result.accepted.index
    assert 1 in result.quarantined.index
