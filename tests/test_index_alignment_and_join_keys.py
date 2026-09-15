"""Duplicate row labels, tz-aware date bounds and honest join-key scoring.

Covers #231 (parts 2-4: clean_text, validate_fields, suggest_join_keys on a
non-unique row index), #233 (part 3: tz-aware vs naive date bounds in
validate_fields), #272 (missing join keys score 0) and #273 (int vs
NaN-promoted float key columns overlap).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise.join_assist import _key_text

# ---------------------------------------------------------------------------
# helpers


def _dup(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with every row label duplicated, as pd.concat would leave it."""
    out = df.copy()
    out.index = [i // 2 for i in range(len(df))]
    assert not out.index.is_unique
    return out


def _issue_key(issue) -> tuple:
    return (
        issue.column,
        issue.classification,
        issue.rule,
        repr(issue.original),
        repr(issue.cleaned),
    )


def _assert_same_field_report(frame: pd.DataFrame, control, report) -> None:
    """``report`` (on ``frame``) matches ``control`` (on a RangeIndex copy)."""
    assert [_issue_key(i) for i in report.issues] == [_issue_key(i) for i in control.issues]
    assert [i.row for i in report.issues] == [
        None if i.row is None else frame.index[i.row] for i in control.issues
    ]
    assert [{k: v for k, v in c.items() if k != "row"} for c in report.normalized_cells] == [
        {k: v for k, v in c.items() if k != "row"} for c in control.normalized_cells
    ]
    assert [c["row"] for c in report.normalized_cells] == [
        frame.index[c["row"]] for c in control.normalized_cells
    ]
    assert report.inferred_types == control.inferred_types


# ---------------------------------------------------------------------------
# #231 part 2 — clean_text


def test_clean_text_duplicate_labels_keep_each_rows_value() -> None:
    df = pd.DataFrame({"t": [" alice", "bob "]}, index=[7, 7])
    out, rep = fd.clean_text(df)
    assert out["t"].tolist() == ["alice", "bob"]
    assert list(out.index) == [7, 7]
    assert [c["row"] for c in rep.changes] == [7, 7]
    assert [(c["original"], c["cleaned"]) for c in rep.changes] == [
        (" alice", "alice"),
        ("bob ", "bob"),
    ]
    assert df["t"].tolist() == [" alice", "bob "]  # input untouched


def test_clean_text_after_concat_matches_unique_index_control() -> None:
    part = pd.DataFrame({"name": [" Ann", "Bo  b", None, "ok"], "n": [1, 2, 3, 4]})
    df = pd.concat([part, part.assign(name=["x ", " y", "z", None])])
    out, rep = fd.clean_text(df)
    ctrl_out, ctrl_rep = fd.clean_text(df.reset_index(drop=True))
    assert out["name"].tolist() == ctrl_out["name"].tolist()
    assert list(out.index) == list(df.index)
    assert len(rep.changes) == len(ctrl_rep.changes) > 0
    assert [c["row"] for c in rep.changes] == [df.index[c["row"]] for c in ctrl_rep.changes]


# ---------------------------------------------------------------------------
# #231 part 3 — validate_fields


def test_validate_fields_duplicate_labels_issue_repro() -> None:
    df = pd.DataFrame({"x": ["ok", "bad id!", "ok2"]}, index=[0, 0, 1])
    rep = fd.validate_fields(df, {"x": "identifier"})
    flagged = [(i.row, i.original) for i in rep.issues]
    assert flagged == [(0, "bad id!")]
    _assert_same_field_report(
        df, fd.validate_fields(df.reset_index(drop=True), {"x": "identifier"}), rep
    )


@pytest.mark.parametrize(
    ("frame", "schema", "kwargs"),
    [
        # suspect cells + text normalization audit
        (
            pd.DataFrame({"code": [" A1", "bad id!", "B2  ", "C3", "no way", "D4"]}),
            {"code": "identifier"},
            {},
        ),
        # numeric outliers
        (
            pd.DataFrame({"amt": [10, 11, 12, 10, 11, 12, 10, 11, 5000, "abc"]}),
            {"amt": "numeric"},
            {},
        ),
        # rare allowed categories
        (
            pd.DataFrame({"c": ["a"] * 30 + ["b", "zzz"]}),
            {"c": fd.FieldSpec(allowed_values=["a", "b"])},
            {"rare_threshold": 0.05},
        ),
        # no spec: column-consensus contamination
        (pd.DataFrame({"n": ["1", "2", "3", "4", "five", "6", "7", "8"]}), None, {}),
    ],
)
def test_validate_fields_duplicate_labels_match_unique_control(frame, schema, kwargs) -> None:
    df = _dup(pd.concat([frame, frame], ignore_index=True))
    control = fd.validate_fields(df.reset_index(drop=True), schema, **kwargs)
    report = fd.validate_fields(df, schema, **kwargs)
    assert control.issues, "fixture must produce issues"
    _assert_same_field_report(df, control, report)


def test_validate_fields_policy_split_keeps_labels_on_string_index() -> None:
    df = pd.DataFrame({"x": ["ok", "bad id!", "ok2"]}, index=["r1", "r2", "r2"])
    rep = fd.validate_fields(df, {"x": "identifier"})
    assert [i.row for i in rep.issues] == ["r2"]


# ---------------------------------------------------------------------------
# #233 part 3 — tz-aware vs naive date bounds


def test_validate_fields_offset_aware_value_with_naive_bound_issue_repro() -> None:
    df = pd.DataFrame({"ts": ["2024-01-01T10:00:00+05:30", "2023-05-05"]})
    rep = fd.validate_fields(
        df, {"ts": fd.FieldSpec(semantic_type="date", min_value="1900-01-01")}
    )
    assert isinstance(rep, fd.FieldValidationReport)
    assert rep.issues == []


def test_date_bounds_compare_in_utc() -> None:
    df = pd.DataFrame(
        {
            "ts": [
                "2024-01-01T02:00:00+05:30",  # 2023-12-31T20:30Z -> below the minimum
                "2024-01-01T10:00:00+05:30",  # 2024-01-01T04:30Z -> fine
                "2024-01-05",  # naive, taken as UTC -> fine
                "2024-12-31T23:00:00-05:00",  # 2025-01-01T04:00Z -> above the maximum
            ]
        }
    )
    spec = fd.FieldSpec(
        semantic_type="date", min_value="2024-01-01", max_value="2024-12-31T23:59:59"
    )
    rep = fd.validate_fields(df, {"ts": spec})
    assert sorted((i.row, i.rule) for i in rep.issues) == [(0, "min_value"), (3, "max_value")]


def test_aware_bound_with_naive_values_and_aware_column() -> None:
    spec = fd.FieldSpec(
        semantic_type="date",
        min_value="2024-01-01T00:00:00+00:00",
        max_value=pd.Timestamp("2024-06-30", tz="UTC"),
    )
    naive = pd.DataFrame({"ts": ["2023-12-31", "2024-03-01", "2024-07-01"]})
    rep = fd.validate_fields(naive, {"ts": spec})
    assert sorted((i.row, i.rule) for i in rep.issues) == [(0, "min_value"), (2, "max_value")]

    aware = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2023-12-31", "2024-03-01", "2024-07-01"]).tz_localize(
                "Asia/Tokyo"
            )
        }
    )
    naive_spec = fd.FieldSpec(semantic_type="date", min_value="2024-01-01", max_value="2024-06-30")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = fd.validate_fields(aware, {"ts": naive_spec})
    assert sorted((i.row, i.rule) for i in rep.issues) == [(0, "min_value"), (2, "max_value")]


def test_mixed_offsets_with_bounds() -> None:
    df = pd.DataFrame(
        {
            "ts": [
                "2024-03-10T01:30:00-05:00",
                "2024-03-10T03:30:00-04:00",
                "1850-01-01T00:00:00+01:00",
            ]
        }
    )
    rep = fd.validate_fields(
        df, {"ts": fd.FieldSpec(semantic_type="date", min_value="1900-01-01")}
    )
    assert [(i.row, i.rule) for i in rep.issues] == [(2, "min_value")]


# ---------------------------------------------------------------------------
# #231 part 4 — suggest_join_keys on duplicate labels


def test_join_keys_duplicate_labels_issue_repro() -> None:
    left = pd.DataFrame({"company": ["Acme Corp", "Zenith Ltd"]}, index=[0, 0])
    right = pd.DataFrame({"company": ["Acme Corp"]})
    rep = fd.suggest_join_keys(left, right, on=["company"])
    assert [(c.left_index, c.right_index, c.score, c.status) for c in rep.candidates] == [
        (0, 0, 1.0, "match")
    ]


def test_join_keys_duplicate_labels_match_unique_control() -> None:
    base_left = pd.DataFrame(
        {
            "company": ["Acme Corp", "Acme Corp", "Zenith Ltd", "Globex Inc"],
            "country": ["US", "US", "UK", "US"],
        }
    )
    base_right = pd.DataFrame(
        {
            "company": ["Acme Corp", "Zenith Limited", "Globex Inc."],
            "country": ["US", "UK", "US"],
        }
    )
    left = _dup(base_left)
    right = base_right.copy()
    right.index = [5, 5, 6]
    kwargs = {"on": ["company"], "exact_within": ["country"], "review_threshold": 0.5}
    control = fd.suggest_join_keys(base_left, base_right, **kwargs)
    rep = fd.suggest_join_keys(left, right, **kwargs)
    assert control.candidates
    assert [
        (
            left.index[c.left_index],
            right.index[c.right_index],
            c.score,
            c.status,
            c.field_scores,
            c.block,
        )
        for c in control.candidates
    ] == [
        (c.left_index, c.right_index, c.score, c.status, c.field_scores, c.block)
        for c in rep.candidates
    ]
    assert rep.exact_keys == control.exact_keys
    assert rep.pairs_compared == control.pairs_compared


# ---------------------------------------------------------------------------
# #272 — missing keys carry no evidence


def test_nan_keys_on_both_sides_issue_repro() -> None:
    left = pd.DataFrame({"company": [np.nan, "Acme Corp"]})
    right = pd.DataFrame({"company": [np.nan, "Zenith Ltd"]})
    rep = fd.suggest_join_keys(left, right, on=["company"])
    assert all((c.left_index, c.right_index) != (0, 0) for c in rep.candidates)
    assert rep.matches == []


@pytest.mark.parametrize("missing", [None, np.nan, pd.NA, pd.NaT, ""])
def test_missing_field_scores_zero(missing) -> None:
    left = pd.DataFrame({"company": ["Acme Corp"], "city": [missing]}, dtype=object)
    right = pd.DataFrame({"company": ["Acme Corp"], "city": [missing]}, dtype=object)
    rep = fd.suggest_join_keys(left, right, on=["company", "city"], review_threshold=0.1)
    [cand] = rep.candidates
    assert cand.field_scores == {"company": 1.0, "city": 0.0}
    assert cand.score == 0.5
    city = next(k for k in rep.exact_keys if k["column"] == "city")
    assert city["overlap"] == 0.0


def test_missing_on_one_side_scores_zero() -> None:
    left = pd.DataFrame({"company": [None]}, dtype=object)
    right = pd.DataFrame({"company": ["None"]})
    rep = fd.suggest_join_keys(left, right, on=["company"], review_threshold=0.0)
    assert [c.field_scores for c in rep.candidates] == [{"company": 0.0}]


# ---------------------------------------------------------------------------
# #273 — int vs NaN-promoted float keys


def test_int_vs_float_key_columns_issue_repro() -> None:
    left = pd.DataFrame({"customer_id": [101, 102, 103]})
    right = pd.DataFrame({"customer_id": [101, 102, None]})
    assert right["customer_id"].dtype == np.float64
    rep = fd.suggest_join_keys(left, right, on=["customer_id"])
    assert rep.exact_keys == [{"column": "customer_id", "overlap": 1.0, "recommended": True}]
    assert sorted((c.left_index, c.right_index) for c in rep.matches) == [(0, 0), (1, 1)]
    assert all(c.field_scores == {"customer_id": 1.0} for c in rep.matches)


def test_int_vs_float_blocking_columns_line_up() -> None:
    left = pd.DataFrame({"region": [1, 2], "company": ["Acme Corp", "Zenith Ltd"]})
    right = pd.DataFrame({"region": [1.0, np.nan], "company": ["Acme Corp", "Zenith Ltd"]})
    rep = fd.suggest_join_keys(left, right, on=["company"], exact_within=["region"])
    assert [(c.left_index, c.right_index, c.block) for c in rep.matches] == [(0, 0, "1")]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (101.0, "101"),
        (np.float64(-7.0), "-7"),
        (np.float32(3.0), "3"),
        (0.0, "0"),
        (1.5, "1.5"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
        (float("nan"), "nan"),
        (float(2**53), str(float(2**53))),
        (1e300, str(1e300)),
        (101, "101"),
        (True, "True"),
        ("101.0", "101.0"),
    ],
)
def test_key_text(value, expected) -> None:
    assert _key_text(value) == expected
