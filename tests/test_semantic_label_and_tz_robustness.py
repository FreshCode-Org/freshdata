"""Semantic stage robustness: duplicate row labels, non-string column labels,
and tz-aware vs naive datetimes (#231, #232, #233 — semantic parts)."""

from __future__ import annotations

import warnings

import pandas as pd

import freshdata as fd
from freshdata.report import CleanReport
from freshdata.semantic.consistency import (
    _as_datetime,
    _check_date_pair_ordering,
    _check_fahrenheit_in_celsius,
    _check_future_start_dates,
    _check_policy_durations,
    _check_timezone_transitions,
)

DUP_INDEX = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def _report() -> CleanReport:
    return CleanReport(rows_before=10, rows_after=10, cols_before=2, cols_after=2)


def _clean(df: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fd.clean(df, verbose=False, return_report=True, **kwargs)


# -- #231 part 1: duplicate row index labels -----------------------------------


def _ordering_frame(index=None) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "start_date": pd.date_range("2024-01-01", periods=10).astype(str),
            "end_date": pd.date_range("2024-02-01", periods=10).astype(str),
        },
        index=index,
    )
    df.iloc[3, 1] = "2023-01-01"
    return df


def test_clean_with_duplicate_row_labels_reports_date_ordering_break():
    _, control = _clean(_ordering_frame().reset_index(drop=True), semantic_mode="auto")
    _, report = _clean(_ordering_frame(DUP_INDEX), semantic_mode="auto")

    def ordering(rep):
        return [w for w in rep.warnings if "break the date ordering" in w]

    assert ordering(control)
    assert len(ordering(report)) == len(ordering(control))
    # Position 3 carries the original (duplicated) label 1.
    assert all("(row 1)" in w for w in ordering(report))


def test_date_pair_ordering_reports_original_labels_on_duplicate_index():
    report = _report()
    _check_date_pair_ordering(_ordering_frame(DUP_INDEX), report)
    assert report.warnings
    assert all("(row 1)" in w for w in report.warnings)


def test_fahrenheit_check_handles_duplicate_row_labels():
    temps = [20.0, 21.0, 22.0, 23.0, 21.0, 71.6, 22.0, 20.0, 23.0, 21.5]
    unique = _report()
    _check_fahrenheit_in_celsius(pd.DataFrame({"temp_c": temps}), unique)
    duplicated = _report()
    _check_fahrenheit_in_celsius(pd.DataFrame({"temp_c": temps}, index=DUP_INDEX), duplicated)

    assert len(unique.warnings) == 1 and "(row 5)" in unique.warnings[0]
    assert len(duplicated.warnings) == 1 and "(row 2)" in duplicated.warnings[0]
    assert "1 value(s)" in duplicated.warnings[0]


def test_fahrenheit_check_ignores_a_tied_maximum():
    temps = [20.0, 21.0, 22.0, 23.0, 21.0, 71.6, 71.6, 20.0, 23.0, 21.5]
    report = _report()
    _check_fahrenheit_in_celsius(pd.DataFrame({"temp_c": temps}, index=DUP_INDEX), report)
    assert report.warnings == []


def test_timezone_transition_found_on_duplicate_row_labels():
    df = pd.DataFrame(
        {
            "timezone": ["UTC"] * 9 + ["UTC→IST"],
            "window": ["09:00-17:00"] * 10,
        },
        index=DUP_INDEX,
    )
    report = _report()
    _check_timezone_transitions(df, report)
    assert any("column 'timezone'" in w and "(row 4)" in w for w in report.warnings)
    assert any("column 'window'" in w and "(row 4)" in w for w in report.warnings)


def test_policy_duration_conflict_found_on_duplicate_row_labels():
    df = pd.DataFrame(
        {"retention": ["7 years", "1 year"], "purge_after": ["30 days", "2 years"]},
        index=[7, 7],
    )
    report = _report()
    _check_policy_durations(df, report)
    assert len(report.warnings) == 2
    assert all("1 row(s) (row 7)" in w for w in report.warnings)


# -- #232 part 1: non-string column labels -------------------------------------


def test_semantic_clean_accepts_integer_column_labels():
    df = pd.DataFrame({1: ["yes", "no", "yes", "no"], 2: ["twenty", "5", "6", "7"]})
    out, report = _clean(df, semantic_mode="auto", column_names=False)

    assert out[2].tolist() == [20, 5, 6, 7]
    assert "2" not in out.columns
    semantic = [a for a in report.actions if a.step == "semantic"]
    assert any(a.column == "2" and a.description.startswith("Normalized") for a in semantic)


def test_semantic_repair_skips_labels_that_stringify_alike():
    df = pd.DataFrame({1: ["a", "b", "c", "d"], "1": ["twenty", "5", "6", "7"]})
    out, report = _clean(df, semantic_mode="auto", column_names=False)

    assert list(out.columns)[:2] == [1, "1"]
    assert out["1"].tolist() == ["twenty", "5", "6", "7"]
    assert any("semantic repair skipped column(s) ['1']" in w for w in report.warnings)
    assert not [a for a in report.actions if a.step == "semantic" and a.column == "1"]


def test_semantic_clean_with_colliding_labels_still_repairs_other_columns():
    df = pd.DataFrame(
        {1: ["a", "b", "c", "d"], "1": ["x", "y", "z", "w"], "qty": ["twenty", "5", "6", "7"]}
    )
    out, _ = _clean(df, semantic_mode="auto", column_names=False)
    assert out["qty"].tolist() == [20, 5, 6, 7]


# -- #233 part 1: tz-aware vs naive datetimes ----------------------------------


def test_clean_with_tz_aware_start_and_naive_end_completes():
    df = pd.DataFrame(
        {
            "start_date": pd.date_range("2024-01-01", periods=10, tz="UTC"),
            "end_date": pd.date_range("2024-02-01", periods=10),
        }
    )
    _clean(df, semantic_mode="auto")


def test_date_pair_ordering_compares_aware_and_naive_on_utc():
    start = pd.Series(pd.date_range("2024-01-01", periods=10, tz="UTC"))
    end = pd.Series(pd.date_range("2024-02-01", periods=10))
    end.iloc[5] = pd.Timestamp("2023-12-01")
    report = _report()
    _check_date_pair_ordering(pd.DataFrame({"start_date": start, "end_date": end}), report)
    assert report.warnings
    assert all("(row 5)" in w for w in report.warnings)


def test_as_datetime_normalizes_offsets_to_naive_utc():
    aware = pd.Series(["2024-01-01T10:00:00+05:00"] * 5)
    mixed = pd.Series(["2024-03-10T01:30:00-05:00", "2024-03-10T03:30:00-04:00"] * 3)
    tz_column = pd.Series(pd.date_range("2024-01-01", periods=5, tz="Asia/Kolkata"))

    parsed_aware = _as_datetime(aware)
    parsed_mixed = _as_datetime(mixed)
    parsed_tz = _as_datetime(tz_column)

    assert parsed_aware is not None and parsed_aware.dt.tz is None
    assert parsed_aware.iloc[0] == pd.Timestamp("2024-01-01 05:00:00")
    assert parsed_mixed is not None and parsed_mixed.dt.tz is None
    assert parsed_mixed.iloc[0] == pd.Timestamp("2024-03-10 06:30:00")
    assert parsed_mixed.iloc[1] == pd.Timestamp("2024-03-10 07:30:00")
    assert parsed_tz is not None and parsed_tz.dt.tz is None
    assert parsed_tz.iloc[0] == pd.Timestamp("2023-12-31 18:30:00")


def test_date_pair_ordering_with_mixed_offset_strings():
    starts = [f"2024-03-{d:02d}T08:00:00-05:00" for d in range(1, 11)]
    ends = [f"2024-03-{d:02d}T10:00:00-04:00" for d in range(1, 11)]
    # 08:00-05:00 is 13:00 UTC; 12:00-04:00 would be 16:00 UTC (ordered), but
    # 07:00-04:00 is 11:00 UTC, before its start.
    ends[4] = "2024-03-05T07:00:00-04:00"
    report = _report()
    _check_date_pair_ordering(pd.DataFrame({"start_date": starts, "end_date": ends}), report)
    assert report.warnings
    assert all("(row 4)" in w for w in report.warnings)


def test_future_start_dates_with_tz_aware_column_and_naive_reference():
    starts = pd.Series(pd.date_range("2024-01-01", periods=10, tz="UTC"))
    starts.iloc[7] = pd.Timestamp("2025-01-01", tz="UTC")
    report = _report()
    _check_future_start_dates(
        pd.DataFrame({"start_date": starts}), {"reference_date": "2024-06-01"}, report
    )
    assert len(report.warnings) == 1 and "(row 7)" in report.warnings[0]
    assert "2024-06-01" in report.warnings[0]


def test_future_start_dates_with_naive_column_and_aware_reference():
    starts = pd.Series(pd.date_range("2024-05-25", periods=10))
    report = _report()
    # 2024-06-01T00:00+05:30 is 2024-05-31 18:30 UTC, so only 2024-06-01 onward
    # (positions 7-9) lies after it.
    _check_future_start_dates(
        pd.DataFrame({"start_date": starts}),
        {"reference_date": "2024-06-01T00:00:00+05:30"},
        report,
    )
    assert len(report.warnings) == 1
    assert "3 date(s)" in report.warnings[0]
    assert "(row 7) (row 8) (row 9)" in report.warnings[0]
