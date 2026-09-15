"""Regression tests for GTFS-ST004 stop_sequence handling (#319) and strict ICPN format (#321)."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import run_domain
from freshdata.domains.media.validator import is_valid_icpn


def _stop_times(trip_ids, stop_sequence):
    n = len(trip_ids)
    times = [f"08:{i:02d}:00" for i in range(n)]
    return pd.DataFrame(
        {
            "trip_id": trip_ids,
            "arrival_time": times,
            "departure_time": times,
            "stop_id": [f"s{i}" for i in range(n)],
            "stop_sequence": stop_sequence,
        }
    )


def _violations(df: pd.DataFrame) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, outcome = run_domain(df, "transport", gtfs_file="stop_times")
    return {r.rule_id: r.violation_rows for r in outcome.report.results if r.violated}


# -- #319: GTFS-ST004 -------------------------------------------------------------


def test_issue_319_repro_unsorted_rows_are_valid():
    st = pd.DataFrame(
        {
            "trip_id": ["T", "T", "T"],
            "arrival_time": ["08:00:00", "08:10:00", "08:05:00"],
            "departure_time": ["08:00:00", "08:10:00", "08:05:00"],
            "stop_id": ["a", "c", "b"],
            "stop_sequence": [1, 3, 2],
        }
    )
    assert "GTFS-ST004" not in _violations(st)
    assert "GTFS-ST004" not in _violations(st.sort_values("stop_sequence"))


def test_unsorted_interleaved_trips_with_gaps_pass():
    st = _stop_times(["A", "B", "A", "B", "A"], [30, 2, 10, 1, 20])
    assert "GTFS-ST004" not in _violations(st)


def test_duplicated_stop_sequence_is_flagged():
    st = _stop_times(["T", "T", "T", "U"], [1, 2, 1, 1])
    # Row 2 repeats T's sequence 1; trip U's 1 is independent.
    assert _violations(st).get("GTFS-ST004") == [2]


def test_duplicate_detected_across_numeric_spellings():
    st = _stop_times(["T", "T"], ["1", 1.0])
    assert _violations(st).get("GTFS-ST004") == [1]


def test_missing_trip_or_sequence_not_treated_as_duplicate():
    st = _stop_times([None, None, "T", "T"], [1, 1, None, None])
    assert "GTFS-ST004" not in _violations(st)


def test_duplicate_flagged_via_clean_report():
    st = _stop_times(["T", "T", "T"], [3, 1, 3])
    _, rep = fd.clean(
        st, domain="transport", gtfs_file="stop_times", return_report=True, verbose=False
    )
    st004 = [
        f
        for f in rep.domain_findings
        if f["rule_id"] == "GTFS-ST004" and f["status"] == "violated"
    ]
    assert len(st004) == 1 and st004[0]["n_violations"] == 1


# -- #321: is_valid_icpn ----------------------------------------------------------


def test_issue_321_repro():
    vals = ["tel: 036000291452", "call 0360-0029-1452 now", "036000291452"]
    assert [is_valid_icpn(v) for v in vals] == [False, False, True]


@pytest.mark.parametrize(
    "value",
    [
        "036000291452",  # UPC-A
        "4006381333931",  # EAN-13
        "0-36000-29145-2",  # hyphen-grouped UPC
        "0 36000 29145 2",  # space-grouped UPC
        "400 6381 333931",  # space-grouped EAN
        "  036000291452  ",  # surrounding whitespace
    ],
)
def test_formatted_valid_icpn_accepted(value):
    assert is_valid_icpn(value)


@pytest.mark.parametrize(
    "value",
    [
        "tel: 036000291452",
        "call 0360-0029-1452 now",
        "UPC036000291452",
        "036000291452x",
        "-036000291452",
        "036000291452-",
        "0360.0029.1452",
        "0360_0029_1452",
        "036000291452\n4006381333931",
        "",
    ],
)
def test_icpn_with_extra_text_rejected(value):
    assert not is_valid_icpn(value)
