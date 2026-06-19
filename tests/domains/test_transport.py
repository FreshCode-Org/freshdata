"""Conformance tests for the transport (GTFS) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import DomainError, UnknownDomainError
from freshdata.domains.transport import TransportValidator


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated" for f in rep.domain_findings)


@pytest.fixture
def feed():
    return {
        "stops": pd.DataFrame({
            "stop_id": ["S1", "S2", "S3"],
            "stop_name": ["A", "B", "C"],
            "stop_lat": [40.7, 40.8, 40.9],
            "stop_lon": [-74.0, -74.1, -74.2],
        }),
        "routes": pd.DataFrame({
            "route_id": ["R1", "R2"],
            "route_short_name": ["1", "2"],
            "route_type": [3, 1],
        }),
        "trips": pd.DataFrame({
            "route_id": ["R1", "R2"],
            "service_id": ["WK", "WK"],
            "trip_id": ["T1", "T2"],
        }),
        "stop_times": pd.DataFrame({
            "trip_id": ["T1", "T1", "T2"],
            "arrival_time": ["08:00:00", "25:30:00", "09:00:00"],
            "departure_time": ["08:00:00", "25:31:00", "09:00:00"],
            "stop_id": ["S1", "S2", "S1"],
            "stop_sequence": [1, 2, 1],
        }),
    }


def test_full_feed_happy_path(feed):
    out, rep = fd.clean(feed, domain="transport", return_report=True, verbose=False)
    assert isinstance(out, dict) and set(out) == set(feed)     # dict in -> dict out
    assert rep.domain == "transport"
    assert rep.domain_trust_score >= 0.95
    assert not [f for f in rep.domain_findings if f["status"] == "violated"]


def test_single_file_happy_path(feed):
    out, rep = fd.clean(feed["stops"], domain="transport", gtfs_file="stops",
                        return_report=True, verbose=False)
    assert isinstance(out, pd.DataFrame)                       # frame in -> frame out
    assert rep.domain_trust_score >= 0.95


def test_single_frame_requires_file_selector(feed):
    with pytest.raises(TypeError, match="gtfs_file"):
        fd.clean(feed["stops"], domain="transport", verbose=False)


def test_unknown_single_file_is_rejected():
    with pytest.raises(DomainError, match="unsupported GTFS file"):
        fd.clean(
            pd.DataFrame({"value": [1]}),
            domain="transport",
            gtfs_file="stopz.txt",
            verbose=False,
        )


@pytest.mark.parametrize("field", ["stop_id", "stop_lat", "stop_lon"])
def test_required_stop_field_missing(feed, field):
    df = feed["stops"].drop(columns=[field])
    _, rep = fd.clean(df, domain="transport", gtfs_file="stops",
                      return_report=True, verbose=False)
    assert any("MISSING_REQUIRED_FIELD" in f["message"] and f["status"] == "violated"
               for f in rep.domain_findings)


def test_required_stop_field_partial_null_is_reported(feed):
    stops = feed["stops"].copy()
    stops.loc[0, "stop_id"] = None
    _, rep = fd.clean(
        stops, domain="transport", gtfs_file="stops", return_report=True, verbose=False
    )
    assert _violated(rep, "GTFS-S001")


def test_format_violations_coords_and_time(feed):
    stops = feed["stops"].copy()
    stops.loc[0, "stop_lat"] = 999.0          # GTFS-S002 out of range
    stops.loc[1, "stop_lon"] = -500.0         # GTFS-S003 out of range
    _, rep_s = fd.clean(stops, domain="transport", gtfs_file="stops",
                        return_report=True, verbose=False)
    assert _violated(rep_s, "GTFS-S002")
    assert _violated(rep_s, "GTFS-S003")

    st = feed["stop_times"].copy()
    st.loc[0, "arrival_time"] = "8am"         # GTFS-ST002 bad format
    _, rep_t = fd.clean(st, domain="transport", gtfs_file="stop_times",
                        return_report=True, verbose=False)
    assert _violated(rep_t, "GTFS-ST002")


def test_post_midnight_times_are_valid(feed):
    # Hours may exceed 23 in GTFS; 25:30:00 must not be flagged.
    _, rep = fd.clean(feed["stop_times"], domain="transport", gtfs_file="stop_times",
                      return_report=True, verbose=False)
    assert not _violated(rep, "GTFS-ST002")


def test_reference_violation_route_type(feed):
    routes = feed["routes"].copy()
    routes.loc[0, "route_type"] = 99          # not a valid GTFS route_type
    _, rep = fd.clean(routes, domain="transport", gtfs_file="routes",
                      return_report=True, verbose=False)
    assert _violated(rep, "GTFS-R002")


def test_business_uniqueness_and_ordering(feed):
    stops = feed["stops"].copy()
    stops.loc[2, "stop_id"] = "S1"            # duplicate -> GTFS-S004
    _, rep_s = fd.clean(stops, domain="transport", gtfs_file="stops",
                        return_report=True, verbose=False)
    assert _violated(rep_s, "GTFS-S004")

    st = feed["stop_times"].copy()
    st.loc[1, "departure_time"] = "25:29:00"  # earlier than arrival -> GTFS-ST003
    st.loc[1, "stop_sequence"] = 1            # not increasing within T1 -> GTFS-ST004
    _, rep_t = fd.clean(st, domain="transport", gtfs_file="stop_times",
                        return_report=True, verbose=False)
    assert _violated(rep_t, "GTFS-ST003")
    assert _violated(rep_t, "GTFS-ST004")


def test_cross_file_reference_full_feed_only(feed):
    feed["trips"].loc[1, "route_id"] = "R99"  # not in routes
    out, rep = fd.clean(feed, domain="transport", return_report=True, verbose=False)
    assert _violated(rep, "GTFS-T003")        # fires in full-feed mode
    # In a full feed the rule is recorded once per file (skipped for non-trips);
    # the trips entry is the one that actually ran.
    t003 = next(f for f in rep.domain_findings
                if f["rule_id"] == "GTFS-T003" and f["status"] == "violated")
    assert t003["n_violations"] == 1 and t003["file"] == "trips"

    # single-file trips: no routes context -> T003 is skipped, not violated
    _, rep2 = fd.clean(feed["trips"], domain="transport", gtfs_file="trips",
                       return_report=True, verbose=False)
    assert any(f["rule_id"] == "GTFS-T003" and f["status"] == "skipped"
               for f in rep2.domain_findings)


def test_txt_file_names_are_normalized_for_rules_and_cross_references(feed):
    txt_feed = {
        "routes.txt": feed["routes"],
        "trips.txt": feed["trips"].assign(route_id=["R1", "R99"]),
    }
    out, rep = fd.clean(txt_feed, domain="transport", return_report=True, verbose=False)
    assert set(out) == set(txt_feed)
    finding = next(
        f
        for f in rep.domain_findings
        if f["rule_id"] == "GTFS-T003" and f["status"] == "violated"
    )
    assert finding["file"] == "trips.txt" and finding["n_violations"] == 1


def test_extra_gtfs_files_are_preserved_and_reported_as_unvalidated(feed):
    data = {
        **feed,
        "agency.txt": pd.DataFrame({
            "agency_name": ["Example Transit"],
            "agency_url": ["https://example.test"],
            "agency_timezone": ["UTC"],
        }),
    }
    out, rep = fd.clean(data, domain="transport", return_report=True, verbose=False)
    assert "agency.txt" in out
    assert any("agency.txt" in warning and "not covered" in warning for warning in rep.warnings)


def test_repairs_are_flag_only_and_ids_untouched(feed):
    stops = feed["stops"].copy()
    stops.loc[0, "stop_lat"] = 999.0
    out, rep = fd.clean(stops, domain="transport", gtfs_file="stops",
                        return_report=True, verbose=False)
    assert out.loc[0, "stop_lat"] == 999.0    # coordinate never mutated
    assert all(a["status"] != "applied" for a in rep.domain_repairs)


def test_id_safety_null_stop_id_never_filled(feed):
    stops = feed["stops"].copy()
    stops.loc[0, "stop_id"] = None
    out, rep = fd.clean(stops, domain="transport", gtfs_file="stops",
                        return_report=True, verbose=False)
    # The id is never imputed, and no repair ever targets it.
    assert pd.isna(out.loc[0, "stop_id"])
    assert not [r for r in rep.domain_repairs
                if r["column"] == "stop_id" and r["status"] == "applied"]


def test_unknown_domain_lists_available(feed):
    with pytest.raises(UnknownDomainError):
        fd.clean(feed, domain="bogus_domain")


def test_standalone_import():
    assert TransportValidator(gtfs_file="stops").multi_frame is True   # importable on its own
