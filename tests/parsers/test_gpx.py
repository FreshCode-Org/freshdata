"""Tests for the GPX parser."""

from __future__ import annotations

import freshdata as fd
from freshdata.parsers.gpx import GPXParser

GPX = """<?xml version="1.0"?>
<gpx version="1.1" xmlns="http://www.topografix.com/GPX/1/1">
  <wpt lat="40.0" lon="-73.0"><name>Start</name><ele>10</ele></wpt>
  <rte><rtept lat="40.05" lon="-73.05"/><rtept lat="40.06" lon="-73.06"/></rte>
  <trk><trkseg>
    <trkpt lat="40.1" lon="-73.1"><ele>11</ele><time>2024-01-01T00:00:00Z</time></trkpt>
    <trkpt lat="40.2" lon="-73.2"/>
  </trkseg></trk>
</gpx>"""


def test_parses_waypoints_routes_tracks():
    result = fd.parse_domain(GPX, format="gpx")
    assert result.suggested_domain == "transport"
    assert len(result.frames["waypoints"]) == 1
    assert len(result.frames["route_points"]) == 2
    assert len(result.frames["track_points"]) == 2
    assert result.metadata == {"tracks": 1, "routes": 1, "waypoints": 1}


def test_point_attributes_and_children():
    wpt = fd.parse_domain(GPX, format="gpx").frames["waypoints"].iloc[0]
    assert wpt["lat"] == 40.0 and wpt["lon"] == -73.0
    assert wpt["name"] == "Start"
    assert wpt["ele"] == "10"


def test_track_points_carry_track_index_and_time():
    trk = fd.parse_domain(GPX, format="gpx").frames["track_points"]
    assert set(trk["track_index"]) == {0}
    assert trk.loc[0, "time"] == "2024-01-01T00:00:00Z"


def test_gpx_10_namespace_supported():
    gpx10 = (
        '<gpx version="1.0" xmlns="http://www.topografix.com/GPX/1/0">'
        '<wpt lat="1.0" lon="2.0"/></gpx>'
    )
    assert len(fd.parse_domain(gpx10, format="gpx").frames["waypoints"]) == 1


def test_invalid_coordinates_warned_and_skipped():
    bad = ('<gpx xmlns="http://www.topografix.com/GPX/1/1">'
           '<wpt lat="oops" lon="2.0"/></gpx>')
    result = fd.parse_domain(bad, format="gpx")
    assert result.frames["waypoints"].empty
    assert any("lat/lon" in w for w in result.warnings)


def test_malformed_xml_returns_warning_not_exception():
    result = GPXParser().parse("definitely not xml <<<")
    assert all(df.empty for df in result.frames.values())
    assert any("invalid GPX XML" in w for w in result.warnings)


def test_empty_gpx_warns():
    result = fd.parse_domain('<gpx xmlns="http://www.topografix.com/GPX/1/1"/>', format="gpx")
    assert any("no waypoints" in w for w in result.warnings)
