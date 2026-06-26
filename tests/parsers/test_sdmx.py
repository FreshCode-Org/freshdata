"""Tests for the SDMX-ML parser (audit-only on unknown layouts)."""

from __future__ import annotations

import freshdata as fd
from freshdata.parsers.sdmx import SDMXParser

GENERIC = """<m:GenericData xmlns:m="ns" xmlns:g="gns">
  <m:DataSet>
    <g:Series>
      <g:SeriesKey>
        <g:Value id="FREQ" value="A"/>
        <g:Value id="REF_AREA" value="US"/>
      </g:SeriesKey>
      <g:Obs><g:ObsDimension value="2020"/><g:ObsValue value="123.4"/></g:Obs>
      <g:Obs><g:ObsDimension value="2021"/><g:ObsValue value="130.0"/></g:Obs>
    </g:Series>
  </m:DataSet>
</m:GenericData>"""

STRUCTURE_SPECIFIC = """<StructureSpecificData>
  <DataSet>
    <Obs FREQ="A" REF_AREA="US" TIME_PERIOD="2020" OBS_VALUE="9.9"/>
  </DataSet>
</StructureSpecificData>"""


def test_generic_series_to_observations():
    obs = fd.parse_domain(GENERIC, format="sdmx").frames["observations"]
    assert len(obs) == 2
    assert set(obs.columns) >= {"FREQ", "REF_AREA", "OBS_DIMENSION", "OBS_VALUE"}
    assert obs.loc[0, "FREQ"] == "A"
    assert obs.loc[0, "OBS_DIMENSION"] == "2020"
    assert obs.loc[1, "OBS_VALUE"] == "130.0"


def test_metadata_counts_series():
    result = fd.parse_domain(GENERIC, format="sdmx")
    assert result.metadata == {"series": 1, "observations": 2}


def test_structure_specific_obs_attributes():
    obs = fd.parse_domain(STRUCTURE_SPECIFIC, format="sdmx").frames["observations"]
    assert len(obs) == 1
    assert obs.loc[0, "OBS_VALUE"] == "9.9"
    assert obs.loc[0, "TIME_PERIOD"] == "2020"


def test_unrecognized_layout_is_audit_only():
    result = fd.parse_domain("<root><nothing/></root>", format="sdmx")
    assert result.frames["observations"].empty
    assert any("not recognized" in w for w in result.warnings)


def test_invalid_xml_is_audit_only():
    result = SDMXParser().parse("<<<not xml")
    assert result.frames["observations"].empty
    assert any("invalid SDMX XML" in w and "audit only" in w for w in result.warnings)
