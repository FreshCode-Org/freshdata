"""Tests for the parser registry, ParseResult, and the fd.parse_domain / clean_domain_file API."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.parsers import (
    Parser,
    ParseResult,
    UnknownParserError,
    available,
    get_parser,
    register,
)
from freshdata.parsers import registry as parser_registry

HL7 = "\r".join([
    r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101||ORU^R01|MSG001|P|2.5",
    "PID|||12345^^^MRN||DOE^JOHN||19700101|M",
    "OBX|1|NM|8867-4^Heart rate^LN||72|/min|||||F",
])


def test_builtin_formats_registered():
    assert {"hl7v2", "gpx", "sdmx", "edifact"} <= set(available())


def test_get_parser_returns_instance():
    assert isinstance(get_parser("gpx"), Parser)


def test_unknown_format_raises():
    with pytest.raises(UnknownParserError, match="unknown parser format"):
        get_parser("nope")


def test_register_custom_parser_and_unregister():
    class DummyParser(Parser):
        format = "dummy"

        def parse(self, source):
            return ParseResult("dummy", {"d": pd.DataFrame({"x": [1]})})

    try:
        register("dummy", DummyParser)
        assert "dummy" in available()
        assert fd.parse_domain("anything", format="dummy").frame.equals(pd.DataFrame({"x": [1]}))
    finally:
        parser_registry._REGISTERED.pop("dummy", None)


def test_register_rejects_non_parser():
    with pytest.raises(TypeError, match="Parser subclass"):
        register("bad", dict)


def test_parse_result_frame_single_and_multi():
    single = ParseResult("f", {"only": pd.DataFrame({"a": [1, 2]})})
    assert len(single.frame) == 2
    multi = ParseResult("f", {"a": pd.DataFrame(), "b": pd.DataFrame()})
    with pytest.raises(ValueError, match="use .frames"):
        _ = multi.frame


def test_parse_result_to_dict_is_jsonish():
    result = fd.parse_domain(HL7, format="hl7v2")
    d = result.to_dict()
    assert d["format"] == "hl7v2"
    assert d["frames"]["observation"] == 1
    assert d["suggested_domain"] == "healthcare"
    assert isinstance(d["warnings"], list)


def test_parse_domain_accepts_path(tmp_path):
    p = tmp_path / "msg.hl7"
    p.write_text(HL7)
    assert len(fd.parse_domain(str(p), format="hl7v2").frames["observation"]) == 1


def test_clean_domain_file_without_domain_returns_parse_result():
    result = fd.clean_domain_file(HL7, format="hl7v2")
    assert isinstance(result, ParseResult)


def test_clean_domain_file_cleans_selected_frame():
    out = fd.clean_domain_file(HL7, format="hl7v2", domain="healthcare",
                               frame="patient", fhir_resource="Patient")
    assert isinstance(out, pd.DataFrame)
    assert "patient_id" in out.columns


def test_clean_domain_file_multiframe_without_frame_raises():
    with pytest.raises(ValueError, match="pass frame="):
        fd.clean_domain_file(HL7, format="hl7v2", domain="healthcare")
