"""Tests for the UN/EDIFACT parser."""

from __future__ import annotations

import freshdata as fd
from freshdata.parsers.edifact import EDIFACTParser

WITH_UNA = (
    "UNA:+.? 'UNB+UNOA:1+SENDER+RECIPIENT+240101:1200+REF1'"
    "UNH+1+ORDERS:D:96A:UN'BGM+220+ORDER123'UNT+3+1'UNZ+1+REF1'"
)


def test_parses_segments_and_elements():
    result = fd.parse_domain(WITH_UNA, format="edifact")
    seg = result.frames["segments"]
    assert result.metadata["segments"] == 5
    assert set(seg["tag"]) == {"UNB", "UNH", "BGM", "UNT", "UNZ"}


def test_interchange_metadata_from_unb_unh():
    meta = fd.parse_domain(WITH_UNA, format="edifact").metadata
    assert meta["sender"] == "SENDER"
    assert meta["recipient"] == "RECIPIENT"
    assert meta["message_type"] == "ORDERS"


def test_default_delimiters_without_una():
    edi = "UNB+UNOA:1+A+B+240101:1200+R'UNH+1+INVOIC:D:96A:UN'UNT+1+1'UNZ+1+R'"
    meta = fd.parse_domain(edi, format="edifact").metadata
    assert meta["sender"] == "A"
    assert meta["message_type"] == "INVOIC"


def test_release_character_escapes_delimiter():
    # The ?+ is an escaped '+' inside a value, not an element separator.
    edi = "UNB+UNOA:1+ACME?+CO+B+240101:1200+R'UNZ+1+R'"
    seg = fd.parse_domain(edi, format="edifact").frames["segments"]
    unb = seg[seg["tag"] == "UNB"]
    assert (unb["value"] == "ACME+CO").any()


def test_non_edifact_input_warns():
    result = EDIFACTParser().parse("hello world")
    assert any("not be EDIFACT" in w for w in result.warnings)
