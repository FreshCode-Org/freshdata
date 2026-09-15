"""The GPX/SDMX DTD/entity guard holds in every document encoding.

Regression tests for a bypass where a UTF-16 (or UTF-32) document hid its
``<!DOCTYPE``/``<!ENTITY`` markers from a raw byte search, letting ElementTree
expand internal entities.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import freshdata as fd
from freshdata.parsers.base import ParseResult, _detect_xml_encoding, _expat_rejects_dtd
from freshdata.parsers.gpx import GPXParser

# Two nesting levels: &a1; expands to 100 characters when entities are processed.
ENTITIES = '<!ENTITY a0 "AAAAAAAAAA"><!ENTITY a1 "' + "&a0;" * 10 + '">'
REF = "&a1;"


def _gpx_text(declared: str | None, doctype: str) -> str:
    decl = f'<?xml version="1.0" encoding="{declared}"?>' if declared else ""
    return f'{decl}{doctype}<gpx><wpt lat="1" lon="2"><name>{REF}</name></wpt></gpx>'


def _sdmx_text(declared: str | None, doctype: str) -> str:
    decl = f'<?xml version="1.0" encoding="{declared}"?>' if declared else ""
    return (
        f"{decl}{doctype}<DataSet><Series><SeriesKey>"
        f'<Value id="REF" value="{REF}"/></SeriesKey>'
        '<Obs><ObsDimension value="2020"/><ObsValue value="1"/></Obs></Series></DataSet>'
    )


# (id, declared encoding, bytes encoder)
ENCODINGS = [
    ("utf-16-bom", "UTF-16", lambda s: s.encode("utf-16")),
    ("utf-16-le-no-bom", "UTF-16", lambda s: s.encode("utf-16-le")),
    ("utf-16-be-no-bom", "UTF-16", lambda s: s.encode("utf-16-be")),
    ("utf-16-be-bom", "UTF-16", lambda s: b"\xfe\xff" + s.encode("utf-16-be")),
    ("utf-32-bom", "UTF-32", lambda s: s.encode("utf-32")),
    ("utf-32-le-no-bom", "UTF-32", lambda s: s.encode("utf-32-le")),
    ("utf-32-be-no-bom", "UTF-32", lambda s: s.encode("utf-32-be")),
    ("utf-8-sig", "UTF-8", lambda s: s.encode("utf-8-sig")),
    ("iso-8859-1", "ISO-8859-1", lambda s: s.encode("iso-8859-1")),
]


def _max_cell_len(result: ParseResult) -> int:
    return max(
        (len(str(v)) for df in result.frames.values() for v in df.to_numpy().ravel()),
        default=0,
    )


def _assert_rejected(result: ParseResult, label: str) -> None:
    assert any(f"unsafe {label} XML" in w and "not allowed" in w for w in result.warnings), (
        result.warnings
    )
    assert all(df.empty for df in result.frames.values())
    assert _max_cell_len(result) <= len(REF)


@pytest.mark.parametrize(("declared", "encode"), [e[1:] for e in ENCODINGS],
                         ids=[e[0] for e in ENCODINGS])
def test_gpx_entity_expansion_rejected_in_every_encoding(declared, encode):
    doc = encode(_gpx_text(declared, f"<!DOCTYPE g [{ENTITIES}]>"))
    _assert_rejected(fd.parse_domain(doc, format="gpx"), "GPX")


@pytest.mark.parametrize(("declared", "encode"), [e[1:] for e in ENCODINGS],
                         ids=[e[0] for e in ENCODINGS])
def test_sdmx_entity_expansion_rejected_in_every_encoding(declared, encode):
    doc = encode(_sdmx_text(declared, f"<!DOCTYPE g [{ENTITIES}]>"))
    result = fd.parse_domain(doc, format="sdmx")
    _assert_rejected(result, "SDMX")
    assert all("audit only" in w for w in result.warnings)


@pytest.mark.parametrize("doctype", ["<!doctype", "<!DocType"])
def test_lowercase_and_mixed_case_doctype_in_utf16(doctype):
    doc = _gpx_text("UTF-16", f"{doctype} g [{ENTITIES}]>").encode("utf-16")
    _assert_rejected(fd.parse_domain(doc, format="gpx"), "GPX")


def test_external_system_dtd_without_internal_subset_in_utf16():
    text = _gpx_text("UTF-16", '<!DOCTYPE gpx SYSTEM "http://example.invalid/gpx.dtd">')
    doc = text.replace(REF, "x").encode("utf-16")
    _assert_rejected(fd.parse_domain(doc, format="gpx"), "GPX")


def test_parameter_entity_only_dtd_in_utf16():
    text = _gpx_text("UTF-16", '<!DOCTYPE g [<!ENTITY % p "x">]>').replace(REF, "x")
    _assert_rejected(fd.parse_domain(text.encode("utf-16-le"), format="gpx"), "GPX")


def test_utf16_without_bom_or_declaration():
    # expat reads a NUL in the first two bytes as UTF-16, declaration or not.
    for codec in ("utf-16-le", "utf-16-be"):
        doc = _gpx_text(None, f"<!DOCTYPE g [{ENTITIES}]>").encode(codec)
        _assert_rejected(fd.parse_domain(doc, format="gpx"), "GPX")
        leading_space = (" " + _gpx_text(None, f"<!DOCTYPE g [{ENTITIES}]>")).encode(codec)
        _assert_rejected(fd.parse_domain(leading_space, format="gpx"), "GPX")


def test_utf16_with_trailing_odd_byte_still_rejected():
    doc = _gpx_text("UTF-16", f"<!DOCTYPE g [{ENTITIES}]>").encode("utf-16") + b"\x00"
    _assert_rejected(fd.parse_domain(doc, format="gpx"), "GPX")


def test_utf16_declaration_over_utf8_bytes():
    with_dtd = _gpx_text("UTF-16", f"<!DOCTYPE g [{ENTITIES}]>")
    _assert_rejected(fd.parse_domain(with_dtd.encode("utf-8"), format="gpx"), "GPX")
    _assert_rejected(fd.parse_domain(with_dtd, format="gpx"), "GPX")

    without_dtd = _gpx_text("UTF-16", "").replace(REF, "x")
    result = fd.parse_domain(without_dtd.encode("utf-8"), format="gpx")
    assert all(df.empty for df in result.frames.values())
    assert any("invalid GPX XML" in w for w in result.warnings), result.warnings


def test_ebcdic_prefix_is_rejected_as_unsupported():
    doc = '<?xml version="1.0" encoding="IBM037"?><gpx/>'.encode("cp037")
    assert doc.startswith(b"\x4c\x6f\xa7\x94")
    result = fd.parse_domain(doc, format="gpx")
    assert all(df.empty for df in result.frames.values())
    assert any("unsafe GPX XML" in w and "unsupported XML encoding" in w
               for w in result.warnings), result.warnings


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig"])
def test_benign_non_utf8_gpx_still_parses(codec):
    text = (
        '<?xml version="1.0" encoding="UTF-16"?>'
        if codec.startswith("utf-16") else '<?xml version="1.0" encoding="UTF-8"?>'
    ) + '<gpx><wpt lat="40.0" lon="-73.0"><name>Start</name></wpt></gpx>'
    result = fd.parse_domain(text.encode(codec), format="gpx")
    assert not any("XML" in w for w in result.warnings), result.warnings
    wp = result.frames["waypoints"]
    assert len(wp) == 1
    assert wp["name"].iloc[0] == "Start"
    assert wp["lat"].iloc[0] == 40.0


def test_benign_utf16_sdmx_still_parses():
    text = _sdmx_text("UTF-16", "").replace(REF, "A")
    result = fd.parse_domain(text.encode("utf-16"), format="sdmx")
    obs = result.frames["observations"]
    assert len(obs) == 1
    assert obs["REF"].iloc[0] == "A"


def test_path_and_str_sources(tmp_path: Path):
    doc = _gpx_text("UTF-16", f"<!DOCTYPE g [{ENTITIES}]>").encode("utf-16")
    path = tmp_path / "evil.gpx"
    path.write_bytes(doc)

    _assert_rejected(fd.parse_domain(path, format="gpx"), "GPX")
    # clean_domain_file turns an existing str path into a Path.
    _assert_rejected(fd.clean_domain_file(str(path), format="gpx"), "GPX")
    # A str value passed to parse_domain is document content.
    _assert_rejected(
        fd.parse_domain(_gpx_text(None, f"<!DOCTYPE g [{ENTITIES}]>"), format="gpx"), "GPX"
    )

    benign = tmp_path / "ok.gpx"
    benign.write_bytes(
        '<gpx><wpt lat="1" lon="2"><name>ok</name></wpt></gpx>'.encode("utf-16")
    )
    assert len(fd.parse_domain(benign, format="gpx").frames["waypoints"]) == 1
    assert len(fd.clean_domain_file(str(benign), format="gpx").frames["waypoints"]) == 1


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        (b"\x00\x00\xfe\xff<", ("utf-32-be", 4)),
        (b"\xff\xfe\x00\x00<", ("utf-32-le", 4)),
        (b"\xfe\xff\x00<", ("utf-16-be", 2)),
        (b"\xff\xfe<\x00", ("utf-16-le", 2)),
        (b"\xef\xbb\xbf<", ("utf-8", 3)),
        (b"\x00\x00\x00<", ("utf-32-be", 0)),
        (b"<\x00\x00\x00", ("utf-32-le", 0)),
        (b"\x00<\x00?", ("utf-16-be", 0)),
        (b"<\x00?\x00", ("utf-16-le", 0)),
        (b'<?xml version="1.0" encoding=\'latin-1\'?><a/>', ("latin-1", 0)),
        (b"<a/>", ("utf-8", 0)),
    ],
    ids=[
        "bom-utf-32-be", "bom-utf-32-le", "bom-utf-16-be", "bom-utf-16-le", "bom-utf-8",
        "utf-32-be", "utf-32-le", "utf-16-be", "utf-16-le", "declared", "default",
    ],
)
def test_detect_xml_encoding(prefix, expected):
    assert _detect_xml_encoding(prefix) == expected


def test_expat_pass_is_authoritative_and_never_raises_expat_error():
    doc = _gpx_text("UTF-16", f"<!DOCTYPE g [{ENTITIES}]>").encode("utf-16")
    with pytest.raises(ValueError, match="not allowed"):
        _expat_rejects_dtd(doc)
    with pytest.raises(ValueError, match="not allowed"):
        _expat_rejects_dtd(b'<!DOCTYPE g SYSTEM "x.dtd"><g/>')
    # Malformed input and unknown encodings are left for ElementTree to report.
    _expat_rejects_dtd(b"<gpx><unclosed></gpx>")
    _expat_rejects_dtd(b'<?xml version="1.0" encoding="x-unknown-enc"?><gpx/>')
    _expat_rejects_dtd("<gpx/>".encode("utf-16"))


def test_safe_xml_binary_returns_the_original_bytes():
    doc = '<gpx><wpt lat="1" lon="2"/></gpx>'.encode("utf-16")
    assert GPXParser().open_safe_xml_binary(doc).read() == doc
