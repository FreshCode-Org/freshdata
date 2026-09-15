"""Malformed-input and UTF-8 BOM handling across the structural parsers.

Covers three gaps in the "malformed input is recorded in ``ParseResult.warnings``
rather than raising" contract:

* FHIR flatteners raising on shape-malformed resources (#313).
* ``Parser.read_text`` keeping a leading UTF-8 BOM (#314).
* GPX keeping non-finite or out-of-range coordinates (#320).
"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.parsers import fhir
from freshdata.parsers.edifact import EDIFACTParser
from freshdata.parsers.fhir import FHIRParser, _as_dict, _as_list
from freshdata.parsers.hl7v2 import HL7v2Parser

BOM = "\ufeff"
HL7 = "MSH|^~\\&|A|B|C|D|2024||ADT^A01|1|P|2.5\rPID|1||123||DOE^J||1980|F"
FHIR_PATIENT = '{"resourceType":"Patient","id":"1"}'
# Custom UNA delimiters: component "*", element "|", release "?", segment "~".
EDIFACT_UNA = (
    "UNA*|.? ~UNB|UNOA*1|SENDER|RECIPIENT|240101*1200|REF1~"
    "UNH|1|ORDERS*D*96A*UN~UNT|2|1~UNZ|1|REF1~"
)


# -- #314: UTF-8 BOM ------------------------------------------------------------------


def _sources(text: str, tmp_path):
    """Every source kind ``read_text`` accepts, each carrying *text* verbatim."""
    path = tmp_path / "input.txt"
    path.write_bytes(text.encode("utf-8"))
    return {
        "str": text,
        "bytes": text.encode("utf-8"),
        "bytearray": bytearray(text.encode("utf-8")),
        "binary file": io.BytesIO(text.encode("utf-8")),
        "text file": io.StringIO(text),
        "path": path,
    }


SOURCE_KINDS = ["str", "bytes", "bytearray", "binary file", "text file", "path"]


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_read_text_strips_leading_bom(kind, tmp_path):
    source = _sources(BOM + "MSH|x", tmp_path)[kind]
    assert HL7v2Parser().read_text(source) == "MSH|x"


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_read_text_without_bom_is_unchanged(kind, tmp_path):
    # "\n" rather than "\r": Path.read_text applies universal-newline translation.
    source = _sources("MSH|x\nPID|1", tmp_path)[kind]
    assert HL7v2Parser().read_text(source) == "MSH|x\nPID|1"


def test_read_text_keeps_bom_that_is_not_leading():
    assert HL7v2Parser().read_text(f"a{BOM}b") == f"a{BOM}b"
    assert HL7v2Parser().read_text(f"a{BOM}b".encode()) == f"a{BOM}b"


def test_issue_314_repro_hl7_and_fhir_bytes_with_bom():
    h = fd.parse_domain((BOM + HL7).encode("utf-8"), format="hl7v2")
    f = fd.parse_domain(b"\xef\xbb\xbf" + FHIR_PATIENT.encode(), format="fhir")
    assert h.metadata["messages"] == 1
    assert len(h.frames["patient"]) == 1
    assert h.warnings == []
    assert len(f.frames["patient"]) == 1
    assert f.warnings == []


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_hl7_with_bom_every_source_kind(kind, tmp_path):
    result = fd.parse_domain(_sources(BOM + HL7, tmp_path)[kind], format="hl7v2")
    assert result.metadata["messages"] == 1
    assert result.frames["patient"].iloc[0]["patient_id"] == "123"
    assert not any("MSH" in w for w in result.warnings)


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_fhir_with_bom_every_source_kind(kind, tmp_path):
    result = fd.parse_domain(_sources(BOM + FHIR_PATIENT, tmp_path)[kind], format="fhir")
    assert list(result.frames["patient"]["patient_id"]) == ["1"]
    assert result.warnings == []


@pytest.mark.parametrize("kind", SOURCE_KINDS)
def test_edifact_with_bom_honours_una_delimiters(kind, tmp_path):
    with_bom = EDIFACTParser().parse(_sources(BOM + EDIFACT_UNA, tmp_path)[kind])
    without_bom = EDIFACTParser().parse(EDIFACT_UNA)
    assert with_bom.metadata == without_bom.metadata
    assert with_bom.metadata["sender"] == "SENDER"
    assert with_bom.metadata["message_type"] == "ORDERS"
    assert with_bom.warnings == []
    pd.testing.assert_frame_equal(with_bom.frames["segments"], without_bom.frames["segments"])


# -- #313: FHIR shape-malformed resources ---------------------------------------------


def test_as_list_coercion():
    assert _as_list({"a": 1}) == [{"a": 1}]
    assert _as_list([1, 2]) == [1, 2]
    assert _as_list(None) == []
    assert _as_list("text") == []
    assert _as_list(5) == []


def test_as_dict_coercion():
    assert _as_dict({"a": 1}) == {"a": 1}
    assert _as_dict([{"a": 1}, {"b": 2}]) == {"a": 1}
    assert _as_dict(["not-a-dict", {"b": 2}]) == {}
    assert _as_dict([]) == {}
    assert _as_dict(None) == {}
    assert _as_dict("text") == {}
    assert _as_dict(3.5) == {}


ISSUE_313_CASES = {
    "address dict": {"resourceType": "Patient", "id": "1", "address": {"postalCode": "1"}},
    "coding dict": {"resourceType": "Observation", "id": "1", "code": {"coding": {"code": "x"}}},
    "valueQuantity list": {
        "resourceType": "Observation",
        "id": "1",
        "valueQuantity": [{"value": 1}],
    },
    "Encounter.class list (R5)": {
        "resourceType": "Encounter",
        "id": "1",
        "class": [{"coding": []}],
    },
    "resourceType list": {"resourceType": ["Patient"]},
}


@pytest.mark.parametrize("case", list(ISSUE_313_CASES))
def test_issue_313_repro_does_not_raise(case):
    result = fd.parse_domain(ISSUE_313_CASES[case], format="fhir")
    assert set(result.frames) == {
        "patient",
        "observation",
        "encounter",
        "condition",
        "medication_request",
    }


def test_object_valued_repeating_elements_are_read_as_single_item():
    frames = (
        FHIRParser()
        .parse(
            [
                ISSUE_313_CASES["address dict"],
                ISSUE_313_CASES["coding dict"],
                {"resourceType": "Observation", "id": "2", "valueQuantity": [{"value": 1}]},
            ]
        )
        .frames
    )
    assert frames["patient"].iloc[0]["address_postal_code"] == "1"
    assert list(frames["observation"]["code_value"])[0] == "x"
    assert list(frames["observation"]["value_quantity"])[1] == 1


def test_r5_list_valued_encounter_class_does_not_raise():
    result = FHIRParser().parse(ISSUE_313_CASES["Encounter.class list (R5)"])
    enc = result.frames["encounter"]
    assert list(enc["encounter_id"]) == ["1"]
    assert enc.iloc[0]["class_code"] is None


def test_non_string_resource_type_is_skipped_with_warning():
    result = FHIRParser().parse(
        [
            {"resourceType": ["Patient"], "id": "bad"},
            {"resourceType": {"x": 1}},
            {"resourceType": "Patient", "id": "ok"},
        ]
    )
    assert list(result.frames["patient"]["patient_id"]) == ["ok"]
    assert result.metadata["resource_types"] == {"Patient": 1}
    assert "resource with non-string resourceType (list) skipped" in result.warnings
    assert "resource with non-string resourceType (dict) skipped" in result.warnings


def test_scalar_shaped_fields_do_not_raise():
    result = FHIRParser().parse(
        [
            {"resourceType": "Patient", "id": "p", "address": 5, "maritalStatus": []},
            {
                "resourceType": "Observation",
                "id": "o",
                "valueQuantity": "72",
                "interpretation": {"coding": [{"code": "H"}]},
                "effectivePeriod": ["x"],
            },
            {
                "resourceType": "Encounter",
                "id": "e",
                "class": "IMP",
                "period": 7,
                "reasonCode": {"coding": [{"code": "R"}]},
                "hospitalization": [],
            },
            {
                "resourceType": "Condition",
                "id": "c",
                "category": {"coding": [{"code": "k"}]},
                "onsetPeriod": 1,
            },
        ]
    )
    frames = result.frames
    assert frames["patient"].iloc[0]["address_postal_code"] is None
    assert frames["observation"].iloc[0]["value_quantity"] is None
    assert frames["observation"].iloc[0]["interpretation"] == "H"
    assert frames["encounter"].iloc[0]["class_code"] is None
    assert frames["encounter"].iloc[0]["reason_code"] == "R"
    assert frames["condition"].iloc[0]["category"] == "k"


def test_flattener_exception_becomes_per_resource_warning(monkeypatch):
    def boom(resource):
        raise KeyError("bad shape")

    monkeypatch.setitem(fhir._FLATTENERS, "Patient", ("patient", boom))
    result = FHIRParser().parse(
        [
            {"resourceType": "Patient", "id": "p1"},
            {"resourceType": "Condition", "id": "c1"},
        ]
    )
    assert result.frames["patient"].empty
    assert list(result.frames["condition"]["condition_id"]) == ["c1"]
    assert "Patient resource 'p1' skipped: malformed (KeyError)" in result.warnings
    assert result.metadata["resource_types"] == {"Patient": 1, "Condition": 1}


@pytest.mark.parametrize("entry", [5, "text", None])
def test_bundle_with_non_list_entry_does_not_raise(entry):
    result = FHIRParser().parse({"resourceType": "Bundle", "entry": entry})
    assert all(df.empty for df in result.frames.values())
    assert "no FHIR resources found" in result.warnings


def test_bundle_with_object_valued_entry_reads_single_entry():
    result = FHIRParser().parse(
        {"resourceType": "Bundle", "entry": {"resource": {"resourceType": "Patient", "id": "p"}}}
    )
    assert list(result.frames["patient"]["patient_id"]) == ["p"]


def test_well_formed_resources_flatten_exactly_as_before():
    bundle = {
        "resourceType": "Bundle",
        "id": "b",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "birthDate": "1970-01-01",
                    "gender": "female",
                    "maritalStatus": {"coding": [{"code": "M"}]},
                    "address": [{"postalCode": "10001", "country": "US"}, {"postalCode": "2"}],
                    "deceasedDateTime": "2020-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o1",
                    "status": "final",
                    "subject": {"reference": "Patient/p1"},
                    "code": {"coding": [{"system": "loinc", "code": "8867-4"}], "text": "HR"},
                    "effectivePeriod": {"start": "2024-01-01"},
                    "valueQuantity": {"value": 72, "code": "/min"},
                    "interpretation": [{"coding": [{"code": "N"}]}],
                }
            },
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "e1",
                    "status": "finished",
                    "subject": {"reference": "Patient/p1"},
                    "class": {"code": "IMP"},
                    "period": {"start": "2024-01-01", "end": "2024-01-03"},
                    "reasonCode": [{"coding": [{"system": "sct", "code": "R1"}]}],
                    "hospitalization": {"admitSource": {"coding": [{"code": "emd"}]}},
                    "serviceType": {"coding": [{"code": "st"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "c1",
                    "subject": {"reference": "Patient/p1"},
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "verificationStatus": {"coding": [{"code": "confirmed"}]},
                    "category": [{"coding": [{"code": "problem-list-item"}]}],
                    "code": {"coding": [{"system": "icd", "code": "E11", "display": "DM"}]},
                    "onsetPeriod": {"start": "2019-06-01"},
                    "recordedDate": "2019-07-01",
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "m1",
                    "status": "active",
                    "intent": "order",
                    "subject": {"reference": "Patient/p1"},
                    "requester": {"reference": "Practitioner/dr1"},
                    "medicationCodeableConcept": {"coding": [{"system": "rx", "code": "860975"}]},
                    "authoredOn": "2024-01-02",
                }
            },
        ],
    }
    result = fd.parse_domain(json.dumps(bundle), format="fhir")
    records = {name: df.to_dict("records") for name, df in result.frames.items()}
    assert result.warnings == []
    assert records == {
        "patient": [
            {
                "patient_id": "p1",
                "birth_date": "1970-01-01",
                "gender": "female",
                "deceased": True,
                "deceased_date": "2020-01-01",
                "marital_status": "M",
                "address_postal_code": "10001",
                "address_country": "US",
            }
        ],
        "observation": [
            {
                "observation_id": "o1",
                "patient_id": "p1",
                "status": "final",
                "code_system": "loinc",
                "code_value": "8867-4",
                "display": "HR",
                "effective_date": "2024-01-01",
                "value_quantity": 72,
                "value_unit": "/min",
                "value_string": None,
                "interpretation": "N",
            }
        ],
        "encounter": [
            {
                "encounter_id": "e1",
                "patient_id": "p1",
                "status": "finished",
                "class_code": "IMP",
                "period_start": "2024-01-01",
                "period_end": "2024-01-03",
                "reason_code": "R1",
                "reason_code_system": "sct",
                "hospitalization_admit_source": "emd",
                "service_type": "st",
            }
        ],
        "condition": [
            {
                "condition_id": "c1",
                "patient_id": "p1",
                "clinical_status": "active",
                "verification_status": "confirmed",
                "category": "problem-list-item",
                "code_system": "icd",
                "code_value": "E11",
                "display": "DM",
                "onset_date": "2019-06-01",
                "recorded_date": "2019-07-01",
            }
        ],
        "medication_request": [
            {
                "medication_request_id": "m1",
                "patient_id": "p1",
                "status": "active",
                "intent": "order",
                "medication_system": "rx",
                "medication_code": "860975",
                "medication_display": None,
                "authored_on": "2024-01-02",
                "requester": "dr1",
            }
        ],
    }


# -- #320: GPX non-finite / out-of-range coordinates ----------------------------------


def test_issue_320_repro_nan_inf_and_non_numeric_points_skipped():
    r = fd.parse_domain(
        '<gpx><wpt lat="nan" lon="inf"><name>x</name></wpt><wpt lat="abc" lon="1"/></gpx>',
        format="gpx",
    )
    assert r.frames["waypoints"].empty
    assert r.warnings.count("wpt with missing/invalid lat/lon skipped") == 2


@pytest.mark.parametrize(
    ("lat", "lon"),
    [
        ("nan", "0"),
        ("0", "nan"),
        ("NaN", "NaN"),
        ("inf", "0"),
        ("0", "-inf"),
        ("Infinity", "1"),
        ("90.0001", "0"),
        ("-90.5", "0"),
        ("0", "180.01"),
        ("0", "-181"),
    ],
)
def test_gpx_rejects_non_finite_or_out_of_range_coordinates(lat, lon):
    r = fd.parse_domain(
        f'<gpx><wpt lat="{lat}" lon="{lon}"/><wpt lat="1" lon="2"/></gpx>', format="gpx"
    )
    assert r.frames["waypoints"].to_dict("records") == [{"lat": 1.0, "lon": 2.0}]
    assert r.warnings == ["wpt with missing/invalid lat/lon skipped"]


def test_gpx_keeps_boundary_coordinates():
    r = fd.parse_domain(
        '<gpx><wpt lat="90" lon="180"/><wpt lat="-90" lon="-180"/></gpx>', format="gpx"
    )
    assert r.frames["waypoints"].to_dict("records") == [
        {"lat": 90.0, "lon": 180.0},
        {"lat": -90.0, "lon": -180.0},
    ]
    assert r.warnings == []


def test_gpx_non_finite_track_and_route_points_skipped():
    gpx = (
        '<gpx><rte><rtept lat="1" lon="1"/><rtept lat="inf" lon="1"/></rte>'
        '<trk><trkseg><trkpt lat="2" lon="nan"/><trkpt lat="2" lon="2"/></trkseg></trk></gpx>'
    )
    r = fd.parse_domain(gpx, format="gpx")
    assert r.frames["route_points"].to_dict("records") == [
        {"lat": 1.0, "lon": 1.0, "route_index": 0}
    ]
    assert r.frames["track_points"].to_dict("records") == [
        {"lat": 2.0, "lon": 2.0, "track_index": 0}
    ]
    assert "rtept with missing/invalid lat/lon skipped" in r.warnings
    assert "trkpt with missing/invalid lat/lon skipped" in r.warnings
