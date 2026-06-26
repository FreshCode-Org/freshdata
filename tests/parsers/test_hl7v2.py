"""Tests for the HL7 v2 ER7 parser."""

from __future__ import annotations

import freshdata as fd
from freshdata.parsers.hl7v2 import HL7v2Parser

MESSAGE = "\r".join([
    r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101120000||ORU^R01|MSG001|P|2.5",
    "PID|||12345^^^MRN||DOE^JOHN||19700101|M",
    "PV1|1|I|ICU^101^1||||||||||||||||VN9000",
    "OBX|1|NM|8867-4^Heart rate^LN||72|/min|60-100|N|||F|||20240101120000",
    "OBX|2|CE|44054006^Diabetes^SCT||positive||||||F",
])


def test_parses_three_frames():
    result = fd.parse_domain(MESSAGE, format="hl7v2")
    assert result.suggested_domain == "healthcare"
    assert len(result.frames["patient"]) == 1
    assert len(result.frames["encounter"]) == 1
    assert len(result.frames["observation"]) == 2
    assert result.metadata["message_type"] == "ORU^R01"


def test_patient_fields_decoded():
    patient = fd.parse_domain(MESSAGE, format="hl7v2").frames["patient"].iloc[0]
    assert patient["patient_id"] == "12345"
    assert patient["family_name"] == "DOE"
    assert patient["given_name"] == "JOHN"
    assert patient["gender"] == "M"
    assert patient["birth_date"] == "19700101"


def test_code_systems_mapped_to_uris():
    obs = fd.parse_domain(MESSAGE, format="hl7v2").frames["observation"]
    assert obs.loc[0, "code_system"] == "http://loinc.org"
    assert obs.loc[0, "code"] == "8867-4"
    assert obs.loc[0, "unit"] == "/min"
    assert obs.loc[1, "code_system"] == "http://snomed.info/sct"


def test_observations_linked_to_patient():
    obs = fd.parse_domain(MESSAGE, format="hl7v2").frames["observation"]
    assert set(obs["patient_id"]) == {"12345"}


def test_multiple_messages_increment_count():
    two = MESSAGE + "\r" + "\r".join([
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240102||ORU^R01|MSG002|P|2.5",
        "PID|||67890^^^MRN||ROE^JANE||19800202|F",
    ])
    result = fd.parse_domain(two, format="hl7v2")
    assert result.metadata["messages"] == 2
    assert set(result.frames["patient"]["patient_id"]) == {"12345", "67890"}


def test_lf_line_endings_tolerated():
    lf = MESSAGE.replace("\r", "\n")
    assert len(fd.parse_domain(lf, format="hl7v2").frames["observation"]) == 2


def test_unknown_segments_warned_not_dropped_silently():
    msg = MESSAGE + "\rZZZ|custom|segment"
    warnings = fd.parse_domain(msg, format="hl7v2").warnings
    assert any("ZZZ" in w for w in warnings)


def test_non_hl7_input_warns():
    result = HL7v2Parser().parse("this is not hl7")
    assert any("no MSH" in w for w in result.warnings)


# -- OBR (order) segment ------------------------------------------------------------

ORU_WITH_OBR = "\r".join([
    r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101120000||ORU^R01|MSG010|P|2.5",
    "PID|||12345^^^MRN||DOE^JOHN||19700101|M",
    "OBR|1|PL9001|FL7001|24323-8^Comprehensive metabolic panel^LN|||20240101080000",
    "OBX|1|NM|2345-7^Glucose^LN||95|mg/dL|70-110|N|||F",
    "OBX|2|NM|2951-2^Sodium^LN||140|mmol/L|135-145|N|||F",
])


def test_obr_produces_order_frame():
    result = fd.parse_domain(ORU_WITH_OBR, format="hl7v2")
    order = result.frames["order"]
    assert len(order) == 1
    row = order.iloc[0]
    assert row["patient_id"] == "12345"
    assert row["placer_order"] == "PL9001"
    assert row["filler_order"] == "FL7001"
    assert row["service_code"] == "24323-8"
    assert row["observed_at"] == "20240101080000"


def test_observations_link_to_their_order():
    obs = fd.parse_domain(ORU_WITH_OBR, format="hl7v2").frames["observation"]
    assert len(obs) == 2
    assert set(obs["order_id"]) == {"FL7001"}  # filler order number links OBX to OBR
