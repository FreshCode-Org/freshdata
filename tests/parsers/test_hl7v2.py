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


def test_order_frame_has_service_display_and_system():
    order = fd.parse_domain(ORU_WITH_OBR, format="hl7v2").frames["order"].iloc[0]
    assert order["service_display"] == "Comprehensive metabolic panel"
    assert order["service_system"] == "LN"


def test_obr_uses_placer_order_when_filler_absent():
    # OBR-3 (filler order) is empty: current_order falls back to placer_order (OBR-2).
    msg = "\r".join([
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101||ORU^R01|MSG020|P|2.5",
        "PID|||99999^^^MRN||SMITH^JOHN||19800101|M",
        "OBR|1|PL5555||17861-6^Albumin^LN|||20240101090000",
        "OBX|1|NM|17861-6^Albumin^LN||4.5|g/dL|3.5-5.0|N|||F",
    ])
    result = fd.parse_domain(msg, format="hl7v2")
    order = result.frames["order"].iloc[0]
    # No filler order -> placer order is used as order_id.
    assert order["order_id"] == "PL5555"
    obs = result.frames["observation"].iloc[0]
    assert obs["order_id"] == "PL5555"


def test_multiple_obr_segments_each_links_correct_obx():
    msg = "\r".join([
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101||ORU^R01|MSG030|P|2.5",
        "PID|||11111^^^MRN||DOE^JANE||19900101|F",
        "OBR|1|PL1111|FL1111|24323-8^CMP^LN|||20240101080000",
        "OBX|1|NM|2345-7^Glucose^LN||95|mg/dL|70-110|N|||F",
        "OBR|2|PL2222|FL2222|24357-6^CBC^LN|||20240101090000",
        "OBX|1|NM|789-8^RBC^LN||4.8|10*6/uL|4.2-5.4|N|||F",
        "OBX|2|NM|787-2^MCV^LN||90|fL|80-100|N|||F",
    ])
    result = fd.parse_domain(msg, format="hl7v2")
    orders = result.frames["order"]
    assert len(orders) == 2
    obs = result.frames["observation"]
    assert len(obs) == 3
    # OBX after first OBR links to FL1111.
    assert obs.iloc[0]["order_id"] == "FL1111"
    # OBX after second OBR links to FL2222.
    assert obs.iloc[1]["order_id"] == "FL2222"
    assert obs.iloc[2]["order_id"] == "FL2222"


def test_obr_resets_order_id_on_new_message():
    # Second message starts with no OBR: its OBX should have order_id=None.
    msg = "\r".join([
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101||ORU^R01|MSG040|P|2.5",
        "PID|||A001^^^MRN||DOE^JOHN||19700101|M",
        "OBR|1|PL9000|FL9000|8867-4^Heart rate^LN|||20240101080000",
        "OBX|1|NM|8867-4^Heart rate^LN||72|/min|60-100|N|||F",
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240102||ORU^R01|MSG041|P|2.5",
        "PID|||B001^^^MRN||ROE^JANE||19800202|F",
        "OBX|1|NM|8867-4^Heart rate^LN||80|/min|60-100|N|||F",
    ])
    result = fd.parse_domain(msg, format="hl7v2")
    obs = result.frames["observation"]
    # First OBX (linked to OBR in message 1) has the order id.
    assert obs.iloc[0]["order_id"] == "FL9000"
    # Second OBX (in message 2, no OBR) has no order id.
    assert obs.iloc[1]["order_id"] is None


def test_obr_before_pid_uses_message_index_patient_id():
    # OBR appears before PID: patient_id should fall back to MSG{index}.
    msg = "\r".join([
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101||ORU^R01|MSG050|P|2.5",
        "OBR|1|PL0001|FL0001|8867-4^Heart rate^LN|||20240101080000",
        "PID|||55555^^^MRN||DOE^JOHN||19700101|M",
    ])
    result = fd.parse_domain(msg, format="hl7v2")
    order = result.frames["order"].iloc[0]
    # Before PID is parsed, patient_id uses MSG{index}.
    assert order["patient_id"] == "MSG1"


def test_message_without_obr_has_empty_order_frame():
    # Standard ADT with no OBR should still yield an order frame (empty).
    result = fd.parse_domain(MESSAGE, format="hl7v2")
    assert "order" in result.frames
    assert len(result.frames["order"]) == 0
