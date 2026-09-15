"""HL7 v2 delimiter handling: MSH-1/MSH-2, repetitions and escape sequences (#261)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.parsers.hl7v2 import HL7v2Parser


def _parse(text: str):
    return HL7v2Parser().parse(text)


# -- issue reproduction -------------------------------------------------------------

ISSUE_MSG = (
    "MSH|^~\\&|A|B|C|D|202401011200||ORU^R01|1|P|2.5\r"
    "PID|1||12345~98765||DOE^JANE||19800101|F\r"
    "OBX|1|ST|8867-4^HR^LN||ratio 5\\S\\3 \\T\\ note||||||F"
)


def test_issue_repro_first_repetition_and_escapes():
    result = fd.parse_domain(ISSUE_MSG, format="hl7v2")
    assert result.frames["patient"]["patient_id"].iloc[0] == "12345"
    assert result.frames["observation"]["value"].iloc[0] == "ratio 5^3 & note"


def test_issue_repro_hash_field_separator_honoured():
    result = fd.parse_domain(ISSUE_MSG.replace("|", "#"), format="hl7v2")
    assert result.metadata["messages"] == 1
    assert result.metadata["message_type"] == "ORU^R01"
    assert result.warnings == []
    assert result.frames["patient"]["patient_id"].iloc[0] == "12345"
    assert result.frames["observation"]["value"].iloc[0] == "ratio 5^3 & note"


# -- custom delimiters --------------------------------------------------------------

CUSTOM = "\r".join(
    [
        "MSH#$*@!#LAB#HOSP#EHR#CLINIC#20240101##ORU$R01#MSG1#P#2.5",
        "PID#1##111$$$MRN*222$$$SSN##DOE$JOHN##19700101#M",
        "PV1#1#I#ICU$101" + "#" * 16 + "VN1",  # PV1-19 visit number
        "OBR#1#PL1#FL1#24323-8$Panel$LN###20240101080000",
        "OBX#1#CE#8867-4$Heart rate$LN##a$b!c*second#/min#####F",
    ]
)


def test_custom_delimiters_from_msh_1_and_msh_2():
    result = _parse(CUSTOM)
    assert result.metadata == {"messages": 1, "message_type": "ORU^R01"}
    assert result.warnings == []

    patient = result.frames["patient"].iloc[0]
    assert patient["patient_id"] == "111"
    assert (patient["family_name"], patient["given_name"]) == ("DOE", "JOHN")
    assert (patient["birth_date"], patient["gender"]) == ("19700101", "M")

    encounter = result.frames["encounter"].iloc[0]
    assert (encounter["class"], encounter["location"]) == ("inpatient", "ICU")
    assert encounter["visit_number"] == "VN1"

    order = result.frames["order"].iloc[0]
    assert (order["order_id"], order["service_display"]) == ("FL1", "Panel")

    obs = result.frames["observation"].iloc[0]
    assert (obs["code"], obs["display"]) == ("8867-4", "Heart rate")
    assert obs["code_system"] == "http://loinc.org"
    # Whole-field values use the standard ^ / & / ~ separators; OBX-5 keeps repetitions.
    assert obs["value"] == "a^b&c~second"
    assert (obs["unit"], obs["status"]) == ("/min", "F")


def test_partial_msh_2_defaults_missing_encoding_characters():
    # MSH-2 only declares the component separator; ~ \ & fall back to the defaults.
    msg = "MSH|$|A\rPID|||1~2||DOE$JOHN\rOBX|1|ST|X||a\\S\\b&c"
    result = _parse(msg)
    patient = result.frames["patient"].iloc[0]
    assert (patient["patient_id"], patient["given_name"]) == ("1", "JOHN")
    assert result.frames["observation"]["value"].iloc[0] == "a$b&c"


# -- repetitions --------------------------------------------------------------------


def test_repetitions_take_first_value_for_single_valued_columns():
    msg = "\r".join(
        [
            r"MSH|^~\&|A|B|C|D|20240101||ADT^A01~ADT^A04|1|P|2.5",
            "PID|1||A1^^^MRN~B2^^^SSN||DOE^JOHN~ALIAS^JACK||19700101~19700102|M~F",
            "PV1|1|I~O|ICU^1~ER^2",
            "OBX|1|NM|1-1^X^LN~2-2^Y^SCT||5~6|mg~g|||||F~C",
        ]
    )
    result = _parse(msg)
    assert result.metadata["message_type"] == "ADT^A01"
    patient = result.frames["patient"].iloc[0]
    assert patient.to_dict() == {
        "patient_id": "A1",
        "family_name": "DOE",
        "given_name": "JOHN",
        "birth_date": "19700101",
        "gender": "M",
    }
    encounter = result.frames["encounter"].iloc[0]
    assert (encounter["class_code"], encounter["class"]) == ("I", "inpatient")
    assert encounter["location"] == "ICU"
    obs = result.frames["observation"].iloc[0]
    assert (obs["code"], obs["code_system"]) == ("1-1", "http://loinc.org")
    # OBX-5 is a repeating field and keeps every repetition (see tests below).
    assert (obs["value"], obs["unit"], obs["status"]) == ("5~6", "mg", "F")


# -- OBX-5 repetitions --------------------------------------------------------------


def test_obx5_keeps_every_repetition_with_standard_delimiters():
    msg = "\r".join(
        [
            r"MSH|^~\&|A",
            "OBX|1|CE|C||1^One^LN~2^Two&x^SCT~ 3 |mg~g",
            "OBX|2|NM|C||5~6",
        ]
    )
    obs = _parse(msg).frames["observation"]
    assert list(obs["value"]) == ["1^One^LN~2^Two&x^SCT~3", "5~6"]
    assert list(obs["unit"]) == ["mg", ""]  # OBX-6 is still first repetition only


def test_obx5_custom_repetition_character_reemitted_as_tilde():
    msg = "MSH#$*@!#A\rOBX#1#CE#C##1$One$LN*2$Two!x$SCT*5#mg*g"
    obs = _parse(msg).frames["observation"].iloc[0]
    assert obs["value"] == "1^One^LN~2^Two&x^SCT~5"
    assert obs["unit"] == "mg"


def test_escaped_repetition_decoded_after_split_not_split_on():
    std = "\r".join([r"MSH|^~\&|A", r"OBX|1|ST|C||a\R\b~c|u\R\v~w"])
    obs = _parse(std).frames["observation"].iloc[0]
    assert obs["value"] == "a~b~c"
    assert obs["unit"] == "u~v"  # only one repetition: "\R\" did not split
    custom = "MSH#$*@!#A\rOBX#1#ST#C##a@R@b*c"
    # @R@ decodes to the message's repetition character, after the split on "*".
    assert _parse(custom).frames["observation"]["value"].iloc[0] == "a*b~c"


# -- escape sequences ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("escaped", "decoded"),
    [
        (r"a\F\b", "a|b"),
        (r"a\S\b", "a^b"),
        (r"a\T\b", "a&b"),
        (r"a\R\b", "a~b"),
        (r"a\E\b", "a\\b"),
        (r"\E\S\E\ ", r"\S\ "),  # \E\ is decoded once, not re-scanned
        (r"x\H\bold\N\y", r"x\H\bold\N\y"),  # unknown escapes kept verbatim
        (r"a\X0D\b", r"a\X0D\b"),
        (r"a\.br\b", r"a\.br\b"),
        ("trailing\\", "trailing\\"),  # unterminated escape left as-is
    ],
)
def test_escape_sequences_decoded_in_values(escaped, decoded):
    msg = f"MSH|^~\\&|A\rPID|||ID1||{escaped}^{escaped}\rOBX|1|ST|C||{escaped}"
    result = _parse(msg)
    assert result.frames["observation"]["value"].iloc[0] == decoded.strip()
    patient = result.frames["patient"].iloc[0]
    assert patient["family_name"] == decoded.strip()
    assert patient["given_name"] == decoded.strip()


def test_escapes_use_the_message_delimiters():
    msg = "MSH#$*@!#A\rOBX#1#ST#C##p@F@q@S@r@T@s@R@t@E@u\\S\\v"
    value = _parse(msg).frames["observation"]["value"].iloc[0]
    assert value == "p#q$r!s*t@u\\S\\v"


def test_escaped_delimiters_do_not_split_components():
    msg = "MSH|^~\\&|A\rPID|||ID\\S\\1^^^MRN||DOE\\F\\SMITH^JOHN\\R\\PAUL"
    patient = _parse(msg).frames["patient"].iloc[0]
    assert patient["patient_id"] == "ID^1"
    assert patient["family_name"] == "DOE|SMITH"
    assert patient["given_name"] == "JOHN~PAUL"


# -- multiple messages --------------------------------------------------------------


def test_each_message_uses_its_own_delimiters():
    msg = "\r".join(
        [
            r"MSH|^~\&|A|B|C|D|20240101||ADT^A01|1|P|2.5",
            "PID|||P1~X||DOE^JOHN",
            "OBX|1|ST|C1^One^LN||a\\S\\b",
            "MSH#$*@!#A#B#C#D#20240102##ORU$R01#2#P#2.5",
            "PID###P2*Y##ROE$JANE",
            "OBX#1#ST#C2$Two$SCT##c@S@d",
            r"MSH|^~\&|A|B|C|D|20240103||ADT^A08|3|P|2.5",
            "PID|||P3||POE^JIM",
        ]
    )
    result = _parse(msg)
    assert result.metadata == {"messages": 3, "message_type": "ADT^A08"}
    assert result.warnings == []
    patients = result.frames["patient"]
    assert list(patients["patient_id"]) == ["P1", "P2", "P3"]
    assert list(patients["given_name"]) == ["JOHN", "JANE", "JIM"]
    obs = result.frames["observation"]
    assert list(obs["patient_id"]) == ["P1", "P2"]
    assert list(obs["code_system"]) == ["http://loinc.org", "http://snomed.info/sct"]
    assert list(obs["value"]) == ["a^b", "c$d"]


def test_segment_id_prefix_with_other_separator_is_unknown():
    # "PIDX|..." is not a PID segment; unknown segment names keep the old warning key.
    msg = "MSH|^~\\&|A\rPIDX|||1\rZZ1|x"
    result = _parse(msg)
    assert result.frames["patient"].empty
    assert result.warnings == ["skipped unrecognized segment types: PIDX(1), ZZ1(1)"]


# -- standard messages are unchanged ------------------------------------------------

STANDARD = "\r".join(
    [
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240101120000||ORU^R01^ORU_R01|MSG001|P|2.5",
        "PID|1||12345^^^MRN||DOE^JOHN^Q||19700101|M",
        "PV1|1|I|ICU^101^1||||||||||||||||VN9000^^^HOSP",
        "OBR|1|PL9001^LAB|FL7001^LAB|24323-8^Comprehensive metabolic panel^LN|||20240101080000",
        "OBX|1|NM|8867-4^Heart rate^LN||72|/min|60-100|N|||F|||20240101120000",
        "OBX|2|CE|44054006^Diabetes^SCT||44054006^Diabetes^SCT||||||F",
        "OBX|3|ST|1234-5^Note^I10|| free text  |||||| C ",
        "ZZZ|custom|segment",
        r"MSH|^~\&|LAB|HOSP|EHR|CLINIC|20240102||ADT^A01|MSG002|P|2.5",
        "PV1|1|E",
        "PID|||67890||ROE^JANE||19800202|F",
        "OBX|1|NM|789-8^RBC^LN||4.8|10*6/uL",
    ]
)

# Frames produced for STANDARD by the parser before #261 was fixed.
EXPECTED_STANDARD = {
    "patient": [
        {
            "patient_id": "12345",
            "family_name": "DOE",
            "given_name": "JOHN",
            "birth_date": "19700101",
            "gender": "M",
        },
        {
            "patient_id": "67890",
            "family_name": "ROE",
            "given_name": "JANE",
            "birth_date": "19800202",
            "gender": "F",
        },
    ],
    "encounter": [
        {
            "patient_id": "12345",
            "visit_number": "VN9000",
            "class_code": "I",
            "class": "inpatient",
            "location": "ICU",
        },
        {
            "patient_id": "MSG2",
            "visit_number": "",
            "class_code": "E",
            "class": "emergency",
            "location": "",
        },
    ],
    "order": [
        {
            "patient_id": "12345",
            "order_id": "FL7001",
            "placer_order": "PL9001",
            "filler_order": "FL7001",
            "service_code": "24323-8",
            "service_display": "Comprehensive metabolic panel",
            "service_system": "LN",
            "observed_at": "20240101080000",
        },
    ],
    "observation": [
        {
            "patient_id": "12345",
            "order_id": "FL7001",
            "code": "8867-4",
            "display": "Heart rate",
            "code_system": "http://loinc.org",
            "value": "72",
            "unit": "/min",
            "status": "F",
            "observed_at": "20240101120000",
        },
        {
            "patient_id": "12345",
            "order_id": "FL7001",
            "code": "44054006",
            "display": "Diabetes",
            "code_system": "http://snomed.info/sct",
            "value": "44054006^Diabetes^SCT",
            "unit": "",
            "status": "F",
            "observed_at": "",
        },
        {
            "patient_id": "12345",
            "order_id": "FL7001",
            "code": "1234-5",
            "display": "Note",
            "code_system": "ICD-10",
            "value": "free text",
            "unit": "",
            "status": "C",
            "observed_at": "",
        },
        {
            "patient_id": "67890",
            "order_id": None,
            "code": "789-8",
            "display": "RBC",
            "code_system": "http://loinc.org",
            "value": "4.8",
            "unit": "10*6/uL",
            "status": "",
            "observed_at": "",
        },
    ],
}


def test_standard_message_frames_unchanged():
    result = fd.parse_domain(STANDARD, format="hl7v2")
    assert result.metadata == {"messages": 2, "message_type": "ADT^A01"}
    assert result.warnings == ["skipped unrecognized segment types: ZZZ(1)"]
    assert set(result.frames) == set(EXPECTED_STANDARD)
    for name, rows in EXPECTED_STANDARD.items():
        pd.testing.assert_frame_equal(result.frames[name], pd.DataFrame(rows))
