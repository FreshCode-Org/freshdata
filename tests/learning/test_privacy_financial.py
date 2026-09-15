"""fd.learn(privacy='mask') must not store raw card numbers, IBANs or IPs."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
import freshdata.enterprise.privacy as enterprise_privacy
from freshdata.enterprise import cli
from freshdata.enterprise.config import PIIDetectionConfig
from freshdata.learning import privacy as learning_privacy
from freshdata.learning.audit import find_raw_financial_literals
from freshdata.learning.privacy import detect_sensitive_columns, is_masked_token
from freshdata.learning.profile import load_profile
from freshdata.learning.types import SENSITIVE_SEMANTIC_TYPES

CARDS = ["4111 1111 1111 1111", "5500 0000 0000 0004"]
IBANS = ["DE89370400440532013000", "GB82WEST12345698765432"]
EMAILS = ["asha@gmail.com", "ravi@yahoo.com"]


def _saved_text(profile, tmp_path: Path) -> str:
    path = tmp_path / "p.fdprofile"
    profile.save(path)
    with zipfile.ZipFile(path) as z:
        return "".join(z.read(n).decode("utf-8") for n in z.namelist())


def test_poc_cards_and_ibans_are_masked(tmp_path):
    messy = pd.DataFrame(
        {
            "card_number": [c + " " for c in CARDS] * 10,
            "iban": [i.lower() for i in IBANS] * 10,
            "email": [" " + e.upper() for e in EMAILS] * 10,
        }
    )
    clean = pd.DataFrame(
        {"card_number": CARDS * 10, "iban": IBANS * 10, "email": EMAILS * 10}
    )
    profile = fd.learn(messy, clean, min_support=2)
    text = _saved_text(profile, tmp_path)

    assert [v for v in CARDS + IBANS if v in text] == []
    assert [v for v in EMAILS if v in text] == []  # control
    sensitive = profile.audit().sensitive_columns
    assert sensitive["card_number"] == "payment_card"
    assert sensitive["iban"] == "bank_account"
    for column in ("card_number", "iban"):
        entries = profile.value_maps[column].entries
        assert entries and all(e.masked and is_masked_token(e.raw_value) for e in entries)
    assert profile.audit().raw_sensitive_literals == []


def test_ip_addresses_are_masked(tmp_path):
    ips = ["10.20.30.40", "192.168.1.77"]
    messy = pd.DataFrame({"server": [" " + ip for ip in ips] * 10})
    clean = pd.DataFrame({"server": ips * 10})
    profile = fd.learn(messy, clean, min_support=2)
    assert profile.audit().sensitive_columns == {"server": "ip_address"}
    text = _saved_text(profile, tmp_path)
    assert [ip for ip in ips if ip in text] == []


def test_value_detected_pans_with_mixed_separators(tmp_path):
    raw = ["4111-1111-1111-1111", "5500 0000 0000 0004", "4012888888881881"]
    messy = pd.DataFrame({"ref": [r + "  " for r in raw] * 10})
    clean = pd.DataFrame({"ref": raw * 10})
    profile = fd.learn(messy, clean, min_support=2)
    assert profile.audit().sensitive_columns == {"ref": "payment_card"}
    text = _saved_text(profile, tmp_path)
    assert [r for r in raw if r in text] == []


def test_card_number_name_hint_covers_undetectable_values(tmp_path):
    messy = pd.DataFrame({"card_number": ["xxxx-1234 ", "xxxx-9876 "] * 10})
    clean = pd.DataFrame({"card_number": ["xxxx-1234", "xxxx-9876"] * 10})
    profile = fd.learn(messy, clean, min_support=2)
    assert profile.audit().sensitive_columns == {"card_number": "payment_card"}
    assert "xxxx-1234" not in _saved_text(profile, tmp_path)


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("card_number", "payment_card"),
        ("Credit Card", "payment_card"),
        ("customer_pan", "payment_card"),
        ("cardNumber", "payment_card"),
        ("IBAN", "bank_account"),
        ("acct", "bank_account"),
        ("acct_id", "bank_account"),
        ("routing_number", "bank_account"),
        ("client_ip_address", "ip_address"),
        ("dob", "date_of_birth"),
        ("birth_date", "date_of_birth"),
        ("company_name", None),
        ("japan_region", None),
        ("panel", None),
        ("account_manager", None),
        ("status", None),
    ],
)
def test_token_aware_name_hints(column, expected):
    df = pd.DataFrame({column: ["alpha", "beta", "gamma"]})
    assert detect_sensitive_columns(df).get(column) == expected


def test_every_default_pii_type_is_sensitive_and_unknown_fails_closed():
    for entity in PIIDetectionConfig().entities:
        mapped = learning_privacy._entity_sensitive_type(entity, "x")
        if entity == "DATE_OF_BIRTH":
            assert mapped is None
        else:
            assert mapped in SENSITIVE_SEMANTIC_TYPES, entity
    assert learning_privacy._entity_sensitive_type("SOME_NEW_TYPE", "x") == "free_text"
    assert learning_privacy._entity_sensitive_type("zip_code", "x") == "postal_code"
    assert learning_privacy._entity_sensitive_type("ICD_CODE", "x") == "health_data"


def test_dates_count_as_date_of_birth_only_with_a_name_hint(monkeypatch):
    dates = ["1990-01-02", "1985-12-31", "2001-07-04"]
    df = pd.DataFrame({"visit_date": dates, "patient_dob": dates})
    sensitive = detect_sensitive_columns(df)
    assert "visit_date" not in sensitive
    assert sensitive["patient_dob"] == "date_of_birth"

    class _Entity:
        entity_type = "DATE_OF_BIRTH"

    class _Report:
        def by_column(self):
            return {"shipped": [_Entity()], "date_of_birth": [_Entity()]}

    monkeypatch.setattr(enterprise_privacy, "detect_pii", lambda df: _Report())
    assert learning_privacy._pii_scan_types(pd.DataFrame()) == {
        "date_of_birth": "date_of_birth"
    }


def test_non_sensitive_columns_still_replay():
    messy = pd.DataFrame(
        {
            "status": ["Shipped ", "PENDING"] * 10,
            "card_number": [c + " " for c in CARDS] * 10,
        }
    )
    clean = pd.DataFrame({"status": ["shipped", "pending"] * 10, "card_number": CARDS * 10})
    profile = fd.learn(messy, clean, min_support=2)
    assert "status" not in profile.audit().sensitive_columns
    status_entries = profile.value_maps["status"].replayable_entries()
    assert {e.raw_value for e in status_entries} == {"Shipped ", "PENDING"}
    assert profile.value_maps["card_number"].replayable_entries() == []


def _legacy_profile(monkeypatch, tmp_path: Path) -> Path:
    """A profile as older versions wrote it: raw PANs/IBANs, privacy 'mask'."""
    monkeypatch.setattr(learning_privacy, "_pii_scan_types", lambda df: {})
    messy = pd.DataFrame(
        {"ref": [c + " " for c in CARDS] * 10, "code": [i.lower() for i in IBANS] * 10}
    )
    clean = pd.DataFrame({"ref": CARDS * 10, "code": IBANS * 10})
    profile = fd.learn(messy, clean, min_support=2)
    monkeypatch.undo()
    path = tmp_path / "legacy.fdprofile"
    profile.save(path)
    assert CARDS[0] in _saved_text(profile, tmp_path / "copy")  # really raw
    return path


def test_profile_audit_flags_raw_cards_and_ibans(monkeypatch, tmp_path, capsys):
    (tmp_path / "copy").mkdir()
    path = _legacy_profile(monkeypatch, tmp_path)
    loaded = load_profile(path)
    findings = loaded.audit().raw_sensitive_literals
    assert {(f["column"], f["kind"]) for f in findings} == {
        ("ref", "payment_card"),
        ("code", "bank_account"),
    }
    assert find_raw_financial_literals(loaded) == findings

    assert cli.main(["profile", "audit", str(path)]) == 1
    out = capsys.readouterr().out
    assert "RAW CARD NUMBERS / IBANS" in out and "re-learn" in out
    assert not [v for v in CARDS + IBANS if v in out or v.lower() in out]

    assert cli.main(["profile", "audit", str(path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["raw_sensitive_literals"]) == 2


def test_profile_audit_passes_a_masked_profile(tmp_path, capsys):
    messy = pd.DataFrame({"card_number": [c + " " for c in CARDS] * 10})
    clean = pd.DataFrame({"card_number": CARDS * 10})
    path = tmp_path / "ok.fdprofile"
    fd.learn(messy, clean, min_support=2).save(path)
    assert cli.main(["profile", "audit", str(path)]) == 0
    assert "RAW CARD" not in capsys.readouterr().out
