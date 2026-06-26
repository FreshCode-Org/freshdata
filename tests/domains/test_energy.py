"""Behavior + conformance tests for the energy (SCADA / Modbus) domain pack."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import available
from freshdata.domains.energy import EnergyValidator


@pytest.fixture
def good_energy() -> pd.DataFrame:
    """A clean Modbus telemetry frame: every rule should pass."""
    return pd.DataFrame({
        "timestamp": ["2025-06-01T00:00:00", "2025-06-01T00:00:01", "2025-06-01T00:00:02"],
        "asset_id": ["RTU-1", "RTU-1", "RTU-2"],
        "register_address": [1, 100, 200],
        "function_code": [3, 16, 4],
        "value": [50.1, 50.2, 12.0],
        "quality": ["good", "good", "uncertain"],
        "unit": ["V", "kW", "A"],
        "object_type": ["holding_register", "holding_register", "input_register"],
    })


def _clean(df: pd.DataFrame):
    return fd.clean(df, domain="energy", return_report=True, verbose=False)


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated"
               for f in rep.domain_findings)


def _applied(rep, rule_id: str) -> list[dict]:
    return [a for a in rep.domain_repairs
            if a["rule_id"] == rule_id and a["status"] == "applied"]


# -- happy path ---------------------------------------------------------------------


def test_valid_frame_passes(good_energy):
    out, rep = _clean(good_energy)
    assert rep.domain == "energy"
    assert rep.domain_trust_score >= 0.95
    nonpass = [f for f in rep.domain_findings if f["status"] == "violated"]
    assert nonpass == []
    pd.testing.assert_frame_equal(out, good_energy)


def test_all_documented_function_codes_pass(good_energy):
    df = pd.DataFrame({
        "timestamp": ["2025-06-01T00:00:00"] * 8,
        "asset_id": [f"RTU-{i}" for i in range(8)],
        "function_code": [1, 2, 3, 4, 5, 6, 15, 16],
    })
    _, rep = _clean(df)
    assert not _violated(rep, "ENG-006")


# -- schema layer -------------------------------------------------------------------


def test_timestamp_required(good_energy):
    df = good_energy.copy()
    df.loc[1, "timestamp"] = None
    out, rep = _clean(df)
    assert _violated(rep, "ENG-001")
    assert pd.isna(out.loc[1, "timestamp"])  # never invented


def test_asset_id_required_and_never_imputed(good_energy):
    df = good_energy.copy()
    df.loc[0, "asset_id"] = None
    out, rep = _clean(df)
    assert _violated(rep, "ENG-002")
    assert pd.isna(out.loc[0, "asset_id"])  # IDs are never filled
    assert not [a for a in rep.domain_repairs
                if a["column"] == "asset_id" and a["status"] == "applied"]


# -- format layer -------------------------------------------------------------------


def test_timestamp_must_be_iso(good_energy):
    df = good_energy.copy()
    df.loc[1, "timestamp"] = "01/06/2025 banana"
    _, rep = _clean(df)
    assert _violated(rep, "ENG-003")


def test_timestamp_not_in_future(good_energy):
    df = good_energy.copy()
    # Comfortably future, but within pandas' Timestamp bound (< 2262-04-11).
    future = (pd.Timestamp.now() + pd.Timedelta(days=400)).strftime("%Y-%m-%dT%H:%M:%S")
    df.loc[1, "timestamp"] = future
    _, rep = _clean(df)
    assert _violated(rep, "ENG-004")
    eng004 = next(f for f in rep.domain_findings if f["rule_id"] == "ENG-004")
    assert eng004["severity"] == "warning"


def test_register_address_out_of_range(good_energy):
    df = good_energy.copy()
    df.loc[0, "register_address"] = 70000   # > 65535
    df.loc[1, "register_address"] = -1       # < 0
    _, rep = _clean(df)
    assert _violated(rep, "ENG-005")


def test_unknown_function_code_flagged(good_energy):
    df = good_energy.copy()
    df.loc[0, "function_code"] = 99
    _, rep = _clean(df)
    assert _violated(rep, "ENG-006")
    eng006 = next(f for f in rep.domain_findings if f["rule_id"] == "ENG-006")
    assert eng006["severity"] == "error"


def test_value_must_be_numeric(good_energy):
    # Tested via the validator directly: fd.clean's generic pass would convert a
    # non-numeric sentinel to NaN before domain validation ever sees it.
    df = good_energy.copy()
    df["value"] = df["value"].astype(object)
    df.loc[1, "value"] = "FAULT"
    report = EnergyValidator().validate(df)
    assert "ENG-007" in {r.rule_id for r in report.results if r.violated}


# -- reference layer ----------------------------------------------------------------


def test_quality_synonyms_are_coerced(good_energy):
    df = good_energy.copy()
    df["quality"] = df["quality"].astype(object)
    df.loc[0, "quality"] = "ok"     # -> good
    df.loc[1, "quality"] = "UNC"    # -> uncertain
    out, rep = _clean(df)
    applied = _applied(rep, "ENG-008")
    fixes = {(a["row"], a["to"]) for a in applied}
    assert (0, "good") in fixes and (1, "uncertain") in fixes
    assert out.loc[0, "quality"] == "good" and out.loc[1, "quality"] == "uncertain"


def test_unknown_quality_is_unresolvable(good_energy):
    df = good_energy.copy()
    df["quality"] = df["quality"].astype(object)
    df.loc[0, "quality"] = "weird"
    out, rep = _clean(df)
    assert out.loc[0, "quality"] == "weird"  # left untouched
    assert any(a["rule_id"] == "ENG-008" and a["status"] == "unresolvable"
               for a in rep.domain_repairs)


def test_unknown_unit_flagged(good_energy):
    df = good_energy.copy()
    df["unit"] = df["unit"].astype(object)
    df.loc[0, "unit"] = "blorp"
    _, rep = _clean(df)
    assert _violated(rep, "ENG-009")


def test_unit_rule_skipped_when_column_absent(good_energy):
    df = good_energy.drop(columns=["unit"])
    _, rep = _clean(df)
    assert any(f["rule_id"] == "ENG-009" and f["status"] == "skipped"
               for f in rep.domain_findings)


# -- business + semantic layers -----------------------------------------------------


def test_duplicate_reading_flagged(good_energy):
    df = good_energy.copy()
    df.loc[2, ["asset_id", "register_address", "timestamp"]] = \
        df.loc[0, ["asset_id", "register_address", "timestamp"]].to_numpy()
    _, rep = _clean(df)
    assert _violated(rep, "ENG-010")


def test_function_object_inconsistency_flagged(good_energy):
    df = good_energy.copy()
    df.loc[0, "object_type"] = "coil"   # function_code 3 is a holding_register
    _, rep = _clean(df)
    assert _violated(rep, "ENG-011")


def test_function_object_rule_skipped_without_object_type(good_energy):
    df = good_energy.drop(columns=["object_type"])
    _, rep = _clean(df)
    assert any(f["rule_id"] == "ENG-011" and f["status"] == "skipped"
               for f in rep.domain_findings)


def test_low_quality_value_is_audited_not_modified(good_energy):
    df = good_energy.copy()
    df["quality"] = df["quality"].astype(object)
    df.loc[0, "quality"] = "bad"   # bad quality but a value is present
    out, rep = _clean(df)
    assert _violated(rep, "ENG-012")
    eng012 = next(f for f in rep.domain_findings if f["rule_id"] == "ENG-012")
    assert eng012["severity"] == "info"
    assert out.loc[0, "value"] == good_energy.loc[0, "value"]  # value untouched


# -- mapping, mutation, packaging ---------------------------------------------------


def test_detect_columns_with_aliases():
    messy = pd.DataFrame({
        "ts": ["2025-06-01T00:00:00"],
        "tag_name": ["RTU-9"],
        "reg_addr": [40],
        "fc": [3],
        "reading": [1.0],
        "qual": ["good"],
    })
    mapping = EnergyValidator().detect_columns(messy)
    assert mapping.actual("timestamp") == "ts"
    assert mapping.actual("asset_id") == "tag_name"
    assert mapping.actual("register_address") == "reg_addr"
    assert mapping.actual("function_code") == "fc"
    assert mapping.actual("value") == "reading"
    assert mapping.actual("quality") == "qual"


def test_validation_never_mutates_input(good_energy):
    before = good_energy.copy()
    EnergyValidator().validate(good_energy)
    pd.testing.assert_frame_equal(good_energy, before)


def test_reference_sources_are_documented():
    sources = EnergyValidator().reference_sources()
    assert sources
    assert all("disclaimer" in s and "version" in s for s in sources)


def test_standalone_import():
    v = EnergyValidator()
    assert v.domain_name == "energy"
    assert "energy" in available()
