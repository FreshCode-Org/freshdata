"""Tests for the soft-warning quality-debt ledger."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

import freshdata as fd


def _dirty() -> pd.DataFrame:
    return pd.DataFrame({
        "amount": [1.0, 2.0, None, 4.0, 100000.0, 2.0],
        "name": ["x", "x", "y", None, "z", "x"],
        "id": [1, 2, 3, 4, 5, 2],
    })


def test_returns_cleaned_and_gate() -> None:
    cleaned, gate = fd.evaluate_quality_debt(_dirty(), ledger=None)
    assert isinstance(cleaned, pd.DataFrame)
    assert gate.status in ("pass", "warn", "fail")
    assert len(gate.items) == 9  # nine dimensions


def test_persistence_and_history(tmp_path) -> None:
    ledger = str(tmp_path / "debt.sqlite")
    fd.evaluate_quality_debt(_dirty(), ledger=ledger)
    fd.evaluate_quality_debt(_dirty(), ledger=ledger)
    conn = sqlite3.connect(ledger)
    n_runs = conn.execute("SELECT COUNT(*) FROM debt_runs").fetchone()[0]
    conn.close()
    assert n_runs == 2


def test_escalation_warn_then_fail(tmp_path) -> None:
    ledger = str(tmp_path / "debt.sqlite")
    _, g1 = fd.evaluate_quality_debt(_dirty(), debt_policy="warn_then_fail", ledger=ledger)
    assert g1.status == "warn"  # first sighting only warns
    worse = pd.concat([_dirty()] * 3).reset_index(drop=True)  # many more duplicates
    _, g2 = fd.evaluate_quality_debt(worse, debt_policy="warn_then_fail", ledger=ledger)
    assert g2.status == "fail"  # repeated/worsening escalates


def test_policy_warn_never_fails(tmp_path) -> None:
    worse = pd.concat([_dirty()] * 3).reset_index(drop=True)
    _, gate = fd.evaluate_quality_debt(worse, debt_policy="warn", ledger=None)
    assert gate.status != "fail"


def test_machine_and_human_readable() -> None:
    _, gate = fd.evaluate_quality_debt(_dirty(), ledger=None)
    assert "quality-debt gate" in gate.summary()
    payload = gate.to_dict()
    assert payload["items"] and "status" in payload
    assert list(gate.to_frame().columns)[:2] == ["dimension", "score"]
    assert "<div class=\"fd-report\"" in gate.to_html()


def test_invalid_policy() -> None:
    with pytest.raises(ValueError, match="debt_policy"):
        fd.evaluate_quality_debt(_dirty(), debt_policy="nope")


def test_show_and_schema_drift_with_baseline(tmp_path, monkeypatch) -> None:
    baseline = _dirty().drop(columns=["id"])
    cleaned, gate = fd.evaluate_quality_debt(_dirty(), baseline=baseline, ledger=None)
    drift = next(i for i in gate.items if i.dimension == "schema_drift")
    assert drift.score > 0  # 'id' column added vs baseline
    monkeypatch.chdir(tmp_path)
    assert gate.show().endswith(".html")


def test_policy_fail_immediately() -> None:
    _, gate = fd.evaluate_quality_debt(_dirty(), debt_policy="fail", ledger=None)
    assert gate.status == "fail"
