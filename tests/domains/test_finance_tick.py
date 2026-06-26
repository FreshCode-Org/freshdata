"""Tests for finance_mode="tick" (market tick / trade-data validation)."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains.finance import FinanceValidator


@pytest.fixture
def good_ticks() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": ["2024-03-15T09:30:00.1", "2024-03-15T09:30:00.2", "2024-03-15T09:30:00.3"],
        "symbol": ["AAPL", "AAPL", "MSFT"],
        "exchange": ["XNAS", "XNAS", "XNAS"],
        "price": [170.1, 170.2, 410.0],
        "size": [100, 200, 50],
        "bid": [170.0, 170.1, 409.9],
        "ask": [170.2, 170.3, 410.1],
        "currency": ["USD", "USD", "USD"],
    })


def _clean(df: pd.DataFrame):
    return fd.clean(df, domain="finance", finance_mode="tick",
                    return_report=True, verbose=False)


def _violated(rep, rule_id: str) -> bool:
    return any(f["rule_id"] == rule_id and f["status"] == "violated"
               for f in rep.domain_findings)


def test_valid_ticks_pass(good_ticks):
    out, rep = _clean(good_ticks)
    assert rep.domain == "finance"
    assert rep.domain_trust_score >= 0.95
    assert [f for f in rep.domain_findings if f["status"] == "violated"] == []
    pd.testing.assert_frame_equal(out, good_ticks)


def test_timestamp_required(good_ticks):
    df = good_ticks.copy()
    df.loc[1, "timestamp"] = None
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-001")


def test_symbol_required_and_never_imputed(good_ticks):
    df = good_ticks.copy()
    df.loc[1, "symbol"] = None
    out, rep = _clean(df)
    assert _violated(rep, "FIN-T-002")
    assert pd.isna(out.loc[1, "symbol"])  # IDs are never filled
    assert not [a for a in rep.domain_repairs
                if a["column"] == "symbol" and a["status"] == "applied"]


def test_timestamp_iso_and_not_future(good_ticks):
    df = good_ticks.copy()
    df.loc[0, "timestamp"] = "15/03/2024 banana"
    df.loc[1, "timestamp"] = "2099-01-01T00:00:00"
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-003")
    assert _violated(rep, "FIN-T-004")


def test_price_and_size_must_be_positive(good_ticks):
    df = good_ticks.copy()
    df.loc[0, "price"] = -5.0
    df.loc[1, "size"] = 0
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-005")
    assert _violated(rep, "FIN-T-006")


def test_currency_validated_against_iso4217_reference(good_ticks):
    df = good_ticks.copy()
    df.loc[0, "currency"] = "ZZZ"
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-007")


def test_currency_case_is_coerced_via_reference_layer(good_ticks):
    df = good_ticks.copy()
    df["currency"] = ["usd", "Usd", "eur"]  # lowercase ISO-4217 should still validate
    _, rep = _clean(df)
    assert not _violated(rep, "FIN-T-007")


def test_crossed_quote_flagged(good_ticks):
    df = good_ticks.copy()
    df.loc[0, "bid"] = 999.0  # bid > ask
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-008")


def test_duplicate_tick_flagged(good_ticks):
    df = good_ticks.copy()
    df.loc[1, ["symbol", "timestamp", "price", "size"]] = \
        df.loc[0, ["symbol", "timestamp", "price", "size"]].to_numpy()
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-009")


def test_control_completeness_is_info_audit(good_ticks):
    df = good_ticks.copy()
    df.loc[0, "exchange"] = None  # missing control dimension
    _, rep = _clean(df)
    assert _violated(rep, "FIN-T-010")
    eng = next(f for f in rep.domain_findings if f["rule_id"] == "FIN-T-010")
    assert eng["severity"] == "info"


# -- mode plumbing ------------------------------------------------------------------


def test_ledger_mode_is_default_and_unaffected():
    ledger = pd.DataFrame({
        "transaction_id": ["T1", "T1"],
        "date": ["2024-01-01", "2024-01-01"],
        "debit": [100.0, 0.0],
        "credit": [0.0, 100.0],
        "currency": ["USD", "USD"],
        "entity_id": ["E1", "E1"],
    })
    _, rep = fd.clean(ledger, domain="finance", return_report=True, verbose=False)
    assert rep.domain == "finance"
    # Ledger rule IDs (FIN-*) are present; tick rule IDs are not.
    ids = {f["rule_id"] for f in rep.domain_findings}
    assert not any(i.startswith("FIN-T-") for i in ids)


def test_invalid_finance_mode_rejected():
    with pytest.raises(ValueError, match="finance_mode must be"):
        FinanceValidator(finance_mode="intraday")


def test_finance_mode_requires_finance_domain(good_ticks):
    with pytest.raises(TypeError, match="requires domain='finance'"):
        fd.clean(good_ticks, domain="transport", finance_mode="tick", verbose=False)
