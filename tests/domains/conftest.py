"""Shared fixtures for domain-pack tests."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def good_finance() -> pd.DataFrame:
    """A valid, balanced finance ledger: two transactions, debits == credits."""
    return pd.DataFrame({
        "transaction_id": ["T1", "T1", "T2", "T2"],
        "date": ["2024-01-15", "2024-01-15", "2024-02-01", "2024-02-01"],
        "account_code": ["AC1001", "AC2002", "AC1001", "AC3003"],
        "debit": [100.00, 0.00, 50.00, 0.00],
        "credit": [0.00, 100.00, 0.00, 50.00],
        "currency": ["USD", "USD", "EUR", "EUR"],
        "description": ["sale", "sale", "refund", "refund"],
        "entity_id": ["E1", "E1", "E2", "E2"],
    })


@pytest.fixture
def messy_finance() -> pd.DataFrame:
    """A ledger with messy-but-mappable column names (aliases + casing)."""
    return pd.DataFrame({
        "Txn ID": ["T1", "T1"],
        "Posting Date": ["2024-01-15", "2024-01-15"],
        "DR": [100.0, 0.0],
        "CR": [0.0, 100.0],
        "CCY": ["USD", "USD"],
    })
