"""Regression tests for the duplicates and pii_risk quality-debt dimensions."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import privacy
from freshdata.quality import DebtItem, QualityDebtGate


def _item(gate: QualityDebtGate, dimension: str) -> DebtItem:
    return next(i for i in gate.items if i.dimension == dimension)


def _half_duplicated() -> pd.DataFrame:
    return pd.DataFrame({"a": list(range(50)) * 2, "b": list("xy" * 25) * 2})


def test_duplicates_detected_without_removal() -> None:
    """#264: the default drop_duplicates=False must still register debt."""
    cleaned, gate = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="fail", ledger=None, verbose=False
    )
    assert len(cleaned) == 100  # nothing removed by default
    dup = _item(gate, "duplicates")
    assert dup.score == pytest.approx(0.5)
    assert dup.over
    assert dup.detail == "50 duplicate row(s) detected (0 removed)"
    assert gate.status == "fail"


def test_duplicates_removed_still_scored() -> None:
    cleaned, gate = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="fail", ledger=None, verbose=False, drop_duplicates=True
    )
    assert len(cleaned) == 50
    dup = _item(gate, "duplicates")
    assert dup.score == pytest.approx(0.5)
    assert dup.over
    assert dup.detail == "50 duplicate row(s) detected (50 removed)"


def test_no_duplicates_scores_zero() -> None:
    df = pd.DataFrame({"a": range(20), "b": list("xy" * 10)})
    _, gate = fd.evaluate_quality_debt(df, ledger=None, verbose=False)
    dup = _item(gate, "duplicates")
    assert dup.score == 0.0
    assert not dup.over
    assert dup.detail == "0 duplicate row(s) detected (0 removed)"


def test_pii_risk_counts_columns_not_matches() -> None:
    """#286: ten matching cells in one column are one PII column."""
    df = pd.DataFrame({"email": [f"user{i}@x.com" for i in range(10)], "v": range(10)})
    _, gate = fd.evaluate_quality_debt(df, debt_policy="warn", ledger=None, verbose=False)
    pii = _item(gate, "pii_risk")
    assert pii.detail == "1 potential PII column(s)"
    assert pii.score == pytest.approx(0.5)


def _entity(text: str, metadata: dict) -> privacy.PIIEntity:
    return privacy.PIIEntity(
        entity_type="EMAIL_ADDRESS",
        start=0,
        end=len(text),
        text=text,
        score=1.0,
        source="regex",
        metadata=metadata,
    )


def test_pii_risk_ignores_findings_without_column(monkeypatch: pytest.MonkeyPatch) -> None:
    entities = [
        _entity("x", {"column": "email"}),
        _entity("y", {"column": "email"}),
        _entity("z", {}),
    ]
    monkeypatch.setattr(
        privacy, "detect_pii", lambda _df: privacy.PIIScanReport(entities=entities)
    )
    df = pd.DataFrame({"email": ["a", "b"], "v": [1, 2], "w": [3, 4], "z": [5, 6]})
    _, gate = fd.evaluate_quality_debt(df, ledger=None, verbose=False)
    pii = _item(gate, "pii_risk")
    assert pii.detail == "1 potential PII column(s)"
    assert pii.score == pytest.approx(0.25)
