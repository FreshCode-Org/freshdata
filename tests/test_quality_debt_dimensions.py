"""Regression tests for the duplicates and pii_risk quality-debt dimensions."""

from __future__ import annotations

import json
import sqlite3

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import privacy
from freshdata.quality import DebtItem, QualityDebtGate, _unhashable_note


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


_DUP_KEYS = ["dimension", "score", "threshold", "over_threshold", "previous", "worsening",
             "detail"]


def test_plain_frame_duplicates_serialisation_unchanged() -> None:
    """Detectable duplicates keep the exact prior scores and output."""
    _, gate = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="fail", ledger=None, verbose=False
    )
    dup = _item(gate, "duplicates")
    assert dup.assessed
    assert list(dup.to_dict()) == _DUP_KEYS
    assert dup.to_dict() == {
        "dimension": "duplicates", "score": 0.5, "threshold": 0.1, "over_threshold": True,
        "previous": None, "worsening": False,
        "detail": "50 duplicate row(s) detected (0 removed)",
    }
    assert gate.unassessed == []
    assert "not assessed" not in gate.summary()
    assert gate.total_score == pytest.approx(sum(i.score for i in gate.items))

    _, clean_gate = fd.evaluate_quality_debt(
        pd.DataFrame({"a": range(20), "b": list("xy" * 10)}), ledger=None, verbose=False
    )
    assert _item(clean_gate, "duplicates").to_dict() == {
        "dimension": "duplicates", "score": 0.0, "threshold": 0.1, "over_threshold": False,
        "previous": None, "worsening": False,
        "detail": "0 duplicate row(s) detected (0 removed)",
    }
    assert clean_gate.unassessed == []


def _list_frame() -> pd.DataFrame:
    # Rows 2 and 3 are real duplicates that cannot be detected.
    return pd.DataFrame({"id": [1, 2, 2], "tags": [["x"], ["y", "z"], ["y", "z"]]})


def _arrow_frame(kind: str) -> pd.DataFrame:
    pa = pytest.importorskip("pyarrow")
    if not hasattr(pd, "ArrowDtype"):
        pytest.skip("pd.ArrowDtype is not available in this pandas version")
    types = {
        "list": (pa.list_(pa.string()), [["x"], ["y", "z"], ["y", "z"]]),
        "struct": (pa.struct([("k", pa.int64()), ("v", pa.string())]),
                   [{"k": 1, "v": "x"}, {"k": 2, "v": "y"}, {"k": 2, "v": "y"}]),
    }
    dtype, values = types[kind]
    try:
        nested = pd.array(values, dtype=pd.ArrowDtype(dtype))
    except (TypeError, ValueError, NotImplementedError) as exc:  # pragma: no cover
        pytest.skip(f"nested ArrowDtype {kind!r} unsupported here: {exc}")
    return pd.DataFrame({"id": [1, 2, 2], "tags": nested})


def _assert_duplicates_unassessed(gate: QualityDebtGate) -> None:
    dup = _item(gate, "duplicates")
    assert not dup.assessed
    assert not dup.over
    assert not dup.worsening
    assert dup.detail == (
        "duplicate rows could not be checked: column(s) 'tags' hold unhashable values"
    )
    payload = dup.to_dict()
    assert list(payload) == _DUP_KEYS
    assert payload["score"] is None
    assert payload["over_threshold"] is None
    assert gate.unassessed == [dup]
    assert dup not in gate.warned
    # No passing evidence: excluded from the total, flagged in every output.
    assert gate.total_score == pytest.approx(
        round(sum(i.score for i in gate.items if i.assessed), 4))
    assert f"? duplicates: not assessed — {dup.detail}" in gate.summary()
    assert "n/a" in gate.to_html()
    frame = gate.to_frame()
    assert pd.isna(frame.loc[frame["dimension"] == "duplicates", "score"].iloc[0])
    decoded = json.loads(json.dumps(gate.to_dict()))
    assert decoded == gate.to_dict()
    assert next(i for i in decoded["items"] if i["dimension"] == "duplicates") == payload


@pytest.mark.parametrize("policy", ["warn", "fail", "warn_then_fail"])
def test_object_list_duplicates_unassessed(policy: str) -> None:
    _, gate = fd.evaluate_quality_debt(
        _list_frame(), debt_policy=policy, ledger=None, verbose=False
    )
    _assert_duplicates_unassessed(gate)
    assert gate.status == "pass"  # undetectable duplicates never gate on their own


@pytest.mark.parametrize("kind", ["list", "struct"])
def test_nested_arrow_duplicates_unassessed(kind: str) -> None:
    _, gate = fd.evaluate_quality_debt(
        _arrow_frame(kind), debt_policy="fail", ledger=None, verbose=False
    )
    _assert_duplicates_unassessed(gate)
    assert gate.status == "pass"


def test_object_dict_column_named_in_note() -> None:
    df = pd.DataFrame({"id": [1, 2], "meta": [{"a": 1}, {"a": 1}], "tags": [["x"], ["y"]]})
    _, gate = fd.evaluate_quality_debt(df, ledger=None, verbose=False)
    assert _item(gate, "duplicates").detail == (
        "duplicate rows could not be checked: column(s) 'meta', 'tags' hold unhashable values"
    )


def test_unhashable_note_truncates_long_column_lists() -> None:
    df = pd.DataFrame({f"c{i}": [[i], [i]] for i in range(7)})
    assert _unhashable_note(df) == (
        "column(s) 'c0', 'c1', 'c2', 'c3', 'c4' and 2 more hold unhashable values"
    )


def test_removed_duplicates_stay_a_known_lower_bound() -> None:
    """Rows removed via a hashable duplicate_subset are still counted."""
    _, gate = fd.evaluate_quality_debt(
        _list_frame(), debt_policy="warn", ledger=None, verbose=False,
        drop_duplicates=True, duplicate_subset=["id"],
    )
    dup = _item(gate, "duplicates")
    assert dup.assessed
    assert dup.score == pytest.approx(1 / 3)
    assert dup.over
    assert dup.detail == (
        "1 duplicate row(s) removed; remaining duplicate rows could not be checked: "
        "column(s) 'tags' hold unhashable values"
    )


def test_unassessed_duplicates_skip_the_ledger(tmp_path) -> None:
    ledger = str(tmp_path / "debt.sqlite")
    _, g1 = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="warn_then_fail", ledger=ledger, verbose=False
    )
    assert g1.status == "warn" and _item(g1, "duplicates").over
    _, g2 = fd.evaluate_quality_debt(
        _list_frame(), debt_policy="warn_then_fail", ledger=ledger, verbose=False
    )
    dup2 = _item(g2, "duplicates")
    assert dup2.previous == pytest.approx(0.5)
    assert not dup2.assessed and not dup2.worsening
    assert g2.status == "pass"
    conn = sqlite3.connect(ledger)
    try:
        rows = conn.execute(
            "SELECT run_id FROM debt_items WHERE dimension='duplicates'").fetchall()
        n_items = conn.execute("SELECT COUNT(*) FROM debt_items").fetchone()[0]
    finally:
        conn.close()
    assert [r[0] for r in rows] == [1]  # the unassessed run recorded no score
    assert n_items == 9 + 8
    # The next measured run escalates from the last run that measured duplicates.
    _, g3 = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="warn_then_fail", ledger=ledger, verbose=False
    )
    assert _item(g3, "duplicates").previous is None
    assert g3.status == "fail"


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
