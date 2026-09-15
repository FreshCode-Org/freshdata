"""Regression tests for the duplicates and pii_risk quality-debt dimensions."""

from __future__ import annotations

import importlib
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
    # A baseline keeps schema_drift and category_churn measured (#16).
    _, gate = fd.evaluate_quality_debt(
        _half_duplicated(), baseline=_half_duplicated(), debt_policy="fail", ledger=None,
        verbose=False,
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

    clean = pd.DataFrame({"a": range(20), "b": list("xy" * 10)})
    _, clean_gate = fd.evaluate_quality_debt(
        clean, baseline=clean.copy(), ledger=None, verbose=False
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
        _list_frame(), baseline=_list_frame(), debt_policy=policy, ledger=None, verbose=False
    )
    _assert_duplicates_unassessed(gate)
    assert gate.status == "pass"  # undetectable duplicates never gate on their own


@pytest.mark.parametrize("kind", ["list", "struct"])
def test_nested_arrow_duplicates_unassessed(kind: str) -> None:
    frame = _arrow_frame(kind)
    _, gate = fd.evaluate_quality_debt(
        frame, baseline=frame.copy(), debt_policy="fail", ledger=None, verbose=False
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
    # No baseline: schema_drift and category_churn are unassessed on both runs.
    assert n_items == 7 + 6
    # The next measured run escalates from, and compares against, the last run
    # that measured duplicates.
    _, g3 = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="warn_then_fail", ledger=ledger, verbose=False
    )
    dup3 = _item(g3, "duplicates")
    assert dup3.previous == pytest.approx(0.5)
    assert not dup3.worsening
    assert g3.status == "fail"


def test_worsening_compares_against_last_measured_run(tmp_path) -> None:
    ledger = str(tmp_path / "debt.sqlite")
    _, g1 = fd.evaluate_quality_debt(
        _half_duplicated(), debt_policy="warn", ledger=ledger, verbose=False
    )
    first = _item(g1, "duplicates").score
    fd.evaluate_quality_debt(_list_frame(), debt_policy="warn", ledger=ledger, verbose=False)
    mostly_duplicated = pd.DataFrame({"a": [1] * 99 + [2], "b": ["x"] * 100})
    _, g3 = fd.evaluate_quality_debt(
        mostly_duplicated, debt_policy="warn", ledger=ledger, verbose=False
    )
    dup3 = _item(g3, "duplicates")
    assert dup3.previous == pytest.approx(first)
    assert dup3.score > first
    assert dup3.worsening


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


# -- Dimensions that were never measured are not assessed (#16) ---------------


def _raise(exc: type[Exception]):
    def boom(*_args, **_kwargs):
        raise exc("simulated failure")

    return boom


def _assert_unassessed(gate: QualityDebtGate, dimension: str, detail: str) -> None:
    item = _item(gate, dimension)
    assert not item.assessed
    assert not item.over
    assert not item.worsening
    assert item.detail == detail
    payload = item.to_dict()
    assert list(payload) == _DUP_KEYS
    assert payload["score"] is None
    assert payload["over_threshold"] is None
    assert item in gate.unassessed
    assert item not in gate.warned
    assert gate.total_score == pytest.approx(
        round(sum(i.score for i in gate.items if i.assessed), 4))
    assert f"? {dimension}: not assessed — {detail}" in gate.summary()
    frame = gate.to_frame()
    assert pd.isna(frame.loc[frame["dimension"] == dimension, "score"].iloc[0])
    assert json.loads(json.dumps(gate.to_dict())) == gate.to_dict()


def _assert_other_items_unchanged(gate: QualityDebtGate, reference: QualityDebtGate,
                                  dimension: str) -> None:
    assert [i.to_dict() for i in gate.items if i.dimension != dimension] == [
        i.to_dict() for i in reference.items if i.dimension != dimension]


def _baseline_for(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


@pytest.mark.parametrize("exc", [RuntimeError, ValueError, MemoryError])
def test_profiling_failure_is_not_assessed(monkeypatch: pytest.MonkeyPatch, exc) -> None:
    df = pd.DataFrame({"a": ["1", "2", "x"]})
    kwargs = {"baseline": _baseline_for(df), "debt_policy": "fail", "ledger": None,
              "verbose": False, "include_trust_score": False}
    _, reference = fd.evaluate_quality_debt(df, **kwargs)
    assert _item(reference, "type_instability").assessed

    # freshdata.profile is shadowed by the fd.profile function; patch the module.
    monkeypatch.setattr(importlib.import_module("freshdata.profile"), "build_profile",
                        _raise(exc))
    _, gate = fd.evaluate_quality_debt(df, **kwargs)
    _assert_unassessed(
        gate, "type_instability",
        f"column types could not be checked: profiling failed ({exc.__name__})")
    assert [i.dimension for i in gate.unassessed] == ["type_instability"]
    _assert_other_items_unchanged(gate, reference, "type_instability")
    assert "n/a" in gate.to_html()


@pytest.mark.parametrize("exc", [RuntimeError, ImportError])
def test_pii_scan_failure_is_not_assessed(monkeypatch: pytest.MonkeyPatch, exc) -> None:
    df = pd.DataFrame({"email": ["a@b.com", "c@d.com"]})
    kwargs = {"baseline": _baseline_for(df), "debt_policy": "fail", "ledger": None,
              "verbose": False, "include_trust_score": False}
    _, reference = fd.evaluate_quality_debt(df, **kwargs)
    measured = _item(reference, "pii_risk")
    assert measured.assessed and measured.over  # a real scan finds the email column

    monkeypatch.setattr(privacy, "detect_pii", _raise(exc))
    _, gate = fd.evaluate_quality_debt(df, **kwargs)
    _assert_unassessed(gate, "pii_risk",
                       f"PII could not be checked: scan unavailable ({exc.__name__})")
    assert [i.dimension for i in gate.unassessed] == ["pii_risk"]
    _assert_other_items_unchanged(gate, reference, "pii_risk")
    # The scan that never ran is not reported as a clean 0.0.
    assert "0 potential PII column(s)" not in gate.summary()


def test_no_baseline_drift_and_churn_are_not_assessed() -> None:
    df = pd.DataFrame({"id": [1, 2, 3, 4], "cat": ["a", "b", "a", "c"]})
    _, gate = fd.evaluate_quality_debt(df, debt_policy="fail", ledger=None, verbose=False)
    for dimension in ("schema_drift", "category_churn"):
        _assert_unassessed(gate, dimension, "no baseline supplied")
    assert [i.dimension for i in gate.unassessed] == ["schema_drift", "category_churn"]

    _, with_baseline = fd.evaluate_quality_debt(
        df, baseline=df.copy(), debt_policy="fail", ledger=None, verbose=False)
    assert with_baseline.unassessed == []
    assert _item(with_baseline, "schema_drift").to_dict()["score"] == 0.0
    assert _item(with_baseline, "category_churn").to_dict()["score"] == 0.0


def test_missing_dimension_is_not_assessed(monkeypatch: pytest.MonkeyPatch) -> None:
    quality = importlib.import_module("freshdata.quality")
    real = quality._score_debt

    def without_outliers(*args, **kwargs):
        scores = real(*args, **kwargs)
        del scores["outlier_spikes"]
        return scores

    monkeypatch.setattr(quality, "_score_debt", without_outliers)
    df = pd.DataFrame({"a": range(20), "b": list("xy" * 10)})
    _, gate = fd.evaluate_quality_debt(df, baseline=df.copy(), ledger=None, verbose=False)
    _assert_unassessed(gate, "outlier_spikes", "dimension was not scored")


def test_unmeasured_dimensions_skip_the_ledger(tmp_path) -> None:
    """A run without a baseline neither records nor resets drift history."""
    ledger = str(tmp_path / "debt.sqlite")
    df = pd.DataFrame({"a": range(20), "b": list("xy" * 10)})
    baseline = df.drop(columns=["b"])  # "b" is added: drift 1.0

    _, g1 = fd.evaluate_quality_debt(df, baseline=baseline, debt_policy="warn_then_fail",
                                     ledger=ledger, verbose=False)
    assert _item(g1, "schema_drift").over
    _, g2 = fd.evaluate_quality_debt(df, debt_policy="warn_then_fail", ledger=ledger,
                                     verbose=False)
    drift2 = _item(g2, "schema_drift")
    assert not drift2.assessed and drift2.previous == pytest.approx(1.0)
    assert drift2 not in g2.warned
    _, g3 = fd.evaluate_quality_debt(df, baseline=baseline, debt_policy="warn_then_fail",
                                     ledger=ledger, verbose=False)
    drift3 = _item(g3, "schema_drift")
    # Compared against run 1, not a fabricated 0.0 from the baseline-less run 2.
    assert drift3.previous == pytest.approx(1.0)
    assert not drift3.worsening
    assert g3.status == "fail"  # drift over threshold again: repeated
    conn = sqlite3.connect(ledger)
    try:
        rows = conn.execute(
            "SELECT run_id, dimension, over_threshold FROM debt_items "
            "WHERE dimension IN ('schema_drift', 'category_churn') ORDER BY run_id, dimension"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [(1, "category_churn", 0), (1, "schema_drift", 1),
                    (3, "category_churn", 0), (3, "schema_drift", 1)]


def _parity_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "amount": [1.0, 2.0, None, 4.0, 100000.0, 2.0],
        "name": ["x", "x", "y", None, "z", "x"],
        "id": [1, 2, 3, 4, 5, 2],
        "email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com", "i@j.com", "a@b.com"],
        "num_text": ["1", "2", "3", "x", "5", "6"],
    })


def test_fully_measured_run_output_unchanged() -> None:
    """Profiling and PII scan succeed and a baseline is given: output as before #16."""
    baseline = _parity_frame().drop(columns=["id"]).assign(name=["x", "q", "y", None, "w", "x"])
    _, gate = fd.evaluate_quality_debt(_parity_frame(), baseline=baseline, debt_policy="warn",
                                       ledger=None, verbose=False, include_trust_score=False)

    def row(dimension, score, threshold, over, detail):
        return {"dimension": dimension, "score": score, "threshold": threshold,
                "over_threshold": over, "previous": None, "worsening": False, "detail": detail}

    assert [i.to_dict() for i in gate.items] == [
        row("missingness", 0.0556, 0.1, False, "2 missing cell(s) remain"),
        row("duplicates", 0.0, 0.1, False, "0 duplicate row(s) detected (0 removed)"),
        row("schema_drift", 0.25, 0.0, True, "1 added, 0 removed column(s)"),
        row("type_instability", 0.0, 0.1, False, "0 column(s) with unstable types"),
        row("outlier_spikes", 0.1667, 0.1, True, "1 outlier(s) flagged"),
        row("pii_risk", 0.1667, 0.0, True, "1 potential PII column(s)"),
        row("category_churn", 0.0625, 0.1, False, "category distribution churn vs baseline"),
        row("failed_repairs", 0.0, 0.0, False, "0 high-risk action(s)"),
        row("human_review_backlog", 0.2, 0.0, True, "2 item(s) awaiting review"),
    ]
    assert gate.status == "warn"
    assert gate.total_score == 0.9014
    assert gate.unassessed == []
    assert gate.summary() == "\n".join([
        "freshdata quality-debt gate: WARN (policy=warn, total debt 0.90)",
        "  ! schema_drift: 0.25 (threshold 0.00) — 1 added, 0 removed column(s)",
        "  ! human_review_backlog: 0.20 (threshold 0.00) — 2 item(s) awaiting review",
        "  ! outlier_spikes: 0.17 (threshold 0.10) — 1 outlier(s) flagged",
        "  ! pii_risk: 0.17 (threshold 0.00) — 1 potential PII column(s)",
    ])
