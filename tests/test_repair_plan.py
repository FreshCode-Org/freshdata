"""Phase-2 RepairPlan / fd.apply_plan / undo / drift / decisions_hash."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.repairplan import compute_frame_signature

RULES = """CustomerID is unique.
Emails must be valid.
Phone numbers are Indian.
Allowed status values are active, inactive, pending.
Never modify revenue values."""


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cust_id": ["C001", "C002", "C003", "C004"],
            "email_addr": ["a@@b.com", "x @ y.com", "ok@ok.com", "junk"],
            "mobile": ["98765 43210", "09876543210", "+919876543210", "12345"],
            "monthly_revenue": ["1000", "2000", "3000", "4000"],
            "status": [" Active ", "INACTIVE", "actve", "pending"],
        }
    )


def _plan(df=None, **overrides):
    kwargs = {"context": RULES, "semantic_mode": "auto", "verbose": False}
    kwargs.update(overrides)
    return fd.suggest_plan(df if df is not None else _df(), **kwargs).repair_plan


# --------------------------------------------------------------------------- #
# plan construction
# --------------------------------------------------------------------------- #

def test_suggest_plan_exposes_repair_plan_with_ids():
    plan = _plan()
    assert isinstance(plan, fd.RepairPlan)
    ids = [a.id for a in plan.actions]
    assert ids == [f"a{i}" for i in range(1, len(ids) + 1)]
    assert all(a.kind and a.decision in ("auto", "suggest", "skip", "blocked")
               for a in plan.actions)


def test_plan_deterministic_across_runs():
    p1, p2 = _plan(), _plan()
    assert [a.to_dict() for a in p1.actions] == [a.to_dict() for a in p2.actions]
    assert p1.decisions_hash() == p2.decisions_hash()


def test_auto_actions_pre_approved_in_auto_mode():
    plan = _plan()
    auto = [a for a in plan.actions if a.decision == "auto"]
    assert auto and all(a.approval == "approved" for a in auto)


def test_review_mode_keeps_pending():
    plan = _plan(semantic_mode="review")
    suggested = [a for a in plan.actions if a.decision == "suggest"]
    assert suggested and all(a.approval == "pending" for a in suggested)


def test_no_plan_without_context_or_semantic():
    plan = fd.suggest_plan(_df(), verbose=False)
    assert plan.repair_plan is None


# --------------------------------------------------------------------------- #
# review workflow
# --------------------------------------------------------------------------- #

def test_approve_reject_override_and_selectors():
    plan = _plan(semantic_mode="review")
    typo = next(a for a in plan.actions if a.params.get("raw_value") == "actve")
    plan.approve(typo.id)
    assert typo.approval == "approved"
    plan.reject(typo.id, reason="keep the typo")
    assert typo.approval == "rejected"
    assert plan.rejection_reasons[typo.id] == "keep the typo"
    # column selector
    plan.approve("email_addr")
    email_actions = [a for a in plan.actions
                     if a.column == "email_addr" and a.decision != "blocked"]
    assert all(a.approval == "approved" for a in email_actions)
    # kind selector
    plan.reject("phone_format")
    assert all(a.approval == "rejected" for a in plan.actions
               if a.kind == "phone_format")
    # override
    plan.override(typo.id, {"proposed_value": "inactive"})
    assert typo.params["proposed_value"] == "inactive"
    assert typo.source.endswith("+override")
    with pytest.raises(KeyError):
        plan.approve("no_such_selector")


def test_approve_all_respects_max_risk_and_blocked():
    plan = _plan(semantic_mode="review")
    plan.approve_all(max_risk="low")
    for a in plan.actions:
        if a.decision == "blocked":
            assert a.approval != "approved"
        elif a.decision in ("auto", "suggest") and a.risk == "low":
            assert a.approval == "approved"
        elif a.risk in ("medium", "high"):
            assert a.approval != "approved"


def test_blocked_actions_cannot_be_approved():
    plan = _plan()
    blocked = [a for a in plan.actions if a.decision == "blocked"]
    if blocked:  # identifier/protected columns exist in this fixture
        plan.approve([a.id for a in blocked])
        assert all(a.approval == "pending" for a in blocked)


# --------------------------------------------------------------------------- #
# serialization
# --------------------------------------------------------------------------- #

def test_to_dict_from_dict_json_round_trip(tmp_path):
    plan = _plan()
    plan.approve_all(max_risk="low")
    path = tmp_path / "plan.json"
    plan.to_json(path)
    loaded = fd.RepairPlan.from_json(path)
    assert loaded.to_dict() == plan.to_dict()
    loaded2 = fd.RepairPlan.from_dict(json.loads(plan.to_json()))
    assert loaded2.decisions_hash() == plan.decisions_hash()
    # config round-trips well enough to apply
    assert loaded.config.semantic_mode == "auto"
    assert loaded.policy is not None


def test_summary_stable():
    plan = _plan()
    assert plan.summary() == plan.summary()
    assert plan.summary().startswith("freshdata repair plan")
    frame = plan.to_frame()
    assert set(frame.columns) >= {"id", "column", "kind", "decision", "approval"}


# --------------------------------------------------------------------------- #
# apply_plan
# --------------------------------------------------------------------------- #

def test_apply_plan_executes_approved_only():
    df = _df()
    plan = _plan(df)
    out, report = fd.apply_plan(df, plan)
    assert out["email_addr"].tolist() == ["a@b.com", "x@y.com", "ok@ok.com", "junk"]
    assert out["mobile"].tolist() == [
        "+919876543210", "+919876543210", "+919876543210", "12345"
    ]
    assert out["status"].tolist() == ["active", "inactive", "actve", "pending"]
    assert out["monthly_revenue"].equals(df["monthly_revenue"])
    assert df.equals(_df())  # input untouched
    applied = [a for a in report if a.status == "approved"]
    assert applied and all(a.step == "apply_plan" for a in applied)


def test_apply_plan_respects_rejections():
    df = _df()
    plan = _plan(df)
    plan.reject("email_format")
    out, report = fd.apply_plan(df, plan)
    assert out["email_addr"].tolist() == df["email_addr"].tolist()
    rejected = [a for a in report
                if a.metadata.get("issue_type") == "email_format"]
    assert rejected and all(a.count == 0 for a in rejected)


def test_apply_plan_does_not_re_decide():
    df = _df()
    plan = _plan(df)
    # Un-approve everything: an empty execution must change nothing, even
    # though re-profiling would have found repairs.
    for action in plan.actions:
        action.approval = "pending"
    out, _ = fd.apply_plan(df, plan)
    pd.testing.assert_frame_equal(out, df)


def test_apply_plan_accepts_clean_plan_wrapper():
    df = _df()
    clean_plan = fd.suggest_plan(df, context=RULES, semantic_mode="auto", verbose=False)
    out, report = fd.apply_plan(df, clean_plan)
    assert report.decisions_hash


def test_plan_drift_refusal_and_override():
    df = _df()
    plan = _plan(df)
    drifted = df.copy()
    drifted.loc[0, "status"] = "zzz"
    with pytest.raises(fd.PlanDriftError):
        fd.apply_plan(drifted, plan)
    out, _ = fd.apply_plan(drifted, plan, allow_drift=True)
    assert out["mobile"].tolist()[0] == "+919876543210"
    # column removed entirely -> stale actions recorded as skipped
    smaller = df.drop(columns=["status"])
    out2, report2 = fd.apply_plan(smaller, plan, allow_drift=True)
    assert any("not present" in a.description for a in report2)


def test_frame_signature_properties():
    df = _df()
    sig = compute_frame_signature(df)
    assert sig == compute_frame_signature(df.copy())
    assert sig != compute_frame_signature(df.drop(columns=["status"]))
    assert sig != compute_frame_signature(df.iloc[:2])
    changed = df.copy()
    changed.loc[0, "status"] = "x"
    assert sig != compute_frame_signature(changed)


def test_decisions_hash_stable_and_decision_sensitive():
    df = _df()
    plan = _plan(df)
    _, r1 = fd.apply_plan(df, plan)
    _, r2 = fd.apply_plan(df, plan)
    assert r1.decisions_hash == r2.decisions_hash
    assert r1.decisions_hash in json.dumps(r1.to_dict())
    plan.reject("phone_format")
    _, r3 = fd.apply_plan(df, plan)
    assert r3.decisions_hash != r1.decisions_hash


# --------------------------------------------------------------------------- #
# undo
# --------------------------------------------------------------------------- #

def test_undo_reverts_selected_actions():
    df = _df()
    plan = _plan(df)
    out, report = fd.apply_plan(df, plan, keep_undo=True)
    email_ids = [a.id for a in plan.actions
                 if a.kind == "email_format" and a.approval == "approved"]
    restored = report.revert(out, action_ids=email_ids)
    assert restored["email_addr"].tolist() == df["email_addr"].tolist()
    assert restored["mobile"].tolist() == out["mobile"].tolist()  # untouched
    full = report.revert(out)
    assert full["mobile"].tolist() == df["mobile"].tolist()


def test_undo_requires_keep_undo():
    df = _df()
    plan = _plan(df)
    out, report = fd.apply_plan(df, plan)
    with pytest.raises(ValueError, match="keep_undo"):
        report.revert(out)


def test_undo_cap_marks_actions_irreversible():
    df = _df()
    plan = _plan(df)
    out, report = fd.apply_plan(df, plan, keep_undo=True, undo_cell_limit=1)
    executed = [a for a in plan.actions if a.approval == "approved"
                and a.params.get("proposed_value") is not None]
    assert any(a.reversible is False for a in executed)
    recorded = {e["action_id"] for e in report.undo_log["entries"]}
    assert all(a.id not in recorded for a in executed if a.reversible is False)


def test_unknown_action_id_raises():
    df = _df()
    plan = _plan(df)
    out, report = fd.apply_plan(df, plan, keep_undo=True)
    with pytest.raises(KeyError):
        report.revert(out, action_ids=["nope"])


# --------------------------------------------------------------------------- #
# backward compatibility
# --------------------------------------------------------------------------- #

def test_plain_clean_unchanged():
    df = _df()
    out_plain = fd.clean(df, verbose=False)
    out_none = fd.clean(df, context=None, verbose=False)
    pd.testing.assert_frame_equal(out_plain, out_none)
    report = fd.clean(df, return_report=True, verbose=False)[1]
    assert report.decisions_hash is None
    assert "decisions_hash" not in report.to_dict()
