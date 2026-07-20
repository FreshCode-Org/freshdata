"""Production-audit regression tests for the four confirmed P1 defects.

Each test fails on the pre-audit defaults and pins the remediated behavior:

1. P1-1 — duplicate removal was destructive by default.
2. P1-2 — ``dayfirst="auto"`` inferred a column-wide day/month order from a
   single disambiguating value, silently flipping ambiguous dates.
3. P1-6 — ``outlier_action="auto"`` winsorized heavy tails under
   ``strategy="aggressive"``.
4. P1-7 — a column-NAME regex decided whether outliers were preserved.
"""

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.execution._native_steps import log_widened_bounds
from freshdata.steps.duplicates import DuplicateRatioError
from freshdata.steps.outliers import detection_bounds

QUIET = {"verbose": False}


def lognormal_income(n=3000, seed=0):
    return np.random.default_rng(seed).lognormal(10, 1.8, n)


def outlier_actions(report):
    return [a for a in report if a.step == "outliers"]


# -- Defect 1: duplicates are detected, never removed by default --------------


def test_audit_repro_duplicates_not_removed_by_default():
    # Audit P1-1 repro: 60 rows, 58 exact duplicates — the old default kept 2.
    df = pd.DataFrame({"flag": [0, 1] * 30, "const": [1] * 60})
    out, report = fd.clean(df, return_report=True, **QUIET)
    assert len(out) == 60  # nothing removed
    assert report.duplicates_removed == 0
    [action] = [a for a in report if a.step == "drop_duplicates"]
    assert "none removed" in action.description
    assert any("NOT removed" in w and "drop_duplicates=True" in w
               for w in report.warnings)


def test_no_strategy_reenables_duplicate_removal():
    df = pd.DataFrame({"flag": [0, 1] * 30, "const": [1] * 60})
    for strategy in ("conservative", "balanced", "aggressive"):
        out = fd.clean(df, strategy=strategy, **QUIET)
        assert len(out) == 60, strategy


def test_duplicate_removal_is_opt_in():
    df = pd.DataFrame({"flag": [0, 1] * 30, "const": [1] * 60})
    out, report = fd.clean(df, return_report=True, drop_duplicates=True, **QUIET)
    assert len(out) == 2
    assert report.duplicates_removed == 58


def test_low_ratio_duplicates_reported_without_warning():
    # Below duplicate_threshold: the report notes the detection, no warning.
    df = pd.DataFrame({"a": list(range(30)) + [0], "b": ["x"] * 30 + ["x"]})
    out, report = fd.clean(df, return_report=True, **QUIET)
    assert len(out) == 31
    [action] = [a for a in report if a.step == "drop_duplicates"]
    assert "none removed" in action.description
    assert not any("duplicate ratio" in w for w in report.warnings)


def test_duplicate_ratio_action_error_escalates():
    df = pd.DataFrame({"flag": [0, 1] * 30, "const": [1] * 60})
    with pytest.raises(DuplicateRatioError, match="duplicate ratio"):
        fd.clean(df, duplicate_ratio_action="error", **QUIET)
    # ...whether or not removal is enabled.
    with pytest.raises(DuplicateRatioError):
        fd.clean(df, drop_duplicates=True, duplicate_ratio_action="error", **QUIET)


def test_duplicate_ratio_action_validated():
    with pytest.raises(ValueError, match="duplicate_ratio_action"):
        fd.clean(pd.DataFrame({"a": [1]}), duplicate_ratio_action="explode", **QUIET)


# -- Defect 2: dayfirst="auto" never infers a column-wide order ---------------


def test_audit_repro_disambiguating_value_does_not_flip_column():
    # Audit P1-2 repro: one "13/01/2023" used to flip the WHOLE column to
    # dayfirst, silently reading 01/02/2023 as Feb 1.
    values = ["01/02/2023", "03/04/2023", "05/06/2023", "13/01/2023"]
    out, report = fd.clean(pd.DataFrame({"d": values}), return_report=True, **QUIET)
    assert str(out["d"].dtype).startswith("datetime64")
    # The unambiguous value parses on its own merits...
    assert pd.Timestamp("2023-01-13") in list(out["d"].dropna())
    # ...and no ambiguous value was interpreted: all three are quarantined.
    coerced = set(map(str, report.coerced_cells.get("d", {}).values()))
    assert coerced == {"01/02/2023", "03/04/2023", "05/06/2023"}
    # The audit trail tells the user how to resolve the ambiguity.
    [hint] = [a for a in report if "day/month-ambiguous" in a.description]
    assert "dayfirst" in hint.rationale


def test_all_ambiguous_column_is_preserved_as_text():
    values = ["01/02/2023", "03/04/2023", "05/06/2023"]
    out = fd.clean(pd.DataFrame({"d": values}), **QUIET)
    assert out["d"].tolist() == values  # untouched, still strings


def test_mixed_us_eu_iso_invalid_inputs():
    iso = [f"2023-03-{d:02d}" for d in range(1, 19)]
    values = [*iso, "13/01/2023", "01/20/2023", "05/06/2023", "not a date"]
    out, report = fd.clean(pd.DataFrame({"d": values}), return_report=True, **QUIET)
    assert str(out["d"].dtype).startswith("datetime64")
    parsed = list(out["d"].dropna())
    assert pd.Timestamp("2023-01-13") in parsed  # EU-unambiguous
    assert pd.Timestamp("2023-01-20") in parsed  # US-unambiguous
    coerced = set(map(str, report.coerced_cells.get("d", {}).values()))
    assert "05/06/2023" in coerced   # ambiguous: quarantined
    assert "not a date" in coerced   # invalid: existing coerce path


def test_explicit_dayfirst_parses_whole_column_both_ways():
    values = ["01/02/2023", "03/04/2023", "13/01/2023"]
    eu = fd.clean(pd.DataFrame({"d": values}), dayfirst=True, **QUIET)
    assert eu["d"].tolist() == [pd.Timestamp("2023-02-01"),
                                pd.Timestamp("2023-04-03"),
                                pd.Timestamp("2023-01-13")]
    us = fd.clean(pd.DataFrame({"d": values}), dayfirst=False, **QUIET)
    assert us["d"].tolist()[:2] == [pd.Timestamp("2023-01-02"),
                                    pd.Timestamp("2023-03-04")]
    assert us["d"].tolist()[2] == pd.Timestamp("2023-01-13")  # only valid reading


def test_day_equals_month_is_not_ambiguous():
    values = ["01/01/2023"] * 5 + ["2023-06-15"]
    out, report = fd.clean(pd.DataFrame({"d": values}), return_report=True, **QUIET)
    assert str(out["d"].dtype).startswith("datetime64")
    assert not out["d"].isna().any()
    assert not report.coerced_cells.get("d")


# -- Defect 3: "auto" flags under every strategy; capping is explicit ---------


def test_audit_repro_aggressive_auto_flags_heavy_tail():
    # Audit P1-6 repro: lognormal income, max in the millions — aggressive
    # used to silently winsorize it down to ~170k.
    income = lognormal_income()
    df = pd.DataFrame({"income": income})
    out, report = fd.clean(df, return_report=True, strategy="aggressive", **QUIET)
    assert out["income"].max() == pytest.approx(income.max())  # untouched
    assert "income_outlier" in out.columns
    [action] = outlier_actions(report)
    assert "flagged" in action.description  # full audit trail preserved
    assert action.column == "income"


def test_balanced_auto_still_flags():
    df = pd.DataFrame({"income": lognormal_income()})
    out, report = fd.clean(df, return_report=True, **QUIET)
    assert "income_outlier" in out.columns
    [action] = outlier_actions(report)
    assert "flagged" in action.description


def test_explicit_cap_is_skew_aware_on_heavy_tails():
    income = lognormal_income()
    df = pd.DataFrame({"income": income})
    out, report = fd.clean(df, return_report=True, outlier_action="cap", **QUIET)
    raw_hi = detection_bounds(pd.Series(income), "iqr", 1.5)[1]
    # Capping still tames the absurd top...
    assert out["income"].max() < income.max()
    # ...but the fences were computed in log space, far above the raw fence
    # that used to flatten the legitimate tail.
    assert out["income"].max() > 2 * raw_hi
    [action] = outlier_actions(report)
    assert "capped" in action.description


def test_explicit_cap_keeps_raw_fences_for_symmetric_data():
    rng = np.random.default_rng(0)
    values = rng.normal(100, 10, 500)
    values[-1] = 10_000.0
    df = pd.DataFrame({"v": values})
    out = fd.clean(df, outlier_action="cap", **QUIET)
    assert out["v"].max() < 200.0  # near-normal data: current fences kept


# -- Defect 4: column names must not decide outlier treatment ----------------


def test_audit_repro_column_name_parity():
    # Audit P1-7 repro: identical series, only the name differs.
    income = lognormal_income()
    results = {}
    for name in ("amount", "xyz"):
        out = fd.clean(pd.DataFrame({name: income.copy()}),
                       strategy="aggressive", **QUIET)
        results[name] = (float(out[name].max()), f"{name}_outlier" in out.columns)
    assert results["amount"] == results["xyz"]


def test_column_name_parity_under_explicit_cap():
    income = lognormal_income()
    maxima = {}
    for name in ("amount", "xyz"):
        out = fd.clean(pd.DataFrame({name: income.copy()}),
                       outlier_action="cap", **QUIET)
        maxima[name] = float(out[name].max())
    assert maxima["amount"] == pytest.approx(maxima["xyz"])
    assert maxima["xyz"] < income.max()  # both actually capped


def test_domain_sensitive_names_opt_in_restores_preservation():
    income = lognormal_income()
    df = pd.DataFrame({"amount": income.copy(), "xyz": income.copy()})
    out, report = fd.clean(df, return_report=True, outlier_action="cap",
                           domain_sensitive_names=True, **QUIET)
    assert out["amount"].max() == pytest.approx(income.max())  # preserved
    assert out["xyz"].max() < income.max()                     # still capped
    preserved = [a for a in outlier_actions(report) if "preserved" in a.description]
    assert [a.column for a in preserved] == ["amount"]


def test_domain_sensitive_names_validated():
    with pytest.raises(TypeError, match="domain_sensitive_names"):
        fd.clean(pd.DataFrame({"a": [1]}), domain_sensitive_names="yes", **QUIET)


# -- skew-aware fence internals (shared pandas/native contract) -------------

def test_log_widened_bounds_guards():
    """Each guard clause keeps the raw fences; only the full skewed case widens."""
    kw = {"skew": 5.0, "minimum": 1.0, "log_q1": 1.0, "log_q3": 3.0, "factor": 1.5}
    assert log_widened_bounds(0.0, 10.0, n_non_null=5, **kw) == (0.0, 10.0)
    assert log_widened_bounds(0.0, 10.0, n_non_null=100,
                              **{**kw, "skew": None}) == (0.0, 10.0)
    assert log_widened_bounds(0.0, 10.0, n_non_null=100,
                              **{**kw, "log_q1": float("nan")}) == (0.0, 10.0)
    assert log_widened_bounds(0.0, 10.0, n_non_null=100,
                              **{**kw, "skew": 0.5}) == (0.0, 10.0)
    assert log_widened_bounds(0.0, 10.0, n_non_null=100,
                              **{**kw, "minimum": -1.0}) == (0.0, 10.0)
    assert log_widened_bounds(0.0, 10.0, n_non_null=100,
                              **{**kw, "log_q3": 1.0}) == (0.0, 10.0)  # zero spread
    lo, hi = log_widened_bounds(0.0, 10.0, n_non_null=100, **kw)
    assert lo == 0.0 and hi > 10.0  # widen-only


def test_explicit_cap_preserves_when_widened_fences_clear_the_tail():
    """A lawful skewed tail that fits entirely inside the log-space fences is
    preserved even under an explicit cap request (the audited no-op path):
    exp(normal clipped to ±3sigma) is skewed enough for log fences (skew ~2.9)
    whose span (±3.8 in log space) covers every value, while raw IQR still
    flags the top decile."""
    rng = np.random.default_rng(11)
    vals = np.exp(np.clip(rng.normal(0.0, 1.5, 500), -3.0, 3.0))
    out, report = fd.clean(pd.DataFrame({"v": vals.copy()}),
                           outlier_action="cap", return_report=True, **QUIET)
    preserved = [a for a in outlier_actions(report)
                 if "widened fences" in a.rationale]
    assert len(preserved) == 1  # raw detection fired, log fences cleared it
    assert out["v"].max() == pytest.approx(vals.max())  # nothing rewritten
