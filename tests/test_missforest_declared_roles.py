"""MissForest honors resolved declarations after column-name normalization."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.imputation.missforest import MissForestImputer

ISOLATE = {
    "strategy": "conservative", "fix_dtypes": False, "verbose": False,
    "drop_empty_rows": False, "drop_empty_columns": False,
    "missforest_add_indicators": False,
}


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "Observed Hours": [float(i % 12) for i in range(80)],
        "External Measure": [10 + 2 * (i % 12) + 0.5 * (i % 5) for i in range(80)],
    })
    frame.loc[frame.index[::11], "External Measure"] = float("nan")
    return frame


@pytest.mark.parametrize("role", ["target", "identifier", "both"])
@pytest.mark.parametrize("choice", ["global", "per-column"])
@pytest.mark.parametrize("declaration", ["External Measure", "external_measure"])
def test_normalized_declared_roles_are_preserved_without_loading_models(
    role, choice, declaration, monkeypatch
):
    def forbidden():
        raise AssertionError("a protected column must not train a model")

    monkeypatch.setattr(MissForestImputer, "_sklearn_forests", staticmethod(forbidden))
    frame = _frame()
    original = frame.copy(deep=True)
    options = {}
    if role in ("target", "both"):
        options["target_column"] = declaration
    if role in ("identifier", "both"):
        options["id_columns"] = (declaration,)
    if choice == "global":
        options["impute"] = "missforest"
    else:
        options["impute_strategy"] = {"external_measure": "missforest"}

    out, report = fd.clean(frame, return_report=True, **ISOLATE, **options)

    pd.testing.assert_series_equal(
        out["external_measure"], original["External Measure"].rename("external_measure")
    )
    pd.testing.assert_frame_equal(frame, original)
    (action,) = [a for a in report.actions if a.step == "impute"]
    assert action.count == 0
    assert action.metadata["imputed_count"] == 0
    assert action.metadata["selected_model_type"] is None
    expected = "identifier" if role == "identifier" else "target"
    assert action.metadata["fallback_reason"] == f"{expected} column is never imputed"


@pytest.mark.parametrize("role", ["target", "identifier"])
@pytest.mark.parametrize("choice", ["global", "per-column"])
def test_original_declarations_still_work_when_renaming_is_disabled(role, choice):
    options = {"target_column": "External Measure"} if role == "target" else {
        "id_columns": ("External Measure",)
    }
    options.update({"impute": "missforest"} if choice == "global" else {
        "impute_strategy": {"External Measure": "missforest"}
    })
    frame = _frame()
    out = fd.clean(frame, column_names=False, **ISOLATE, **options)
    pd.testing.assert_frame_equal(out, frame)


def test_eligible_feature_still_trains_while_the_declared_target_is_preserved():
    pytest.importorskip("sklearn")
    frame = _frame()
    frame["Other Measure"] = [20 + 3 * (i % 12) + 0.5 * (i % 5) for i in range(80)]
    frame.loc[frame.index[::13], "Other Measure"] = float("nan")
    out, report = fd.clean(
        frame, impute="missforest", target_column="External Measure",
        missforest_n_estimators=10, missforest_max_iter=2, return_report=True, **ISOLATE,
    )
    pd.testing.assert_series_equal(
        out["external_measure"], frame["External Measure"].rename("external_measure")
    )
    assert out["other_measure"].notna().all()
    (trained,) = [a for a in report.actions if a.column == "other_measure" and a.step == "impute"]
    assert trained.metadata["selected_model_type"] == "regressor"
    assert trained.count == 7
