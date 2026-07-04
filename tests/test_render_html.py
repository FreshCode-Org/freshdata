"""Tests for the optional interactive rendering layer.

Covers: every report surface produces valid, self-contained HTML with *no*
optional viz dependency installed; helpful errors when an optional package is
missing; the audit ledger / decision cards content; and that rendering never
pulls heavy libraries into a plain ``import freshdata``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

import freshdata as fd
from freshdata.render import _optional
from freshdata.render.renderers import (
    render_clean_plan,
    render_compare_clean,
    render_compare_plans,
    render_infer_roles,
)
from freshdata.report import CleanReport


@pytest.fixture
def messy() -> pd.DataFrame:
    return pd.DataFrame({
        "amount": [1, 2, None, 4, 100000],
        "name": ["x", "x", "y", None, "z"],
        "id": [1, 2, 3, 4, 5],
    })


def _is_valid_fragment(html: str) -> bool:
    return (
        isinstance(html, str)
        and '<div class="fd-report"' in html
        and html.count("<div") == html.count("</div>") + html.count("<div")  # sanity
        and "<style>" in html
    )


def test_clean_report_html(messy: pd.DataFrame) -> None:
    _, report = fd.clean(messy, return_report=True)
    html = report.to_html()
    assert '<div class="fd-report"' in html
    assert "Action timeline" in html
    assert "Audit ledger" in html
    # ledger export links are embedded (no server needed)
    assert "clean_ledger.json" in html and "clean_ledger.csv" in html
    assert report._repr_html_() == html


def test_clean_report_renderer_badges_warnings_and_native_handle() -> None:
    report = CleanReport(
        rows_before=10,
        rows_after=0,
        cols_before=3,
        cols_after=0,
        missing_before=4,
        missing_after=0,
        materialized=False,
        backend="duckdb",
        warnings=["semantic value needs review"],
        recommendations=["inspect customer_id"],
    )
    report.add(
        "semantic",
        "accepted approved repair",
        column="segment",
        count=2,
        reversible=True,
        memory_influenced=True,
        human_review=True,
    )
    report.add(
        "drop_rows",
        "removed duplicate row",
        count=1,
        risk="high",
        reversible=False,
    )

    html = report.to_html()

    assert "reversible" in html
    assert "irreversible" in html
    assert "memory" in html
    assert "review" in html
    assert "Warnings" in html
    assert "Needs review" in html
    assert "not materialized" in html


def test_profile_cockpit_html(messy: pd.DataFrame) -> None:
    html = fd.profile(messy).to_html()
    assert "quality score" in html
    assert "Type inference" in html
    assert "Columns (issue-ranked)" in html
    # correlation is on-demand, not eagerly computed
    assert "on demand" in html


def test_profile_cockpit_cardinality_and_outlier_warnings() -> None:
    df = pd.DataFrame(
        {
            "id": [f"id-{i}" for i in range(30)],
            "constant": ["x"] * 30,
            "amount": [1, 2, 3, 4, 5, 100000] * 5,
        }
    )

    html = fd.profile(df).to_html()

    assert "Cardinality warnings" in html
    assert "id (all-unique)" in html
    assert "constant (constant)" in html
    assert "Outlier warnings" in html


def test_plan_decision_cards_html(messy: pd.DataFrame) -> None:
    html = fd.suggest_plan(messy).to_html()
    assert '<div class="fd-report"' in html
    assert "clean plan" in html


def test_clean_plan_renderer_decision_branches() -> None:
    rejected = SimpleNamespace(
        eligible=False,
        rejection_reason="too sparse",
        model_id="median",
        confidence=0.42,
        rationale="numeric fallback",
    )
    alt = SimpleNamespace(model_id="mean")
    plan = SimpleNamespace(
        column_plans={
            "amount": SimpleNamespace(
                missing=rejected,
                missing_alternatives=[alt],
                outlier_action="winsorize",
                n_outliers=2,
            ),
            "clean": SimpleNamespace(
                missing=None,
                missing_alternatives=[],
                outlier_action=None,
                n_outliers=0,
            ),
        },
        config=SimpleNamespace(strategy="balanced"),
        to_dict=lambda: {"strategy": "balanced"},
    )

    html = render_clean_plan(plan)

    assert "needs review" in html
    assert "alternatives: mean" in html
    assert "outliers" in html
    assert "no engine action" in html


def test_explain_diff_explorer_html(messy: pd.DataFrame) -> None:
    rep = fd.explain_clean(messy)
    html = rep.to_html()
    assert "Safe changes" in html and "Risky changes" in html
    # ExplainReport gained to_frame()
    frame = rep.to_frame()
    assert list(frame.columns) == ["column", "before_dtype", "after_dtype", "changed_cells"]


def test_dataframe_surfaces_render_and_stay_dataframes(messy: pd.DataFrame) -> None:
    for result in (fd.compare_plans(messy), fd.compare_clean(messy), fd.infer_roles(messy)):
        assert isinstance(result, pd.DataFrame)  # backward compatible
        html = result._repr_html_()
        assert html and '<div class="fd-report"' in html


def test_dataframe_renderer_fallback_branches() -> None:
    assert "fd-cmp-plans" in render_compare_plans(pd.DataFrame({"x": [1]}))
    assert "clean comparison" in render_compare_clean(pd.DataFrame({"x": [1]}))

    roles = pd.DataFrame(
        {
            "column": ["amount"],
            "role": ["numeric"],
            "confidence": ["not-a-float"],
            "evidence": ["range-like"],
        }
    )
    roles_html = render_infer_roles(roles)

    assert "range-like" in roles_html


def test_all_surfaces_emit_balanced_div_tags(messy: pd.DataFrame) -> None:
    _, report = fd.clean(messy, return_report=True)
    for obj in (report, fd.profile(messy), fd.suggest_plan(messy), fd.explain_clean(messy)):
        html = obj.to_html()
        assert html.count("<div") == html.count("</div>"), type(obj).__name__


def test_missing_optional_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "plotly":
            raise ImportError("no plotly")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(_optional.importlib, "import_module", fake_import_module)
    with pytest.raises(ImportError) as exc:
        _optional.require("plotly", feature="charts")
    assert "freshdata[viz]" in str(exc.value)
    assert "charts" in str(exc.value)


def test_has_probe_is_safe() -> None:
    assert _optional.has("pandas") is True
    assert _optional.has("nonexistent_pkg_xyz") is False


def test_show_writes_file_outside_notebook(messy: pd.DataFrame, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _, report = fd.clean(messy, return_report=True)
    path = report.show()  # no IPython → writes a standalone .html and returns the path
    assert path and path.endswith(".html")
    with open(path, encoding="utf-8") as fh:
        body = fh.read()
    assert "<!doctype html>" in body and "fd-report" in body


def test_reportframe_show_and_repr(messy: pd.DataFrame, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    roles = fd.infer_roles(messy)
    path = roles.show()
    assert path.endswith(".html")
    # Derived frames fall back to a plain DataFrame (no rich repr leakage).
    derived = roles[["column", "role"]]
    assert type(derived) is pd.DataFrame


def test_compare_clean_outcome_dashboard(messy: pd.DataFrame) -> None:
    html = fd.compare_clean(messy)._repr_html_()
    assert "clean comparison" in html


def test_html_primitive_helpers() -> None:
    from freshdata.render import html as H

    assert "fd-del-pos" in H.delta(5) or "fd-del-neg" in H.delta(5)
    assert "<li>" in H.kv_list({"a": 1})
    assert "<details open>" in H.collapsible("s", "b", open_=True)
    assert "data:application/json" in H.json_download("x.json", {"a": 1})
    assert H.esc(None) == ""
    assert "width:100%" in H.bar(2.0)  # clamps above 1.0
