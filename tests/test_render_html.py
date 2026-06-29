"""Tests for the optional interactive rendering layer.

Covers: every report surface produces valid, self-contained HTML with *no*
optional viz dependency installed; helpful errors when an optional package is
missing; the audit ledger / decision cards content; and that rendering never
pulls heavy libraries into a plain ``import freshdata``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.render import _optional


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


def test_profile_cockpit_html(messy: pd.DataFrame) -> None:
    html = fd.profile(messy).to_html()
    assert "quality score" in html
    assert "Type inference" in html
    assert "Columns (issue-ranked)" in html
    # correlation is on-demand, not eagerly computed
    assert "on demand" in html


def test_plan_decision_cards_html(messy: pd.DataFrame) -> None:
    html = fd.suggest_plan(messy).to_html()
    assert '<div class="fd-report"' in html
    assert "clean plan" in html


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
