"""Regression tests for report-rendering fixes (#329, #336, #337, #339)."""

from __future__ import annotations

import re
import shutil
import subprocess

import pandas as pd
import pytest

import freshdata as fd
from freshdata.render import html as H
from freshdata.render._vocabulary import changed_values, plain_step
from freshdata.render.normalize import normalize_clean_report
from freshdata.report import Action, CleanReport

_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.S)


# -- #329: insight issue ids are unique ---------------------------------------


def test_insight_issue_ids_unique_when_column_slugs_collide() -> None:
    df = pd.DataFrame({"a b": [1, None] * 20, "a_b": [1, None] * 20, "A-B": ["x", None] * 20})
    rep = fd.insight_report(df)
    pairs = [(i["id"], i["column"]) for i in rep.issues]
    assert pairs == [
        ("issue.a_b.missing", "a b"),
        ("issue.a_b.missing.2", "a_b"),
        ("issue.a_b.missing.3", "A-B"),
    ]
    action_ids = [i["recommended_action_id"] for i in rep.issues]
    assert action_ids == [
        "action.a_b.missing",
        "action.a_b.missing.2",
        "action.a_b.missing.3",
    ]


def test_insight_ids_unchanged_for_non_colliding_columns() -> None:
    df = pd.DataFrame({"age": [1, None] * 20, "City Name": ["x", None] * 20})
    ids = sorted(i["id"] for i in fd.insight_report(df).issues)
    assert ids == ["issue.age.missing", "issue.city_name.missing"]


def test_insight_recommended_action_ids_point_at_real_actions() -> None:
    df = pd.DataFrame({"a b": [1.0, None] * 20, "a_b": [2.0, None] * 20, "keep": range(40)})
    cleaned, report = fd.clean(df, return_report=True, verbose=False)
    rep = fd.insight_report(df, clean_report=report, cleaned_df=cleaned)
    issue_ids = [i["id"] for i in rep.issues]
    assert len(issue_ids) == len(set(issue_ids))
    action_ids = {a["id"] for a in rep.actions}
    by_column = {a["column"]: a["id"] for a in rep.actions if a["step"] == "missing"}
    for issue in rep.issues:
        if issue["column"] in by_column:
            assert issue["recommended_action_id"] == by_column[issue["column"]]
            assert issue["recommended_action_id"] in action_ids


# -- #336: HTML filter script is valid and id-free -----------------------------


def _clean_html() -> str:
    df = pd.DataFrame({"name": [" a", "b ", None, "b "], "v": [1.0, None, 3.0, 3.0]})
    _, rep = fd.clean(df, return_report=True, verbose=False)
    return rep.to_html()


def test_filterable_table_has_no_inline_handlers_or_global_functions() -> None:
    out = H.filterable_table("fd-ledger", ["a", "b"], [["1", "2"]], filters={"a": 0, "b": 1})
    assert "oninput=" not in out
    assert "fdFilter_" not in out
    assert "document.currentScript" in out
    scripts = _SCRIPT_RE.findall(out)
    assert len(scripts) == 1
    assert "fd-ledger" not in scripts[0]
    # controls, then table, then the script: the DOM-sibling lookup relies on it.
    assert out.index('class="fd-controls"') < out.index("<table") < out.index("<script>")
    assert out.endswith("</table>" + f"<script>{scripts[0]}</script>")


def test_filterable_table_output_is_deterministic() -> None:
    args = ("fd-x", ["h"], [["v"]])
    assert H.filterable_table(*args, filters={"h": 0}) == H.filterable_table(
        *args, filters={"h": 0}
    )
    assert "<script>" not in H.filterable_table(*args)


def test_clean_report_html_filters_have_no_broken_markup() -> None:
    out = _clean_html()
    assert "oninput=" not in out
    assert "fdFilter_" not in out
    assert _SCRIPT_RE.findall(out)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_generated_scripts_parse_as_javascript(tmp_path) -> None:
    df = pd.DataFrame({"name": [" a", "b ", None, "b "], "v": [1.0, None, 3.0, 3.0]})
    pages = [_clean_html(), fd.profile(df).to_html()]
    scripts = [s for page in pages for s in _SCRIPT_RE.findall(page)]
    assert scripts
    node = shutil.which("node")
    assert node is not None
    for n, body in enumerate(scripts):
        path = tmp_path / f"script_{n}.js"
        path.write_text(body, encoding="utf-8")
        proc = subprocess.run(
            [node, "--check", str(path)], capture_output=True, text=True, check=False
        )
        assert proc.returncode == 0, proc.stderr


# -- #337: preserved values are not reported as changes ------------------------


def test_stakeholder_summary_ignores_informational_actions() -> None:
    df = pd.DataFrame(
        {"a": [1.0, None, 3.0, 4.0], "b": [5.0, None, 7.0, 8.0], "c": ["x", "y", "z", "w"]}
    )
    _, rep = fd.clean(df, return_report=True, verbose=False)
    assert rep.actions and all(a.count == 0 for a in rep.actions if a.column)
    s = fd.stakeholder_summary(rep)
    assert not any("changed meaningfully" in line for line in s.what_changed)


def test_stakeholder_summary_counts_only_applied_changes() -> None:
    rep = CleanReport(rows_before=4, rows_after=4, cols_before=4, cols_after=4)
    rep.add("missing", "filled 2 value(s) with median", column="a", count=2)
    rep.add("missing", "preserved 1 missing value(s)", column="b", count=0)
    rep.add("semantic", "proposed fix", column="c", count=3, status="suggested")
    rep.add("semantic", "skipped fix", column="d", count=3, status="skipped")
    s = fd.stakeholder_summary(rep, audience="technical")
    assert "1 column(s) changed meaningfully." in s.what_changed
    assert "Steps applied: missing." in s.what_changed


def test_plain_step_keeps_description_when_nothing_changed() -> None:
    preserved = Action("missing", "a", "preserved 1 missing value(s)", count=0)
    filled = Action("missing", "a", "filled 2 value(s)", count=2)
    suggested = Action("missing", "a", "would fill 2 value(s)", count=2, status="suggested")
    assert plain_step(preserved) == "preserved 1 missing value(s)"
    assert plain_step(filled) == "filled missing values (2)"
    assert plain_step(suggested) == "would fill 2 value(s)"
    assert not changed_values(preserved)
    assert changed_values(filled)
    assert not changed_values(suggested)


def test_peel_column_rows_do_not_claim_fills_for_preserved_gaps() -> None:
    df = pd.DataFrame({"name": [" a", "b ", None, "b "], "v": [1.0, None, 3.0, 3.0]})
    _, rep = fd.clean(df, return_report=True, verbose=False)
    assert any(a.step == "missing" and a.count == 0 for a in rep.actions)
    columns = next(s for s in normalize_clean_report(rep).sections if s.key == "columns")
    for row in columns.rows():
        assert "filled missing values" not in row["what"]


# -- #339: no completeness claim when no cells remain --------------------------


def test_stakeholder_summary_all_columns_dropped() -> None:
    df = pd.DataFrame({"a": [None, None, None], "b": [None, None, None]})
    out, rep = fd.clean(df, return_report=True, verbose=False)
    assert out.shape == (3, 0)
    s = fd.stakeholder_summary(rep)
    assert s.headline.startswith("Cleaning removed every column")
    assert "100.0%" not in s.headline
    assert not any("completeness" in line for line in s.what_changed)
    assert any("unusable column(s) were removed" in line for line in s.what_changed)
    assert s.metrics["completeness"] == "n/a"
    # every export still renders
    assert "n/a" in s.to_html()
    assert s.headline in s.to_markdown()
    assert s.to_dict()["metrics"]["completeness"] == "n/a"


def test_stakeholder_summary_all_records_removed() -> None:
    rep = CleanReport(
        rows_before=2, rows_after=0, cols_before=2, cols_after=2, missing_before=1, missing_after=0
    )
    s = fd.stakeholder_summary(rep)
    assert s.headline.startswith("Cleaning removed every record")
    assert s.metrics["completeness"] == "n/a"
    assert not any("completeness" in line for line in s.what_changed)


def test_stakeholder_summary_unmaterialized_report_makes_no_claim() -> None:
    rep = CleanReport(rows_before=10, cols_before=2, missing_before=3)
    rep.materialized = False
    s = fd.stakeholder_summary(rep)
    assert "%" not in s.headline
    assert s.metrics["completeness"] == "n/a"
    assert not any("completeness" in line for line in s.what_changed)


def test_stakeholder_summary_normal_completeness_unchanged() -> None:
    rep = CleanReport(
        rows_before=2, rows_after=2, cols_before=2, cols_after=2, missing_before=2, missing_after=1
    )
    s = fd.stakeholder_summary(rep)
    assert s.headline.startswith("Cleaning kept 75.0% of fields complete across 2 record(s)")
    assert s.metrics["completeness"] == "75.0%"
    assert "Overall data completeness rose from 50.0% to 75.0%." in s.what_changed
