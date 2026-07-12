"""Layer 2: CleanReport → PeelView normalization."""

from __future__ import annotations

import pytest

from freshdata.render.normalize import normalize, normalize_clean_report, register_normalizer
from freshdata.render.view import PeelView
from freshdata.report import CleanReport


def make_report(**kw) -> CleanReport:
    rep = CleanReport(
        rows_before=10_000,
        rows_after=9_988,
        cols_before=14,
        cols_after=14,
        missing_before=1_204,
        missing_after=0,
        duration_seconds=0.412,
    )
    for key, value in kw.items():
        setattr(rep, key, value)
    return rep


class TestDispatch:
    def test_normalize_dispatches_on_render_kind(self):
        view = normalize(make_report())
        assert isinstance(view, PeelView)
        assert view.kind == "clean_report"

    def test_unknown_kind_raises(self):
        with pytest.raises(KeyError, match="no Peel normalizer"):
            normalize(object())

    def test_register_normalizer_extends_dispatch(self):
        class Custom:
            _render_kind = "custom_thing"

        view = PeelView("custom_thing", ("CLEAN",), "ok", (), (), None, ())
        register_normalizer("custom_thing", lambda obj: view)
        assert normalize(Custom()) is view


class TestStatus:
    def test_untouched_data_is_clean(self):
        assert normalize_clean_report(make_report()).status == ("CLEAN",)

    def test_changes_make_changed(self):
        rep = make_report()
        rep.add("missing", "filled 214 values", column="age", count=214)
        assert normalize_clean_report(rep).status == ("CHANGED",)

    def test_warnings_add_review(self):
        rep = make_report(warnings=["column 'income' is 38% missing"])
        assert normalize_clean_report(rep).status == ("CLEAN", "REVIEW")

    def test_unmaterialized_adds_partial_and_banner(self):
        rep = make_report(materialized=False)
        view = normalize_clean_report(rep)
        assert "PARTIAL" in view.status
        assert "kept in the engine" in view.banner
        assert "kept in the engine" in view.headline

    def test_info_only_attention_is_not_review(self):
        rep = make_report(
            fallback_events=[
                {"backend": "polars", "fallback_step": "impute", "fallback_reason": "dtype"}
            ]
        )
        view = normalize_clean_report(rep)
        assert "REVIEW" not in view.status
        assert view.attention[0].severity == "info"


class TestAttention:
    def test_ids_are_stable_per_source(self):
        rep = make_report(
            warnings=["w-one", "w-two"],
            recommendations=["r-one"],
        )
        view = normalize_clean_report(rep)
        assert [a.id for a in view.attention] == ["W1", "W2", "R1"]
        assert [a.severity for a in view.attention] == ["warning", "warning", "review"]

    def test_domain_errors_outrank_report_warnings(self):
        rep = make_report(
            warnings=["some warning"],
            domain="healthcare",
            domain_findings=[
                {
                    "status": "violated",
                    "severity": "error",
                    "column": "obx_units",
                    "message": "47 values are not valid UCUM units",
                    "count": 47,
                }
            ],
        )
        top = normalize_clean_report(rep).attention[0]
        assert top.id == "D1"
        assert top.severity == "error"
        assert top.domain == "policy"
        assert top.count == 47

    def test_passed_domain_findings_are_not_attention(self):
        rep = make_report(
            domain="energy",
            domain_findings=[{"status": "passed", "severity": "error", "rule": "range"}],
        )
        assert normalize_clean_report(rep).attention == ()

    def test_suggested_actions_become_review_items(self):
        rep = make_report()
        # non-semantic suggested action → generic "held for your review" text
        rep.add(
            "impute",
            "median fill needs confirmation on skewed column",
            column="income",
            count=380,
            status="suggested",
            confidence=0.84,
        )
        view = normalize_clean_report(rep)
        (item,) = view.attention
        assert item.id == "S1"
        assert item.severity == "review"
        assert item.subject == "income"
        assert "held for your review" in item.text
        assert item.detail["action"]["confidence"] == 0.84

    def test_contract_failures_become_policy_items(self):
        rep = make_report(
            contract_violations={
                "passed": False,
                "baseline_name": "orders",
                "baseline_version": 3,
                "findings": [
                    {"status": "failed", "column": "amount", "message": "dtype drift",
                     "check_id": "dtype"},
                    {"status": "passed", "column": "id", "check_id": "presence"},
                ],
            }
        )
        view = normalize_clean_report(rep)
        (item,) = view.attention
        assert item.id == "C1"
        assert item.severity == "error"
        assert item.domain == "policy"
        assert "contract 'orders'" in item.text

    def test_fallback_events_are_plain_language_info(self):
        rep = make_report(
            fallback_events=[
                {"backend": "duckdb", "fallback_step": "outliers", "fallback_reason": "quantile"}
            ]
        )
        (item,) = normalize_clean_report(rep).attention
        assert item.id == "F1"
        assert "continued without the optional engine step" in item.text

    def test_deterministic_across_calls(self):
        rep = make_report(warnings=["a", "b"], recommendations=["c"])
        first = normalize_clean_report(rep)
        second = normalize_clean_report(rep)
        assert first.attention == second.attention
        assert first.status == second.status
        assert first.headline == second.headline


class TestHeadlineAndMetrics:
    def test_headline_contents(self):
        rep = make_report()
        rep.add("missing", "filled", column="age", count=214)
        headline = normalize_clean_report(rep).headline
        assert "9,988 of 10,000 rows kept" in headline
        assert "14 columns" in headline
        assert "214 cells changed" in headline
        assert "0.4s" in headline

    def test_metrics_include_missing_and_protected(self):
        rep = make_report(duplicates_removed=12, columns_preserved=["notes"])
        labels = {m.label: m for m in normalize_clean_report(rep).metrics}
        assert labels["missing"].before == "1,204"
        assert labels["missing"].after == "0"
        assert labels["duplicates"].value == "-12"
        assert labels["protected"].value == "1"


class TestNextStep:
    def test_no_attention_no_next_step(self):
        assert normalize_clean_report(make_report()).next_step is None

    def test_info_only_no_next_step(self):
        rep = make_report(
            fallback_events=[{"backend": "polars", "fallback_step": "x", "fallback_reason": "y"}]
        )
        assert normalize_clean_report(rep).next_step is None

    def test_warning_points_to_explain(self):
        rep = make_report(warnings=["income is 38% missing"])
        assert "explain_clean" in normalize_clean_report(rep).next_step

    def test_suggested_actions_point_to_plan(self):
        rep = make_report()
        rep.add("semantic", "proposed fix", column="c", count=2, status="suggested")
        assert "suggest_plan" in normalize_clean_report(rep).next_step

    def test_domain_findings_win_over_warnings(self):
        rep = make_report(
            warnings=["w"],
            domain="finance",
            domain_findings=[
                {"status": "violated", "severity": "error", "column": "px", "message": "bad"}
            ],
        )
        assert "domain_findings" in normalize_clean_report(rep).next_step


class TestSections:
    def test_column_rows_aggregate_and_order(self):
        rep = make_report(columns_preserved=["notes"])
        rep.add("missing", "filled 214 value(s) with median", column="age", count=214)
        rep.add("outliers", "winsorized 37 value(s)", column="income", count=37)
        rep.add("missing", "filled 380 value(s) with median", column="income", count=380)
        view = normalize_clean_report(rep)
        columns = next(s for s in view.sections if s.key == "columns")
        rows = columns.rows()
        assert rows[0]["column"] == "income"
        assert rows[0]["changed"] == 417
        assert rows[0]["risk"] == "low"
        assert "filled missing values" in rows[0]["what"]
        assert "extreme values adjusted" in rows[0]["what"]
        protected = next(r for r in rows if r["column"] == "notes")
        assert protected["what"] == "protected — left untouched"

    def test_actions_section_uses_report_schema(self):
        rep = make_report()
        rep.add("missing", "filled", column="age", count=214, rationale="median safe")
        actions = next(s for s in normalize_clean_report(rep).sections if s.key == "actions")
        assert actions.count == 1
        (row,) = actions.rows()
        assert row["step"] == "missing"
        assert row["rationale"] == "median safe"

    def test_audit_section_skips_empty_and_keeps_hash(self):
        rep = make_report(decisions_hash="abc123", backend="polars")
        audit = next(s for s in normalize_clean_report(rep).sections if s.key == "audit")
        fields = {r["field"]: r["value"] for r in audit.rows()}
        assert fields["decisions_hash"] == "abc123"
        assert fields["backend"] == "polars"
        assert "streaming" not in fields
        assert "undo_log" not in fields  # never serialized (spec §13)

    def test_audit_ref_is_the_report(self):
        rep = make_report()
        assert normalize_clean_report(rep).audit_ref is rep
