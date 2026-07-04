import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import EnterpriseConfig, SemanticValidatorConfig, clean_enterprise
from freshdata.insight import (
    _action_impact,
    _frame_records,
    _frame_stats,
    _json_scalar,
    _lineage_from_tracker,
    to_pandas_safe,
)
from freshdata.render.renderers import render


def _crm_frame() -> pd.DataFrame:
    n = 40
    return pd.DataFrame(
        {
            "customer_id": list(range(n)),
            "age": [None] * 24 + [31, 34, 37, 41] * 4,
            "salary": ["$1,200.50", "-", "$2,000", "$999,999"] * 10,
            "segment": ["A", None, "B", None] * 10,
            "churn": [0, 1] * 20,
            "notes": ["ok", " prefers email ", None, "vip"] * 10,
        }
    )


def test_insight_report_profiles_actionable_issues() -> None:
    report = fd.insight_report(
        _crm_frame(),
        dataset_name="crm_export",
        strategy="balanced",
        target_column="churn",
        id_columns=("customer_id",),
        preserve_columns=("notes",),
    )

    assert isinstance(report, fd.FreshDataInsightReport)
    payload = report.to_dict()
    assert json.dumps(payload)
    assert payload["schema_version"] == "freshdata.insight.v1"
    assert payload["report_type"] == "anomaly_insight"
    assert payload["dataset"]["name"] == "crm_export"
    assert payload["run"]["api"] == "fd.profile"
    assert payload["run"]["strategy"] == "balanced"
    assert payload["summary"]["issue_count"] >= 4
    assert payload["surfaces"]["cli"]["json_flag"] == "freshdata profile crm_export --json"

    age_issue = next(i for i in payload["issues"] if i["column"] == "age")
    assert age_issue["severity"] == "high"
    assert age_issue["inferred_role"] == "numeric"
    assert "fd.clean" in age_issue["fix_code"]
    assert "strategy=\"aggressive\"" in age_issue["fix_code"]
    assert age_issue["backend_requirement"] == "presentation_only"

    salary_issue = next(i for i in payload["issues"] if i["column"] == "salary")
    assert "outlier_method=\"iqr\"" in salary_issue["fix_code"]


def test_insight_report_maps_clean_report_actions() -> None:
    df = _crm_frame()
    _, clean_report = fd.clean(
        df,
        strategy="balanced",
        target_column="churn",
        id_columns=("customer_id",),
        preserve_columns=("notes",),
        return_report=True,
    )

    insight = fd.insight_report(df, clean_report=clean_report, dataset_name="crm_export")
    payload = insight.to_dict()

    assert payload["summary"]["action_count"] == len(payload["actions"])
    assert payload["actions"]
    first = payload["actions"][0]
    expected_keys = {
        "step",
        "column",
        "description",
        "count",
        "rationale",
        "risk",
        "confidence",
        "status",
    }
    assert expected_keys <= set(first)
    assert any(a["step"] == "missing" and a["column"] == "age" for a in payload["actions"])


def test_insight_report_html_renders_issue_and_action_context() -> None:
    df = _crm_frame()
    _, clean_report = fd.clean(df, return_report=True)
    html = fd.insight_report(df, clean_report=clean_report).to_html()

    assert '<div class="fd-report"' in html
    assert "FreshData insight report" in html
    assert "Action intelligence" in html
    assert "fd.clean" in html


def test_clean_impact_report_links_actions_to_before_after_diffs() -> None:
    df = _crm_frame()
    df["segment"] = [None if i % 10 == 0 else "A" for i in range(len(df))]
    cleaned, clean_report = fd.clean(
        df,
        strategy="balanced",
        target_column="churn",
        id_columns=("customer_id",),
        preserve_columns=("notes",),
        return_report=True,
    )

    insight = fd.insight_report(
        df,
        cleaned_df=cleaned,
        clean_report=clean_report,
        dataset_name="crm_export",
        compare_strategies=("conservative", "balanced", "aggressive"),
    )
    payload = insight.to_dict()

    assert payload["report_type"] == "clean_impact"
    assert payload["run"]["api"] == "fd.clean"
    assert payload["summary"]["rows_before"] == len(df)
    assert payload["summary"]["rows_after"] == len(cleaned)
    assert (
        payload["summary"]["missing_delta"]
        == clean_report.missing_after - clean_report.missing_before
    )
    assert payload["strategy_comparison"]["source"] == "fd.compare_plans"
    assert {row["strategy"] for row in payload["strategy_comparison"]["records"]} >= {
        "conservative",
        "balanced",
        "aggressive",
    }

    segment_action = next(
        a for a in payload["actions"] if a["step"] == "missing" and a["column"] == "segment"
    )
    assert (
        segment_action["impact"]["before"]["missing"]
        > segment_action["impact"]["after"]["missing"]
    )
    assert any(i["recommended_action_id"] == segment_action["id"] for i in payload["issues"])


def test_enterprise_trust_report_surfaces_validation_gate_and_lineage() -> None:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "country": ["US", "CA", "XX", "ZZ"],
            "revenue": [120.0, 240.0, None, 500.0],
        }
    )
    result = clean_enterprise(
        df,
        enterprise=EnterpriseConfig(
            semantic=(
                SemanticValidatorConfig(
                    name="country_reference",
                    kind="reference",
                    columns=("country",),
                    reference=("US", "CA", "GB"),
                ),
            ),
            fail_under_trust=100,
        ),
    )

    insight = fd.insight_report(df, enterprise_result=result, dataset_name="crm_export")
    payload = insight.to_dict()

    assert payload["report_type"] == "trust_gate"
    assert payload["run"]["api"] == "enterprise.validation"
    assert payload["trust"]["gate"]["passed"] is False
    assert payload["trust"]["gate"]["threshold"] == 100
    assert payload["lineage"]["openlineage_run_id"] == result.lineage.run_id

    issue = next(i for i in payload["issues"] if i["id"] == "semantic.country_reference.country")
    assert issue["severity"] == "high"
    assert issue["column"] == "country"
    assert issue["evidence"]["invalid_samples"] == ["XX", "ZZ"]
    assert "SemanticValidatorConfig" in issue["fix_code"]

    ci = payload["surfaces"]["cli"]["ci_summary"]
    assert "freshdata trust gate FAILED" in ci
    assert "country_reference" in ci


def test_insight_html_renders_strategy_and_trust_sections() -> None:
    df = _crm_frame()
    cleaned, clean_report = fd.clean(df, return_report=True)
    clean_html = fd.insight_report(
        df,
        cleaned_df=cleaned,
        clean_report=clean_report,
        compare_strategies=("balanced", "aggressive"),
    ).to_html()

    assert "Strategy comparison" in clean_html
    assert "fd.compare_plans" in clean_html

    score = fd.compute_trust_score(df)
    trust_html = fd.trust_gate_report(df, score, fail_under=100).to_html()

    assert "Trust gate" in trust_html
    assert "freshdata trust gate FAILED" in trust_html


def test_insight_report_clean_frame_has_no_issues_and_text_summary() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    report = fd.insight_report(df)
    payload = report.to_dict()

    assert payload["summary"]["issue_count"] == 0
    assert payload["summary"]["highest_severity"] == "none"
    assert "freshdata insight report" in str(report)
    assert "FreshDataInsightReport" in repr(report)
    assert "strategy_comparison" not in payload


def test_insight_report_rejects_non_dataframe_inputs() -> None:
    score = {"overall": 99.0, "grade": "A"}

    with pytest.raises(TypeError):
        fd.insight_report("not a frame")
    with pytest.raises(TypeError):
        fd.trust_gate_report("not a frame", score)


def test_table_level_action_gets_frame_level_impact() -> None:
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    cleaned, clean_report = fd.clean(df, return_report=True, verbose=False)

    payload = fd.insight_report(df, cleaned_df=cleaned, clean_report=clean_report).to_dict()
    table_action = next(action for action in payload["actions"] if action["column"] is None)

    assert table_action["impact"]["before"]["rows"] == 3
    assert table_action["impact"]["after"]["rows"] == 2
    assert table_action["id"].startswith("action.table.")


def test_unhashable_column_unique_count_stays_serializable() -> None:
    df = pd.DataFrame({"payload": [[1], [2], None], "label": ["a", "b", "c"]})

    payload = fd.insight_report(df).to_dict()
    issue = next(i for i in payload["issues"] if i["column"] == "payload")

    assert issue["evidence"]["unique"] is None


def test_fix_code_branches_use_real_clean_config_parameters() -> None:
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(25)],
            "label": [0, 1] * 12 + [0],
            "notes": [f"long free text value number {i}" for i in range(25)],
            "segment": ["A", None, "B", "A", "B"] * 5,
        }
    )

    issues = {issue["column"]: issue for issue in fd.insight_report(df).to_dict()["issues"]}

    assert "id_columns=('user_id',)" in issues["user_id"]["fix_code"]
    assert "preserve_columns=('notes',)" in issues["notes"]["fix_code"]
    assert 'strategy="balanced"' in issues["segment"]["fix_code"]


def test_target_column_fix_branch_is_actionable() -> None:
    df = pd.DataFrame(
        {
            "prediction": [0, 1, None, 1, 0] * 6,
            "feature": list(range(30)),
        }
    )

    payload = fd.insight_report(df, target_column="prediction").to_dict()
    issue = next(i for i in payload["issues"] if i["column"] == "prediction")

    assert "target_column='prediction'" in issue["fix_code"]
    assert issue["inferred_role"] == "target"


def test_trust_gate_report_handles_dict_scalar_pass_and_no_threshold() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})

    dict_payload = fd.trust_gate_report(df, {"overall": 99.0, "grade": "A"}).to_dict()
    scalar_payload = fd.trust_gate_report(df, 99.0, fail_under=90).to_dict()

    assert dict_payload["trust"]["gate"]["passed"] is None
    assert "NOT EVALUATED" in dict_payload["surfaces"]["cli"]["ci_summary"]
    assert scalar_payload["trust"]["gate"]["passed"] is True
    assert "PASSED" in scalar_payload["surfaces"]["cli"]["ci_summary"]


def test_enterprise_validation_without_invalid_values_keeps_no_validation_issue() -> None:
    df = pd.DataFrame({"country": ["US", "CA"], "value": [1, 2]})
    result = clean_enterprise(
        df,
        enterprise=EnterpriseConfig(
            semantic=(
                SemanticValidatorConfig(
                    name="country_reference",
                    kind="reference",
                    columns=("country",),
                    reference=("US", "CA"),
                ),
            ),
        ),
    )

    payload = fd.insight_report(df, enterprise_result=result).to_dict()

    assert not [issue for issue in payload["issues"] if issue["id"].startswith("semantic.")]
    assert payload["trust"]["gate"]["passed"] is None


def test_to_pandas_safe_returns_none_for_unknown_object() -> None:
    class UnknownFrame:
        pass

    assert to_pandas_safe(None) is None
    assert to_pandas_safe(UnknownFrame()) is None


def test_insight_internal_payload_helpers_cover_serialization_edges() -> None:
    class BrokenScalar:
        def item(self) -> object:
            raise TypeError("not scalar")

    class PlainTracker:
        pass

    assert _action_impact("missing", before=None, after=None) == {"before": {}, "after": {}}
    assert _frame_stats(None) == {}
    assert _json_scalar(BrokenScalar()).__class__ is BrokenScalar
    assert _frame_records(pd.DataFrame({"a": [1], "b": [float("nan")]})) == [
        {"a": 1, "b": None}
    ]
    assert _lineage_from_tracker(None)["input_datasets"] == []
    assert _lineage_from_tracker(PlainTracker()) == {
        "openlineage_run_id": None,
        "input_datasets": ["input"],
        "output_datasets": ["output"],
        "impacted_assets": [],
        "events": [],
    }


def test_insight_html_renderer_empty_sections_and_dispatch_error() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = fd.insight_report(df)
    html = report.to_html()

    assert "no profile issues detected" in html
    assert "pass clean_report= to attach applied actions" in html

    with pytest.raises(ValueError, match="no renderer registered"):
        render(report, "unknown_kind")
