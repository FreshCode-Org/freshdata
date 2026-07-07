"""Tests for the experimental AI Copilot (freshdata.experimental.ai_copilot)."""

from __future__ import annotations

import contextlib
import io
import json

import pandas as pd
import pytest

from freshdata.experimental.ai_copilot import (
    CleaningPlan,
    CopilotReport,
    analyze_dataset,
)

RAW_EMAIL = "jane.doe@example.com"
RAW_PHONE = "555-201-3344"

POLICY = {
    "email": "must_mask",
    "phone": "must_mask",
    "age": "must_be_between_0_and_120",
    "salary": "must_be_positive",
    "plan": "normalize_spelling",
}


@pytest.fixture()
def messy_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "email": [
                RAW_EMAIL,
                "bob@example.com",
                None,
                "carol@example.com",
                "d@example.com",
                "e@example.com",
            ],
            "phone": [
                RAW_PHONE,
                "555-202-1188",
                "555-203-9021",
                None,
                "555-205-3300",
                "555-206-1417",
            ],
            "age": [34, 250, -1, None, 45, 30],
            "salary": [72000.0, -500.0, 64000.0, None, 88000.0, 59000.0],
            "plan": ["Gold", "gold", "GOLD", "Silver", "silver", "Gold"],
            "signup_date": [
                "2023-05-14",
                "14/05/2023",
                "2023-06-01",
                "2023.06.02",
                "June 3 2023",
                "2023-06-04",
            ],
        }
    )
    # one exact duplicate row
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


def test_report_shape_and_printability(messy_df: pd.DataFrame) -> None:
    report = analyze_dataset(messy_df, context_policy=POLICY)
    assert isinstance(report, CopilotReport)
    assert isinstance(report.cleaning_plan, CleaningPlan)
    assert report.cleaning_plan.steps
    # the three headline surfaces are directly printable, non-empty strings
    for text in (report.summary, str(report.cleaning_plan), report.recommended_code):
        assert isinstance(text, str) and text.strip()
    # round-trips through JSON
    payload = json.loads(report.to_json())
    assert payload["audit"]["engine"] == "deterministic-local"
    assert payload["trust_score"]["overall"] > 0


def test_deterministic(messy_df: pd.DataFrame) -> None:
    a = analyze_dataset(messy_df, context_policy=POLICY)
    b = analyze_dataset(messy_df, context_policy=POLICY)
    assert a.summary == b.summary
    assert a.problems == b.problems
    assert a.recommended_code == b.recommended_code
    assert str(a.cleaning_plan) == str(b.cleaning_plan)


def test_detects_expected_problems(messy_df: pd.DataFrame) -> None:
    report = analyze_dataset(messy_df, context_policy=POLICY)
    kinds = {p.kind for p in report.problems}
    assert {"pii", "policy_violation", "duplicate_rows", "missing_values"} <= kinds
    assert "category_noise" in kinds  # Gold/gold/GOLD
    assert "mixed_date_formats" in kinds
    # policy violations carry the offending columns
    violation_cols = {p.column for p in report.problems if p.kind == "policy_violation"}
    assert {"age", "salary"} <= violation_cols
    assert report.policy_violations


def test_no_raw_pii_anywhere_in_report(messy_df: pd.DataFrame) -> None:
    report = analyze_dataset(messy_df, context_policy=POLICY)
    dumped = report.to_json()
    assert RAW_EMAIL not in dumped
    assert RAW_PHONE not in dumped
    assert report.pii_warning is not None
    # masked samples are present but hashed
    samples = report.model_context["sample_rows_masked"]
    assert samples and all(row["email"] != RAW_EMAIL for row in samples)


def test_schema_only_privacy_mode(messy_df: pd.DataFrame) -> None:
    report = analyze_dataset(messy_df, privacy="schema_only", context_policy=POLICY)
    assert "sample_rows_masked" not in report.model_context
    assert RAW_EMAIL not in report.to_json()


def test_invalid_privacy_mode_rejected(messy_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="privacy"):
        analyze_dataset(messy_df, privacy="send_everything")


def test_unsupported_policy_rule_rejected(messy_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unsupported context_policy rule"):
        analyze_dataset(messy_df, context_policy={"age": "must_be_prime"})


def test_date_column_not_flagged_as_phone(messy_df: pd.DataFrame) -> None:
    report = analyze_dataset(messy_df, context_policy=POLICY)
    pii_cols = {p.column for p in report.problems if p.kind == "pii"}
    assert "signup_date" not in pii_cols
    assert "signup_date" in report.audit["pii_suppressed_date_like"]


def test_provider_hook_receives_only_masked_context(messy_df: pd.DataFrame) -> None:
    prompts: list[str] = []

    def fake_provider(prompt: str) -> str:
        prompts.append(prompt)
        return "narrative from fake model"

    with pytest.warns(FutureWarning, match="experimental"):
        report = analyze_dataset(messy_df, context_policy=POLICY, provider=fake_provider)
    assert report.narrative == "narrative from fake model"
    assert report.audit["engine"] == "provider:fake_provider"
    assert len(prompts) == 1
    assert RAW_EMAIL not in prompts[0]
    assert RAW_PHONE not in prompts[0]


def test_provider_failure_never_breaks_report(messy_df: pd.DataFrame) -> None:
    def broken_provider(prompt: str) -> str:
        raise RuntimeError("model unavailable")

    with pytest.warns(FutureWarning):
        report = analyze_dataset(messy_df, context_policy=POLICY, provider=broken_provider)
    assert report.narrative is None
    assert report.audit["engine"] == "deterministic-local"
    assert "model unavailable" in report.audit["provider_error"]
    assert report.summary  # deterministic report still produced


def test_recommended_code_is_copy_ready(messy_df: pd.DataFrame, tmp_path) -> None:
    """The generated pipeline must run as-is against the analyzed file."""
    csv_path = tmp_path / "data.csv"
    messy_df.to_csv(csv_path, index=False)
    report = analyze_dataset(messy_df, context_policy=POLICY, source_hint=str(csv_path))
    namespace: dict[str, object] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(report.recommended_code, "<recommended_code>", "exec"), namespace)  # noqa: S102
    cleaned = namespace["cleaned"]
    assert isinstance(cleaned, pd.DataFrame)
    assert not cleaned.duplicated().any()
    assert cleaned["age"].dropna().between(0, 120).all()
    assert (cleaned["salary"].dropna() >= 0).all()
    assert RAW_EMAIL not in cleaned.to_csv()


def test_protected_unique_and_rule_lists() -> None:
    df = pd.DataFrame(
        {
            "revenue": [10.0, -5.0, 20.0],
            "uid": [1, 2, 2],
        }
    )
    report = analyze_dataset(
        df,
        context_policy={
            "revenue": ["never_modify", "must_be_positive"],
            "uid": "must_be_unique",
        },
    )
    assert "never modify revenue values" in report.audit["policy_sentences"]
    assert "uid is unique" in report.audit["policy_sentences"]
    assert "protected" in report.audit["compiled_policy"]
    violation_cols = {p.column for p in report.problems if p.kind == "policy_violation"}
    assert {"revenue", "uid"} <= violation_cols
    # printable dunder surfaces
    assert str(report) == report.summary
    assert str(report.cleaning_plan.steps[0]).startswith("1. ")


def test_non_string_policy_rule_rejected() -> None:
    df = pd.DataFrame({"age": [1, 2, 3]})
    with pytest.raises(TypeError, match="must be strings"):
        analyze_dataset(df, context_policy={"age": [42]})


def test_clean_dataset_yields_calm_report() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]})
    report = analyze_dataset(df)
    assert report.pii_warning is None
    assert not report.policy_violations
    assert not [p for p in report.problems if p.severity == "high"]
    # plan always ends with the verification step
    assert "audit" in report.cleaning_plan.steps[-1].action.lower()
