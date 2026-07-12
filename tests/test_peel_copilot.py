"""Layer 9: CopilotReport display and the privacy-over-trust invariant (spec §8)."""

from __future__ import annotations

import pytest

from freshdata.enterprise.metrics import TrustScore
from freshdata.experimental.ai_copilot import (
    CleaningPlan,
    CopilotReport,
    DetectedProblem,
    PlanStep,
)
from freshdata.render.normalize import normalize
from freshdata.render.options import get_display, reset_display
from freshdata.render.plain import render_plain


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for var in ("FRESHDATA_DISPLAY", "FRESHDATA_LEGACY_DISPLAY", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)
    reset_display()
    yield
    reset_display()


def make_report(
    *,
    pii="2 columns look like personal data (email, phone)",
    problems=(("missing", "medium", "income is 38% missing", "income", 380),),
    policy_violations=(),
    trust_overall=71.0,
    grade_dims=(62.0, 78.0, 99.0, 45.0),
    provider_error=None,
    steps=(("Mask PII", "fd.clean(df, context='mask emails')"),),
) -> CopilotReport:
    audit = {"model_context_sha256": "3f9cabc", "pii_columns": ["email", "phone"]}
    if provider_error:
        audit["provider_error"] = provider_error
    completeness, validity, uniqueness, consistency = grade_dims
    return CopilotReport(
        goal="prep for churn model",
        summary="deterministic summary text",
        problems=tuple(
            DetectedProblem(kind=k, severity=s, detail=d, column=c, count=n)
            for k, s, d, c, n in problems
        ),
        pii_warning=pii,
        policy_violations=tuple(policy_violations),
        cleaning_plan=CleaningPlan(
            steps=tuple(
                PlanStep(order=i + 1, action=a, rationale="because", tool=t)
                for i, (a, t) in enumerate(steps)
            )
        ),
        recommended_code="df = fd.clean(df, context='mask emails')",
        trust=TrustScore(
            overall=trust_overall,
            completeness=completeness,
            validity=validity,
            uniqueness=uniqueness,
            consistency=consistency,
            n_rows=10_000,
            n_cols=14,
        ),
        model_context={"columns": ["email", "phone"], "sample": "masked"},
        audit=audit,
        narrative=None if provider_error else "the data looks...",
    )


class TestPrivacyOverTrust:
    def test_pii_ranks_first_even_with_grade_a(self):
        rep = make_report(trust_overall=95.0, grade_dims=(99, 99, 99, 99))
        view = normalize(rep)
        assert view.attention[0].id == "A1"
        assert view.attention[0].domain == "privacy"
        assert "REVIEW" in view.status  # grade A cannot suppress REVIEW

    def test_high_trust_does_not_remove_pii_from_glance(self):
        rep = make_report(trust_overall=95.0, grade_dims=(99, 99, 99, 99))
        out = render_plain(normalize(rep), get_display(mode="standard"))
        assert "personal data" in out
        assert "REVIEW" in out

    def test_policy_violation_outranks_lower_severity_data_quality(self):
        # spec §5.3 order: privacy > corruption > policy > reliability > cosmetic,
        # so policy outranks a medium/low data-quality issue but not a corruption one
        rep = make_report(
            pii=None,
            problems=(("consistency", "low", "minor casing noise", "tier", 3),),
            policy_violations=({"message": "income must be clustered", "column": "income"},),
        )
        view = normalize(rep)
        assert view.attention[0].domain == "policy"
        assert view.attention[-1].domain == "cosmetic"


class TestQueue:
    def test_ids_assigned_after_ranking(self):
        view = normalize(make_report())
        assert [a.id for a in view.attention] == [
            f"A{i}" for i in range(1, len(view.attention) + 1)
        ]

    def test_no_pii_is_explicit_not_absent(self):
        view = normalize(make_report(pii=None))
        privacy_metric = next(m for m in view.metrics if m.label == "privacy")
        assert "no PII detected" in privacy_metric.value

    def test_trust_metric_carries_scope_note(self):
        view = normalize(make_report())
        trust_metric = next(m for m in view.metrics if m.label == "trust")
        assert "data quality only" in trust_metric.value
        assert "71/100 (C)" in trust_metric.value


class TestExperimentalAndProvider:
    def test_experimental_banner_always_present(self):
        assert "Experimental API" in normalize(make_report()).banner

    def test_provider_failure_is_labeled_not_absent(self):
        view = normalize(make_report(provider_error="TimeoutError: provider timed out"))
        assert "PARTIAL" in view.status
        assert "narrative unavailable" in view.banner.lower()
        assert "deterministic findings above are complete" in view.banner

    def test_next_step_is_first_plan_tool(self):
        view = normalize(make_report())
        assert view.next_step == "fd.clean(df, context='mask emails')"


class TestRendering:
    def test_attention_property_matches_view(self):
        rep = make_report()
        assert rep.attention == normalize(rep).attention

    def test_repr_html_renders_peel_for_copilot(self):
        # copilot has no legacy renderer → always Peel
        html = make_report().to_html()
        assert "freshdata ai-copilot" in html
        assert "Needs attention" in html
        assert "personal data" in html

    def test_str_still_returns_summary_string(self):
        # backward-compat: __str__ unchanged
        assert str(make_report()) == "deterministic summary text"

    def test_verbose_shows_plan_and_masked_context(self):
        out = render_plain(normalize(make_report()), get_display(mode="verbose"))
        assert "Cleaning plan" in out
        assert "Model context" in out
        assert "masked" in out

    def test_recommended_code_flagged_machine_generated(self):
        out = render_plain(normalize(make_report()), get_display(mode="verbose"))
        assert "machine-generated" in out
