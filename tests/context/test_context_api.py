"""API threading: fd.clean / suggest_plan / clean_csv / Cleaner / fd.validate."""

import pandas as pd
import pytest

import freshdata as fd

ECOMMERCE = """
This is an ecommerce customer dataset.
CustomerID is unique.
Emails must be valid.
Phone numbers are Indian.
Missing Age should be estimated only if confidence >95%.
Never modify revenue values.
"""


@pytest.fixture()
def df():
    return pd.DataFrame(
        {
            "cust_id": [1, 2, 2, 4],
            "full_name": ["Asha", "Ben", "Ben", "Dana"],
            "email_addr": ["a@x.com", "b@x.com", "not-an-email", None],
            "mobile": ["9876543210", None, "9123456789", "9111111111"],
            "age": [25.0, None, 40.0, 35.0],
            "monthly_revenue": [100.0, 200.0, 99999.0, 50.0],
            "city": ["Pune", "Delhi", "Delhi", "Goa"],
        }
    )


# -- the flagship end-to-end fixture -------------------------------------------


def test_ecommerce_context_resolves_to_expected_columns(df):
    policy = fd.compile_context(ECOMMERCE, df=df)
    resolved = {c.resolved_from: c.column for c in policy.constraints}
    assert resolved == {
        "CustomerID": "cust_id",
        "Emails": "email_addr",
        "Phone numbers": "mobile",
        "Age": "age",
        "revenue": "monthly_revenue",
    }
    assert policy.dataset_domain == "ecommerce_customer"
    assert not policy.unresolved
    assert not policy.issues


def test_clean_with_context_protects_and_reports(df):
    cleaned, report = fd.clean(df, context=ECOMMERCE, return_report=True, verbose=False)
    # the protected column is byte-identical
    pd.testing.assert_series_equal(cleaned["monthly_revenue"], df["monthly_revenue"])
    context_actions = [a for a in report.actions if a.step == "context"]
    assert any("compiled context policy" in a.description for a in context_actions)
    assert any(a.column == "monthly_revenue" for a in context_actions)
    policy_dict = context_actions[0].metadata["policy"]
    assert len(policy_dict["constraints"]) == 5


def test_clean_with_policy_matches_context(df):
    policy = fd.compile_context(ECOMMERCE, df=df)
    via_context = fd.clean(df, context=ECOMMERCE, verbose=False)
    via_policy = fd.clean(df, policy=policy, verbose=False)
    pd.testing.assert_frame_equal(via_context, via_policy)


def test_schema_free_policy_resolves_at_clean_time(df):
    policy = fd.compile_context("Never modify revenue values.")  # no schema yet
    cleaned = fd.clean(df, policy=policy, verbose=False)
    pd.testing.assert_series_equal(cleaned["monthly_revenue"], df["monthly_revenue"])


# -- compatibility guarantees ---------------------------------------------------


def test_context_none_changes_nothing(df):
    baseline, base_report = fd.clean(df, return_report=True, verbose=False)
    explicit, explicit_report = fd.clean(
        df, context=None, policy=None, strict=False, return_report=True, verbose=False
    )
    pd.testing.assert_frame_equal(baseline, explicit)
    assert len(base_report.actions) == len(explicit_report.actions)
    assert not any(a.step == "context" for a in base_report.actions)


def test_default_config_unchanged():
    cfg = fd.CleanConfig()
    assert cfg.context is None
    assert cfg.policy is None
    assert cfg.strict is False


def test_context_and_policy_together_raise(df):
    policy = fd.compile_context("age is unique.", df=df)
    with pytest.raises(TypeError, match="mutually exclusive"):
        fd.clean(df, context="age is unique.", policy=policy)
    with pytest.raises(ValueError, match="mutually exclusive"):
        fd.CleanConfig(context="age is unique.", policy=policy)


def test_policy_must_be_a_context_policy(df):
    with pytest.raises(TypeError, match="ContextPolicy"):
        fd.clean(df, policy={"not": "a policy"})


def test_context_rejected_on_non_pandas_engines(df):
    with pytest.raises(TypeError, match="pandas engine"):
        fd.clean(df, context="age is unique.", engine="polars")


def test_strict_clean_raises_on_unparsed(df):
    with pytest.raises(fd.PolicyError):
        fd.clean(df, context="Total gibberish here.", strict=True, verbose=False)
    # non-strict surfaces the same problem as a warning instead
    _, report = fd.clean(
        df, context="Total gibberish here.", return_report=True, verbose=False
    )
    assert any("unparsed_sentence" in w for w in report.warnings)


# -- Cleaner / suggest_plan / clean_csv -----------------------------------------


def test_cleaner_accepts_context(df):
    cleaner = fd.Cleaner(context="Never modify revenue values.", verbose=False)
    cleaned = cleaner.clean(df)
    pd.testing.assert_series_equal(cleaned["monthly_revenue"], df["monthly_revenue"])
    assert cleaner.config.context == "Never modify revenue values."


def test_suggest_plan_accepts_context(df):
    plan = fd.suggest_plan(df, context=ECOMMERCE)
    assert plan.config.policy is not None
    assert "monthly_revenue" in plan.config.preserve_columns
    assert "cust_id" in plan.config.id_columns
    with pytest.raises(fd.PolicyError):
        fd.suggest_plan(df, context="Gibberish.", strict=True)


def test_clean_csv_accepts_context(df, tmp_path):
    src = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    df.to_csv(src, index=False)
    fd.clean_csv(src, output_path=out, context="Never modify revenue values.", verbose=False)
    written = pd.read_csv(out)
    assert written["monthly_revenue"].tolist() == df["monthly_revenue"].tolist()


# -- fd.validate ----------------------------------------------------------------


def test_validate_reports_violations_without_mutating(df):
    before = df.copy(deep=True)
    findings = fd.validate(df, context=ECOMMERCE)
    pd.testing.assert_frame_equal(df, before)  # non-mutating
    rules = {f.rule_name for f in findings}
    assert "context.unique" in rules  # cust_id has a duplicate
    assert "context.protected" in rules
    assert findings.errors and isinstance(findings, fd.FindingList)


def test_validate_allowed_values_and_range():
    frame = pd.DataFrame({"status": ["active", "zombie"], "age": [17, 50]})
    findings = fd.validate(
        frame,
        context="Allowed status values are active, inactive. Age must be between 18 and 100.",
    )
    by_rule = {f.rule_name: f for f in findings}
    assert by_rule["context.allowed_values"].extra["n_violations"] == 1
    assert by_rule["context.range"].extra["n_violations"] == 1
    assert by_rule["context.range"].severity == "error"


def test_validate_surfaces_unresolved_and_unparsed(df):
    findings = fd.validate(df, context="heart_rate is unique. Utter nonsense.")
    rules = {f.rule_name for f in findings}
    assert "context.unresolved_reference" in rules
    assert "context.unparsed_sentence" in rules
    assert not findings.errors  # only warnings
    assert findings.warnings


def test_validate_clean_frame_has_no_errors():
    frame = pd.DataFrame({"cust_id": [1, 2, 3], "status": ["active", "active", "inactive"]})
    findings = fd.validate(
        frame,
        context="cust_id is unique. Allowed status values are active, inactive.",
    )
    assert not findings.errors


def test_validate_requires_context_or_policy(df):
    with pytest.raises(TypeError, match="context="):
        fd.validate(df)
    with pytest.raises(TypeError, match="mutually exclusive"):
        fd.validate(df, context="x is unique.", policy=fd.compile_context("age is unique."))


def test_validate_accepts_precompiled_policy(df):
    policy = fd.compile_context("CustomerID is unique.")
    findings = fd.validate(df, policy=policy)
    assert any(f.rule_name == "context.unique" for f in findings)
