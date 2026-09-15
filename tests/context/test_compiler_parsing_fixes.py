"""Regression tests for context parsing and compile fixes (#301-#304)."""

import warnings

import pandas as pd
import pytest

import freshdata as fd
from freshdata import PolicyError, compile_context
from freshdata.context import parse_context, resolve_policy, split_sentences
from freshdata.context.lexicon import parse_confidence
from freshdata.context.normalize import split_value_list


def _values(text, **kwargs):
    policy = compile_context(text, **kwargs)
    return {c.column: c.params["values"] for c in policy.constraints if c.rule == "allowed_values"}


# -- #301: allowed-values lists ------------------------------------------------------


def test_issue_301_multiword_and_slash_values_stay_whole():
    df = pd.DataFrame({"country": ["Trinidad and Tobago", "Chile"], "code": ["N/A", "OK"]})
    text = (
        "Allowed country values are Trinidad and Tobago, Chile. "
        "Allowed code values are N/A, OK."
    )
    assert _values(text, df=df) == {
        "country": ["Trinidad and Tobago", "Chile"],
        "code": ["N/A", "OK"],
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        findings = fd.validate(df, context=text)
    assert not [f for f in findings if f.rule_name == "context.allowed_values"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("active, inactive, pending", ("active", "inactive", "pending")),
        ("active, inactive or pending", ("active", "inactive", "pending")),
        ("active, inactive, or pending", ("active", "inactive", "pending")),
        ("active, inactive, and pending", ("active", "inactive", "pending")),
        ("male, female or other", ("male", "female", "other")),
        ("email and phone", ("email", "phone")),
        ("active or inactive", ("active", "inactive")),
        ("a; b", ("a", "b")),
        ("400001, 400002", ("400001", "400002")),
        ("[a, b]", ("a", "b")),
        ("CA, OR, WA", ("CA", "OR", "WA")),
        ("N/A", ("N/A",)),
        (",,", ()),
    ],
)
def test_split_value_list_forms(raw, expected):
    assert split_value_list(raw) == expected


def test_quoted_values_are_never_split():
    assert split_value_list("Chile or 'Trinidad and Tobago'") == ("Chile", "Trinidad and Tobago")
    assert split_value_list("\"Bosnia and Herzegovina\", 'N/A (none)'") == (
        "Bosnia and Herzegovina",
        "N/A (none)",
    )
    assert split_value_list("['x', 'y']") == ("x", "y")


def test_apostrophes_inside_words_do_not_open_quotes():
    assert split_value_list("Don't know, won't say") == ("Don't know", "won't say")


def test_quoted_dotted_values_keep_their_periods():
    assert _values("Allowed country values are 'U.S.', 'U.K.'.", columns=["country"]) == {
        "country": ["U.S.", "U.K."]
    }


def test_dedup_key_split_is_unchanged():
    (candidate,) = parse_context("Deduplicate by email and phone.").candidates
    assert candidate.column_refs == ("email", "phone")
    (candidate,) = parse_context("Deduplicate by email, phone and city.").candidates
    assert candidate.column_refs == ("email", "phone", "city")


# -- #302: sentence splitting ----------------------------------------------------------


def test_issue_302_dotted_column_name_is_one_sentence():
    df = pd.DataFrame({"file.name": ["a", "b"], "name": ["x", "x"]})
    policy = compile_context(
        "file.name is unique.", df=df, config=fd.CleanConfig(column_names=False)
    )
    assert [(c.rule, c.column) for c in policy.constraints] == [("unique", "file.name")]
    assert not policy.issues


def test_dotted_column_name_resolves_after_snake_casing():
    df = pd.DataFrame({"file.name": ["a", "b"], "name": ["x", "x"]})
    policy = compile_context("file.name is unique.", df=df)
    assert [(c.rule, c.column) for c in policy.constraints] == [("unique", "file_name")]


def test_splitter_keeps_decimals_and_abbreviations():
    assert split_sentences(
        "Allowed country values are U.S., U.K., Canada. Age must be between 0.5 and 1.5."
    ) == (
        "Allowed country values are U.S., U.K., Canada",
        "Age must be between 0.5 and 1.5",
    )
    assert split_sentences("CustomerID is unique. Emails must be valid!\nProtect age.") == (
        "CustomerID is unique",
        "Emails must be valid",
        "Protect age",
    )


# -- #303: confidence gates ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Impute missing income only if 3 neighbours agree",
        "Fill missing age when age > 50",
        "Missing age should be estimated only if 2 sources agree.",
    ],
)
def test_issue_303_bare_numbers_are_not_confidence(text):
    policy = compile_context(text, columns=["income", "age"])
    assert not policy.constraints
    assert [i.kind for i in policy.issues] == ["unparsed_sentence"]
    with pytest.raises(PolicyError, match="unparsed_sentence"):
        compile_context(text, columns=["income", "age"], strict=True)


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        (">95%", 0.95),
        ("confidence > 95%", 0.95),
        ("confidence above 0.95", 0.95),
        ("only if confidence >= 95 percent", 0.95),
        ("confidence >60%", 0.6),
        ("certainty of at least 0.8", 0.8),
        ("probability is over 90", 0.9),
        ("at least 90% sure", 0.9),
        ("confidence level of 0.7", 0.7),
    ],
)
def test_explicit_confidence_phrases_still_parse(phrase, expected):
    assert parse_confidence(phrase) == expected


@pytest.mark.parametrize("phrase", ["3 neighbours agree", "age > 50", "whenever it feels right"])
def test_non_confidence_numbers_return_none(phrase):
    assert parse_confidence(phrase) is None


def test_documented_impute_phrasings_still_compile():
    for text, expected in (
        ("Missing Age should be estimated only if confidence >95%.", 0.95),
        ("Missing age must be imputed only when confidence above 0.95.", 0.95),
        ("Estimate missing quantity only if confidence >= 90 percent.", 0.9),
    ):
        (c,) = compile_context(text, columns=["age", "quantity"]).constraints
        assert c.params == {"min_confidence": expected}, text


# -- #304: protection conflicts after late resolution ----------------------------------

CONFLICT = "Never modify email. Emails must be valid."


def test_issue_304_schema_free_policy_raises_under_strict():
    df = pd.DataFrame({"email": ["a@b.com", "c@d.com"]})
    with pytest.raises(PolicyError, match="protection_conflict"):
        compile_context(CONFLICT, columns=["email"], strict=True)
    policy = compile_context(CONFLICT, strict=True)  # schema-free: nothing to conflict yet
    with pytest.raises(PolicyError, match="protection_conflict"):
        fd.clean(df, policy=policy, strict=True, verbose=False)


def test_schema_free_resolution_matches_schema_bound_compile():
    bound = compile_context(CONFLICT, columns=["email"])
    late = resolve_policy(compile_context(CONFLICT), ["email"])
    assert [(c.rule, c.column, c.action) for c in late.constraints] == [
        (c.rule, c.column, c.action) for c in bound.constraints
    ]
    assert [(i.kind, i.severity, i.columns) for i in late.issues] == [
        (i.kind, i.severity, i.columns) for i in bound.issues
    ]
    repair = [c for c in late.constraints if c.rule == "valid_format"][0]
    assert repair.action == "validate_only"


def test_non_strict_schema_free_policy_records_issue_and_protects():
    df = pd.DataFrame({"email": ["A@B.COM ", "c@d.com"]})
    policy = compile_context(CONFLICT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fd.clean(df, policy=policy, verbose=False)
    assert out["email"].tolist() == df["email"].tolist()
    resolved = resolve_policy(policy, ["email"])
    assert [i.kind for i in resolved.issues] == ["protection_conflict"]


def test_late_resolution_detects_restatement_and_does_not_duplicate_issues():
    policy = compile_context("Age must be between 1 and 5. age must be between 2 and 6.")
    assert not policy.issues  # different surface refs: no restatement known yet
    resolved = resolve_policy(policy, ["age"])
    (constraint,) = resolved.constraints
    assert constraint.params == {"lo": 2, "hi": 6}
    assert [i.kind for i in resolved.issues] == ["superseded"]
    assert resolve_policy(resolved, ["age"]) is resolved


def test_resolve_policy_without_new_resolution_keeps_issues():
    policy = compile_context("heart_rate is unique. Gibberish here.")
    resolved = resolve_policy(policy, ["age"])
    assert resolved.issues == policy.issues
    assert not resolved.constraints
