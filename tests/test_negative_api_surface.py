"""Negative-input contracts for the public ``freshdata`` API surface (Phase 19).

Every entry point named in ``freshdata.__all__`` that takes user data or user
configuration is fed deliberately invalid input, and each case asserts four
things rather than "something raised":

* the **exact** exception class (never bare ``Exception``),
* a message that names the offending parameter *and* the permitted form,
* **no partial mutation** of the caller's frame — verified with a digest that
  also covers dtypes, the index name and ``DataFrame.attrs`` (which
  ``assert_frame_equal`` ignores),
* determinism — the same bad input raises the same class and message twice.

Some cases deliberately pin *today's* weaker behaviour. Those carry an
``S3:``/``S2:`` comment describing the gap; they are regression anchors, not
endorsements, and they are expected to be rewritten when the gap is closed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.domains import UnknownDomainError
from freshdata.enterprise.contracts import ContractViolation
from freshdata.learning import ProfileFormatError
from freshdata.validation_suite import (
    ColumnRule,
    CrossColumnRule,
    ValidationError,
    ValidationSuite,
    run_suite,
)

# ── helpers ─────────────────────────────────────────────────────────────────────


def frame_digest(df: pd.DataFrame) -> str:
    """A digest of everything a partial mutation could disturb.

    ``assert_frame_equal`` ignores ``.attrs`` and the index *name*, so a step
    that stashed metadata or renamed the index on the caller's frame would slip
    through it. Hashing values, labels, dtypes, the index (values and name) and
    ``attrs`` together closes that hole.
    """
    parts = [
        pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes(),
        repr([repr(c) for c in df.columns]).encode(),
        repr([str(d) for d in df.dtypes]).encode(),
        repr(df.index.name).encode(),
        repr([repr(v) for v in df.index]).encode(),
        json.dumps(df.attrs, sort_keys=True, default=repr).encode(),
    ]
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part)
        digest.update(b"|")
    return digest.hexdigest()


def raises_twice(exc_type, match, call, frame=None):
    """Call *call* twice; assert the same exception and an untouched *frame*.

    Returns the first exception message so a caller can make further
    assertions on it.
    """
    before = frame_digest(frame) if frame is not None else None
    messages = []
    for _ in range(2):
        with pytest.raises(exc_type, match=match) as excinfo:
            call()
        assert type(excinfo.value) is exc_type, (
            f"expected exactly {exc_type.__name__}, got {type(excinfo.value).__name__}"
        )
        messages.append(str(excinfo.value))
        if frame is not None:
            assert frame_digest(frame) == before, "input frame was mutated by a failed call"
    assert messages[0] == messages[1], f"non-deterministic error message: {messages}"
    return messages[0]


@pytest.fixture
def df():
    """A small frame carrying an index name and ``attrs`` so mutation shows up."""
    frame = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "amount": ["  10 ", "20", "30", None, "5000", "12"],
            "notes": ["  hi  ", "yo", None, "hello", "ok", "sure"],
        },
        index=pd.Index([0, 1, 2, 3, 4, 5], name="row_no"),
    )
    frame.attrs["origin"] = "test_negative_api_surface"
    return frame


#: Values that are emphatically not a DataFrame.
NOT_A_FRAME = [
    pytest.param([1, 2, 3], id="list"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param(None, id="none"),
    pytest.param(np.array([[1, 2], [3, 4]]), id="ndarray"),
    pytest.param(42, id="scalar"),
]


# ── 1. wrong DataFrame type ─────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", NOT_A_FRAME)
def test_clean_rejects_non_frames_by_type(bad):
    message = raises_twice(
        TypeError,
        r"expected a pandas or polars DataFrame/LazyFrame",
        lambda: fd.clean(bad, verbose=False),
    )
    assert type(bad).__name__ in message


@pytest.mark.parametrize("bad", NOT_A_FRAME)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(fd.profile, id="profile"),
        pytest.param(fd.infer_roles, id="infer_roles"),
        pytest.param(fd.explain_clean, id="explain_clean"),
        pytest.param(lambda x: fd.validate(x, context="customer_id is unique."), id="validate"),
    ],
)
def test_read_only_entry_points_reject_non_frames_by_type(call, bad):
    """``profile``/``infer_roles``/``explain_clean``/``validate`` share one contract."""
    message = raises_twice(
        TypeError, r"expected pandas or polars DataFrame", lambda: call(bad)
    )
    assert type(bad).__name__ in message


@pytest.mark.parametrize("bad", NOT_A_FRAME)
def test_run_suite_rejects_non_frames_by_type(bad):
    suite = ValidationSuite(name="s", rules=(ColumnRule(name="customer_id", nullable=False),))
    message = raises_twice(
        TypeError, r"expected pandas or polars DataFrame", lambda: run_suite(bad, suite)
    )
    assert type(bad).__name__ in message


@pytest.mark.parametrize("bad", NOT_A_FRAME)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(fd.suggest_plan, id="suggest_plan"),
        pytest.param(fd.plan, id="plan"),
        pytest.param(fd.clean_text, id="clean_text"),
        pytest.param(fd.validate_fields, id="validate_fields"),
        pytest.param(lambda x: fd.compile_context("a is unique.", df=x), id="compile_context"),
    ],
)
def test_planning_entry_points_leak_attributeerror_for_non_frames(call, bad):
    """S3: these five leak ``AttributeError: ... has no attribute 'columns'``.

    ``clean``/``profile``/``infer_roles``/``explain_clean``/``validate``/
    ``run_suite`` all raise ``TypeError`` naming the accepted types. The five
    below reach ``require_unique_labels`` (or the context compiler) with a
    non-frame and surface an internal ``AttributeError`` that names neither the
    parameter nor the expected type. Pinned so the inconsistency is visible.
    """
    if bad is None:
        pytest.skip("compile_context(df=None) is a documented schema-free compile")
    raises_twice(AttributeError, r"has no attribute 'columns'", lambda: call(bad))


def test_compile_context_without_frame_is_schema_free_not_an_error():
    """Disproved suspicion: ``df=None`` is the documented schema-free compile."""
    policy = fd.compile_context("customer_id is unique.", df=None)
    assert isinstance(policy, fd.ContextPolicy)
    assert policy.constraints, "a schema-free policy should still carry its constraints"


# ── 2. duplicate / missing columns ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("func_name", "call"),
    [
        ("suggest_plan", fd.suggest_plan),
        ("clean_text", fd.clean_text),
        ("validate_fields", fd.validate_fields),
        ("explain_clean", fd.explain_clean),
        ("infer_roles", fd.infer_roles),
    ],
)
def test_duplicate_column_labels_name_the_function_and_the_labels(func_name, call):
    dup = pd.DataFrame([[1, "a"], [2, "b"]], columns=["x", "x"])
    message = raises_twice(
        ValueError,
        rf"{func_name} requires unique column labels",
        lambda: call(dup),
        frame=dup,
    )
    assert "['x']" in message


def test_clean_renames_duplicate_labels_instead_of_rejecting_them():
    """Disproved suspicion: ``clean`` is not inconsistent, it repairs by design.

    ``column_names`` deduplicates collisions as step 1, so ``clean``/``profile``
    accept what the planning entry points reject.
    """
    dup = pd.DataFrame([[1, "a"], [2, "b"]], columns=["x", "x"])
    cleaned = fd.clean(dup, verbose=False)
    assert list(cleaned.columns) == ["x", "x_2"]


def test_clean_text_names_the_missing_columns(df):
    message = raises_twice(
        KeyError,
        r"columns not in frame",
        lambda: fd.clean_text(df, columns=["nope", "also_nope"]),
        frame=df,
    )
    assert "nope" in message


def test_contract_gate_reports_the_missing_required_column(df):
    contract = {"name": "c", "columns": [{"name": "zzz", "dtype": "int64"}]}
    message = raises_twice(
        ContractViolation,
        r"required declared column 'zzz' is missing",
        lambda: fd.clean(df, contract=contract, verbose=False),
        frame=df,
    )
    assert "zzz" in message


def test_contract_mapping_without_a_name_raises_a_bare_keyerror(df):
    """S3: an incomplete contract mapping gives ``KeyError: 'name'``.

    Nothing says which parameter was malformed or that ``name``/``columns`` are
    the required keys — well short of the ``impute`` message's quality.
    """
    raises_twice(
        KeyError, r"'name'", lambda: fd.clean(df, contract={"columns": []}, verbose=False),
        frame=df,
    )


# ── 3. invalid thresholds and enum-valued options ───────────────────────────────


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        ({"missing_threshold_low": -0.1}, r"missing_threshold_low must be in \(0, 1\), got -0\.1"),
        ({"missing_threshold_high": 1.5}, r"missing_threshold_high must be in \(0, 1\), got 1\.5"),
        ({"duplicate_threshold": 0.0}, r"duplicate_threshold must be in \(0, 1\), got 0\.0"),
        ({"numeric_threshold": 1.5}, r"numeric_threshold must be in \(0, 1\], got 1\.5"),
        ({"category_threshold": 0.0}, r"category_threshold must be in \(0, 1\], got 0\.0"),
        ({"outlier_factor": -1.0}, r"outlier_factor must be > 0, got -1\.0"),
        ({"sample_size": 0}, r"sample_size must be >= 1, got 0"),
        (
            {"semantic_auto_threshold": 1.7},
            r"semantic_auto_threshold must be in \[0, 1\], got 1\.7",
        ),
        ({"semantic_max_distinct_values": 0}, r"semantic_max_distinct_values must be >= 1"),
    ],
)
def test_out_of_range_thresholds_name_the_option_and_the_range(df, options, pattern):
    raises_twice(ValueError, pattern, lambda: fd.clean(df, verbose=False, **options), frame=df)


def test_missing_thresholds_must_be_ordered_low_medium_high(df):
    message = raises_twice(
        ValueError,
        r"missing thresholds must be ordered: low <= medium <= high",
        lambda: fd.clean(
            df, missing_threshold_low=0.9, missing_threshold_high=0.1, verbose=False
        ),
        frame=df,
    )
    assert "0.9" in message and "0.1" in message


def test_semantic_review_threshold_above_auto_is_rejected(df):
    message = raises_twice(
        ValueError,
        r"semantic_review_threshold must be <= semantic_auto_threshold",
        lambda: fd.clean(
            df,
            semantic_mode="assist",
            semantic_review_threshold=0.99,
            semantic_auto_threshold=0.10,
            verbose=False,
        ),
        frame=df,
    )
    assert "0.99" in message and "0.1" in message


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        ({"missing_threshold_low": "big"}, r"'<' not supported between instances of"),
        ({"duplicate_threshold": None}, r"'<' not supported between instances of"),
    ],
)
def test_non_numeric_thresholds_leak_a_comparison_error(df, options, pattern):
    """S3: a non-numeric threshold produces a raw comparison ``TypeError``.

    ``missing_threshold_low="big"`` should say so by name the way
    ``impute="knn"`` does; instead the caller gets
    ``'<' not supported between instances of 'float' and 'str'``, which names
    neither the parameter nor the expected type.
    """
    raises_twice(TypeError, pattern, lambda: fd.clean(df, verbose=False, **options), frame=df)


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        ({"strategy": "fast"}, r"strategy must be one of .*got 'fast'"),
        ({"impute": "knn"}, r"impute must be one of .*got 'knn'"),
        ({"outliers": "winsorize"}, r"outliers must be one of .*got 'winsorize'"),
        ({"outlier_method": "mad"}, r"outlier_method must be one of .*got 'mad'"),
        ({"outlier_action": "delete"}, r"outlier_action must be one of .*got 'delete'"),
        ({"duplicate_keep": "both"}, r"duplicate_keep must be one of .*got 'both'"),
        ({"string_case": "title"}, r"string_case must be one of .*got 'title'"),
        ({"semantic_mode": "yes"}, r"semantic_mode must be one of .*got 'yes'"),
        (
            {"semantic_privacy_policy": "whatever"},
            r"semantic_privacy_policy must be one of .*got 'whatever'",
        ),
        ({"dayfirst": "maybe"}, r"dayfirst must be True, False, or 'auto', got 'maybe'"),
        (
            {"impute_strategy": {"amount": "knn"}},
            r"impute_strategy values must be one of .*got 'knn'",
        ),
    ],
)
def test_enum_options_name_the_parameter_the_choices_and_the_value(df, options, pattern):
    """The house style set by ``impute="knn"``: parameter, permitted set, value."""
    raises_twice(ValueError, pattern, lambda: fd.clean(df, verbose=False, **options), frame=df)


def test_unknown_option_suggests_the_intended_one(df):
    message = raises_twice(
        TypeError,
        r"unknown option\(s\): 'stratgy' \(did you mean 'strategy'\?\)",
        lambda: fd.clean(df, stratgy="balanced", verbose=False),
        frame=df,
    )
    assert "Valid options:" in message


def test_decimal_and_thousands_separators_must_differ(df):
    raises_twice(
        ValueError,
        r"decimal and thousands separators must differ, both are ','",
        lambda: fd.clean(df, decimal=",", thousands=",", verbose=False),
        frame=df,
    )


def test_config_must_be_a_cleanconfig(df):
    for bad in ([1, 2], "balanced"):
        raises_twice(
            TypeError,
            rf"config must be a CleanConfig, got {type(bad).__name__}",
            lambda b=bad: fd.clean(df, config=b, verbose=False),
            frame=df,
        )


# ── 4. unknown domains, engines and backends ────────────────────────────────────


def test_unknown_domain_lists_the_registered_domains(df):
    message = raises_twice(
        UnknownDomainError,
        r"unknown domain 'pharma'",
        lambda: fd.clean(df, domain="pharma", verbose=False),
        frame=df,
    )
    assert "available domains:" in message
    assert "finance" in message and "healthcare" in message


def test_multi_frame_domain_without_a_file_selector_says_what_to_pass(df):
    raises_twice(
        TypeError,
        r"domain 'transport' requires a feed dict or a single frame with gtfs_file=",
        lambda: fd.clean(df, domain="transport", verbose=False),
        frame=df,
    )


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        ({"column_map": {"a": "b"}}, r"column_map requires a domain= to be set"),
        ({"gtfs_file": "stops.txt"}, r"gtfs_file requires domain='transport'"),
    ],
)
def test_domain_selectors_without_a_domain_are_rejected(df, options, pattern):
    raises_twice(TypeError, pattern, lambda: fd.clean(df, verbose=False, **options), frame=df)


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        ({"engine": "sqlite"}, r"engine must be one of .*got 'sqlite'"),
        ({"output_format": "parquet"}, r"output_format must be one of .*got 'parquet'"),
        (
            {"engine": "polars", "fallback_policy": "boom"},
            r"fallback_policy must be one of \('allow', 'warn', 'error'\), got 'boom'",
        ),
    ],
)
def test_unknown_engine_names_list_the_supported_backends(df, options, pattern):
    raises_twice(ValueError, pattern, lambda: fd.clean(df, verbose=False, **options), frame=df)


def test_fallback_policy_on_the_pandas_engine_explains_the_fix(df):
    message = raises_twice(
        TypeError,
        r"fallback_policy applies to native engines",
        lambda: fd.clean(df, fallback_policy="error", verbose=False),
        frame=df,
    )
    assert "engine='polars'" in message


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        (
            {"contract": {"name": "c", "columns": []}},
            r"contract= is only supported on the in-memory",
        ),
        ({"memory": object()}, r"memory= is only supported on the in-memory pandas engine"),
        ({"profile": "x.fdprofile"}, r"profile= is only supported on the in-memory pandas engine"),
        ({"context": "customer_id is unique."}, r"context=/policy= are only supported on the"),
    ],
)
def test_pandas_only_features_are_refused_on_a_native_engine(df, options, pattern):
    """Unsupported engine + operation combinations fail loudly, not silently."""
    raises_twice(
        TypeError,
        pattern,
        lambda: fd.clean(df, engine="polars", verbose=False, **options),
        frame=df,
    )


def test_unknown_semantic_backend_is_reported_and_is_fatal_under_strict(df):
    _, report = fd.clean(
        df,
        semantic_mode="assist",
        semantic_backends=("telepathy",),
        return_report=True,
        verbose=False,
    )
    assert any("telepathy" in w for w in report.warnings)
    assert any(e["backend"] == "telepathy" for e in report.fallback_events)

    message = raises_twice(
        fd.PolicyError,
        r"unknown semantic backend 'telepathy'",
        lambda: fd.clean(
            df,
            semantic_mode="assist",
            semantic_backends=("telepathy",),
            strict=True,
            verbose=False,
        ),
        frame=df,
    )
    assert "known backends:" in message


def test_unknown_semantic_type_hint_is_accepted_and_overrides_inference(df):
    """S2: a bogus ``semantic_type`` hint wins at confidence 1.0, silently.

    ``semantic_backends`` warns and turns fatal under ``strict``; a ``FieldSpec``
    with an unknown ``semantic_type`` warns; ``config_for_field`` warns. The
    ``semantic_context`` hint path does none of that — a misspelt type is
    adopted verbatim, which quietly disables the checks the real type would
    have brought.
    """
    roles = fd.infer_roles(
        df, semantic_context={"columns": {"customer_id": {"semantic_type": "wingdings"}}}
    )
    row = roles[roles["column"] == "customer_id"].iloc[0]
    assert row["semantic_type"] == "wingdings"
    assert row["semantic_type_confidence"] == 1.0

    # Not even ``strict=True`` turns the unknown type into an error.
    fd.clean(
        df,
        semantic_mode="assist",
        semantic_context={"columns": {"customer_id": {"semantic_type": "wingdings"}}},
        strict=True,
        verbose=False,
    )


def test_semantic_context_hint_for_an_unknown_column_is_silently_dropped(df):
    """S3: an unresolvable hint column is ignored, even under ``strict=True``.

    ``context="Never modify zzz."`` raises ``PolicyError`` under strict for the
    same unresolved reference, so the two hint channels disagree.
    """
    _, report = fd.clean(
        df,
        semantic_mode="assist",
        semantic_context={"columns": {"zzz_not_a_column": {"semantic_type": "email"}}},
        strict=True,
        return_report=True,
        verbose=False,
    )
    assert not [w for w in report.warnings if "zzz_not_a_column" in w]


def test_unknown_field_type_warns_when_it_would_lose_protection(df):
    """The behaviour the ``semantic_context`` path is missing, for contrast."""
    with pytest.warns(UserWarning, match=r"unknown semantic_type 'wingdings'"):
        fd.clean_text(
            df,
            columns=["notes"],
            config=fd.TextCleanConfig(case="lower", remove_punctuation=True),
            field_types={"notes": "wingdings"},
        )


# ── 5. contradictory configuration ──────────────────────────────────────────────


def test_context_and_policy_together_are_rejected_with_the_remedy(df):
    policy = fd.compile_context("customer_id is unique.", df=df)
    message = raises_twice(
        TypeError,
        r"context= and policy= are mutually exclusive",
        lambda: fd.clean(df, context="customer_id is unique.", policy=policy, verbose=False),
        frame=df,
    )
    assert "ContextPolicy" in message


def test_validate_needs_exactly_one_of_suite_or_context(df):
    raises_twice(
        TypeError,
        r"fd\.validate needs suite= \(a ValidationSuite\), context= \(rules text\) or policy=",
        lambda: fd.validate(df),
        frame=df,
    )
    raises_twice(
        TypeError,
        r"fd\.validate takes either suite= or context=/policy=, not both",
        lambda: fd.validate(df, suite="s.json", context="customer_id is unique."),
        frame=df,
    )


def test_source_provenance_requires_a_report(df):
    raises_twice(
        ValueError,
        r"source_provenance requires return_report=True",
        lambda: fd.clean(df, source_provenance={"source": "crm"}, verbose=False),
        frame=df,
    )


def test_impute_alias_conflict_is_rejected(df):
    raises_twice(
        TypeError,
        r"impute_method= conflicts with impute=; pass only one imputation option",
        lambda: fd.clean(df, impute="mean", impute_method="median", verbose=False),
        frame=df,
    )


@pytest.mark.parametrize(
    ("options", "pattern"),
    [
        ({"policy": "never modify amount"}, r"policy must be a freshdata\.ContextPolicy"),
        ({"context": 123}, r"context must be a string, got int"),
        ({"strict": 1}, r"strict must be a bool, got 1"),
        ({"progress_callback": "nope"}, r"progress_callback must be callable"),
        ({"domain_sensitive_names": "yes"}, r"domain_sensitive_names must be a bool"),
        (
            {"impute_strategy": [("amount", "mean")]},
            r"impute_strategy must be a mapping of column name to strategy",
        ),
    ],
)
def test_wrongly_typed_options_name_the_parameter_and_the_expected_type(df, options, pattern):
    raises_twice(TypeError, pattern, lambda: fd.clean(df, verbose=False, **options), frame=df)


def test_strict_context_turns_an_unresolved_column_into_an_error(df):
    message = raises_twice(
        fd.PolicyError,
        r"strict context compile failed",
        lambda: fd.clean(df, context="Never modify zzz.", strict=True, verbose=False),
        frame=df,
    )
    assert "unresolved column reference 'zzz'" in message


def test_strict_context_turns_an_unparsable_sentence_into_an_error(df):
    message = raises_twice(
        fd.PolicyError,
        r"strict context compile failed",
        lambda: fd.clean(df, context="Blorble the wumpus quickly.", strict=True, verbose=False),
        frame=df,
    )
    assert "unparsed_sentence" in message


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"on_missing": "explode"}, r"on_missing must be fail\|warn\|ignore, got 'explode'"),
        (
            {"on_unexpected": "explode"},
            r"on_unexpected must be fail\|warn\|preserve, got 'explode'",
        ),
    ],
)
def test_contract_gate_policies_name_their_choices(df, kwargs, pattern):
    contract = {"name": "c", "columns": [{"name": "customer_id", "dtype": "object"}]}
    raises_twice(
        ValueError,
        pattern,
        lambda: fd.clean(df, contract=contract, verbose=False, **kwargs),
        frame=df,
    )
    raises_twice(
        ValueError,
        pattern,
        lambda: fd.suggest_plan(df, contract=contract, **kwargs),
        frame=df,
    )


# ── 6. plans ────────────────────────────────────────────────────────────────────


def test_apply_plan_rejects_a_non_plan_by_type(df):
    message = raises_twice(
        TypeError,
        r"plan must be a RepairPlan or CleanPlan, got str",
        lambda: fd.apply_plan(df, "not-a-plan"),
        frame=df,
    )
    assert "RepairPlan" in message


def test_apply_plan_explains_a_cleanplan_without_planned_actions(df):
    plan = fd.suggest_plan(df)
    assert plan.repair_plan is None
    message = raises_twice(
        TypeError,
        r"this CleanPlan has no repair_plan",
        lambda: fd.apply_plan(df, plan),
        frame=df,
    )
    assert "semantic_mode=" in message and "context=" in message


def test_apply_plan_detects_drift_against_a_changed_frame(df):
    plan = fd.suggest_plan(df, context="Never modify notes.").repair_plan
    assert plan is not None
    drifted = df.assign(amount=["1", "2", "3", "4", "5", "6"])
    message = raises_twice(
        fd.PlanDriftError,
        r"allow_drift",
        lambda: fd.apply_plan(drifted, plan),
        frame=drifted,
    )
    assert "signature" in message.lower() or "changed" in message.lower()


def test_plan_accepts_any_engine_name_and_answers_confidently(df):
    """S2: ``fd.plan(engine=...)`` never validates the backend name.

    ``fd.clean(df, engine="sqlite")`` raises ``ValueError`` listing the six
    supported engines. ``fd.plan`` — documented as the front door that answers
    "which engine would run this, and would it fall back?" *before* any data is
    touched — takes the same string, echoes it back as ``plan.backend`` and
    attaches a plausible ``fallback_reason``. A typo therefore produces a
    confident, wrong execution verdict instead of an error.
    """
    for engine in ("sqlite", "telepathy", "duckdd"):
        plan = fd.plan(df, engine=engine)
        assert plan.backend == engine
        assert plan.fallback_reason  # reads as a real verdict about a real backend

    # The same name on the execution front door is rejected.
    raises_twice(
        ValueError,
        r"engine must be one of .*got 'sqlite'",
        lambda: fd.clean(df, engine="sqlite", verbose=False),
        frame=df,
    )


# ── 7. validation suites ────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [None, "suite", 42])
def test_run_suite_leaks_attributeerror_for_a_non_suite(df, bad):
    """S3: ``run_suite`` never type-checks ``suite``.

    ``fd.validate(df, suite=...)`` does (``suite must be a ValidationSuite or a
    path to one, got dict``), so the direct entry point is the weaker of the
    two front doors onto the same engine.
    """
    raises_twice(
        AttributeError, r"has no attribute 'to_contract'", lambda: run_suite(df, bad), frame=df
    )


def test_validate_rejects_a_non_suite_by_type(df):
    raises_twice(
        TypeError,
        r"suite must be a ValidationSuite or a path to one, got dict",
        lambda: fd.validate(df, suite={"name": "s"}),
        frame=df,
    )


def test_validation_suite_rejects_a_non_rule(df):
    raises_twice(
        TypeError,
        r"rules must be ColumnRule/ColumnContract, got str",
        lambda: ValidationSuite(name="s", rules="customer_id"),
    )


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"op": "=~"}, r"op must be one of .*got '=~'"),
        ({"op": "<", "mostly": 2.0}, r"mostly must be in \(0, 1\], got 2\.0"),
    ],
)
def test_cross_column_rule_validates_its_own_arguments(kwargs, pattern):
    kwargs = {"left": "a", "op": "<", "right": "b", **kwargs}
    raises_twice(ValueError, pattern, lambda: CrossColumnRule(**kwargs))


# ── 8. corrupted files: policies, memories, suites, profiles ────────────────────


def test_corrupted_policy_json_raises_a_json_error(tmp_path):
    path = tmp_path / "policy.json"
    path.write_text("{not json at all", encoding="utf-8")
    raises_twice(
        json.JSONDecodeError, r"Expecting property name", lambda: fd.ContextPolicy.from_json(path)
    )


def test_a_policy_json_of_the_wrong_shape_loads_as_an_empty_policy(tmp_path):
    """S3: a structurally wrong policy file is accepted and governs nothing.

    ``from_dict`` reads only the keys it knows, so a file that is valid JSON
    but not a freshdata policy yields a policy with no constraints and no
    protected columns — ``fd.clean(df, policy=...)`` then silently cleans
    without any of the rules the caller believed they had saved.
    """
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"rules": "never modify amount"}), encoding="utf-8")
    policy = fd.ContextPolicy.from_json(path)
    assert policy.constraints == ()
    assert policy.protected_columns == ()


def test_missing_policy_file_raises_filenotfound(tmp_path):
    raises_twice(
        FileNotFoundError,
        r"No such file or directory",
        lambda: fd.ContextPolicy.from_json(tmp_path / "absent.json"),
    )


def test_corrupted_memory_file_raises_a_json_error(tmp_path):
    path = tmp_path / "memory.fdmem"
    path.write_text("{oops", encoding="utf-8")
    raises_twice(
        json.JSONDecodeError, r"Expecting property name", lambda: fd.load_cleaning_memory(path)
    )


def test_a_memory_file_of_the_wrong_shape_loads_as_an_empty_memory(tmp_path):
    """S3: same silent-empty hazard as the policy loader, with higher stakes.

    A truncated or foreign ``.fdmem`` produces a ``CleaningMemory`` with no
    dataset id and no accepted decisions; ``fd.clean(df, memory=...)`` then
    replays nothing and reports no problem.
    """
    path = tmp_path / "memory.fdmem"
    path.write_text(json.dumps({"version": 99, "entries": "nope"}), encoding="utf-8")
    memory = fd.load_cleaning_memory(path)
    assert memory.dataset_id == ""
    assert not memory.accepted


def test_missing_memory_file_names_the_path(tmp_path):
    message = raises_twice(
        FileNotFoundError,
        r"cleaning memory not found",
        lambda: fd.load_cleaning_memory(tmp_path / "absent.fdmem"),
    )
    assert "absent.fdmem" in message


def test_memory_must_be_a_cleaningmemory_not_a_path(df):
    """``memory=`` takes the object; a path is a common and rejected mistake."""
    for bad in ("memory.fdmem", {"accepted": []}, 7):
        message = raises_twice(
            TypeError,
            r"memory= must be a CleaningMemory",
            lambda b=bad: fd.clean(df, memory=b, verbose=False),
            frame=df,
        )
        assert "fd.learn_cleaning_memory" in message


def test_corrupted_suite_json_raises_a_json_error(df, tmp_path):
    path = tmp_path / "suite.json"
    path.write_text("<<<not json", encoding="utf-8")
    raises_twice(
        json.JSONDecodeError, r"Expecting value", lambda: fd.validate(df, suite=path), frame=df
    )


def test_a_suite_json_missing_its_name_raises_a_bare_keyerror(df, tmp_path):
    """S3: ``KeyError: 'name'`` names neither the file nor the required keys."""
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    raises_twice(KeyError, r"'name'", lambda: fd.validate(df, suite=path), frame=df)


def test_corrupted_profile_archive_names_the_file_and_the_format(df, tmp_path):
    path = tmp_path / "bad.fdprofile"
    path.write_text("{broken", encoding="utf-8")
    message = raises_twice(
        ProfileFormatError,
        r"is not a valid \.fdprofile archive",
        lambda: fd.clean(df, profile=path, verbose=False),
        frame=df,
    )
    assert "bad.fdprofile" in message


def test_missing_profile_file_names_the_path(df, tmp_path):
    raises_twice(
        ProfileFormatError,
        r"profile not found",
        lambda: fd.clean(df, profile=tmp_path / "absent.fdprofile", verbose=False),
        frame=df,
    )


def test_profile_must_be_a_profile_or_a_path(df):
    message = raises_twice(
        TypeError,
        r"profile= must be a LearningProfile or a path to a \.fdprofile \(got int\)",
        lambda: fd.clean(df, profile=42, verbose=False),
        frame=df,
    )
    assert ".fdprofile" in message


# ── 9. file front doors ─────────────────────────────────────────────────────────


def test_clean_csv_missing_file_raises_filenotfound(tmp_path):
    raises_twice(
        FileNotFoundError,
        r"No such file or directory",
        lambda: fd.clean_csv(tmp_path / "absent.csv"),
    )


def test_clean_excel_missing_file_raises_filenotfound(tmp_path):
    raises_twice(
        FileNotFoundError,
        r"No such file or directory",
        lambda: fd.clean_excel(tmp_path / "absent.xlsx"),
    )


def test_clean_csv_given_a_dataframe_fails_with_an_internal_message(df):
    """S3: passing a frame to the path-taking front door is unreadable.

    ``fd.clean_csv(df)`` is a natural slip (``fd.clean`` takes a frame) and the
    caller gets ``argument of type 'method' is not iterable`` from the
    leading-zero pre-scan, which mentions neither ``path`` nor what it wanted.
    """
    raises_twice(TypeError, r"argument of type 'method' is not iterable", lambda: fd.clean_csv(df))


def test_clean_excel_refuses_a_multi_sheet_selection(tmp_path, df):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, sheet_name="one", index=False)
        df.to_excel(writer, sheet_name="two", index=False)
    message = raises_twice(
        TypeError,
        r"clean_excel cleans a single sheet",
        lambda: fd.clean_excel(path, read_excel_kwargs={"sheet_name": None}),
    )
    assert "sheet_name" in message


def test_clean_csv_propagates_option_validation_before_reading(tmp_path):
    path = tmp_path / "in.csv"
    path.write_text("a,b\n1,x\n2,y\n", encoding="utf-8")
    raises_twice(
        ValueError,
        r"impute must be one of .*got 'knn'",
        lambda: fd.clean_csv(path, impute="knn", verbose=False),
    )


def test_clean_with_an_unsupported_file_path_names_the_path(tmp_path):
    path = tmp_path / "data.weird"
    path.write_text("nothing", encoding="utf-8")
    message = raises_twice(
        ValueError, r"unsupported file type for path", lambda: fd.clean(str(path))
    )
    assert "data.weird" in message


# ── 10. text and field validation ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    [
        ({"case": "Sentence"}, r"unsupported case: 'Sentence'"),
        ({"unicode_form": "NFZ"}, r"unsupported unicode_form: 'NFZ'"),
    ],
)
def test_textcleanconfig_rejects_unknown_choices(kwargs, pattern):
    raises_twice(ValueError, pattern, lambda: fd.TextCleanConfig(**kwargs))


def test_clean_text_with_a_mapping_instead_of_a_config_leaks_attributeerror(df):
    """S3: ``config=`` is not type-checked, unlike ``fd.clean``'s ``config=``."""
    raises_twice(
        AttributeError,
        r"'dict' object has no attribute 'unicode_form'",
        lambda: fd.clean_text(df, config={"case": "lower"}),
        frame=df,
    )


def test_validate_fields_with_a_mapping_instead_of_a_policy_leaks_attributeerror(df):
    """S3: ``policy=`` is not type-checked either."""
    raises_twice(
        AttributeError,
        r"'dict' object has no attribute 'normalize_text'",
        lambda: fd.validate_fields(df, {"notes": "free_text"}, policy={"x": 1}),
        frame=df,
    )


def test_validate_fields_warns_about_an_unknown_semantic_type(df):
    with pytest.warns(UserWarning, match=r"unknown semantic_type 'wingdings'"):
        fd.validate_fields(df, {"notes": "wingdings"})


def test_validate_fields_leaves_the_input_frame_untouched(df):
    before = frame_digest(df)
    fd.validate_fields(df, {"customer_id": "identifier"})
    assert frame_digest(df) == before


def test_validate_fields_does_not_range_check_its_own_thresholds():
    """S3: ``rare_threshold`` and ``outlier_fence`` accept impossible values.

    ``CleanConfig`` rejects every analogous knob by name (``outlier_factor must
    be > 0, got -1.0``). ``fd.validate_fields`` validates neither of its two, so
    a negative Tukey fence silently reclassifies *every* value as a
    ``statistical_outlier`` and a frequency threshold above 1.0 is accepted.
    """
    frame = pd.DataFrame(
        {"amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1000.0]}
    )
    schema = {"amount": "numeric"}

    sane = fd.validate_fields(frame, schema)
    assert len(sane.issues) == 1

    # A negative fence is nonsense; it is accepted and flags the whole column.
    negative_fence = fd.validate_fields(frame, schema, outlier_fence=-3.0)
    assert len(negative_fence.issues) == len(frame)
    assert {i.classification for i in negative_fence.issues} == {"statistical_outlier"}

    # A "frequency" above 1.0 is impossible; also accepted without comment.
    assert fd.validate_fields(frame, schema, rare_threshold=5.0) is not None


def test_a_suite_rule_for_an_absent_column_reports_instead_of_raising(df):
    """Validation reports; only ``raise_if_failed`` converts that into an error."""
    suite = ValidationSuite(name="s", rules=(ColumnRule(name="zzz", nullable=False),))
    before = frame_digest(df)
    result = run_suite(df, suite)
    assert frame_digest(df) == before, "validation must be read-only"
    assert not result.passed
    assert result.n_errors == 1
    assert "required column 'zzz' is missing" in result.findings[0].message

    message = raises_twice(
        ValidationError,
        r"validation suite 's' failed with 1 error\(s\)",
        result.raise_if_failed,
        frame=df,
    )
    assert "zzz" in message


def test_non_pandas_input_to_a_suite_records_the_materialization(df):
    """A silent materialization would itself be a finding; it is recorded."""
    polars = pytest.importorskip("polars")
    suite = ValidationSuite(name="s", rules=(ColumnRule(name="customer_id", nullable=False),))
    result = run_suite(polars.from_pandas(df), suite)
    assert result.execution["backend"] == "pandas"
    assert result.execution["fallback"], "materializing a polars frame must not be silent"
    assert "materialized" in result.execution["fallback"][0]["reason"]
