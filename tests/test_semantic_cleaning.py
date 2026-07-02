"""Tests for the Semantic Cleaning Layer.

Small, deterministic pandas frames exercise each expert, the policy gate across
modes, column protection, config validation, reporting, preview, the engine
hand-off, and the native-engine fallback.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.experts import VALUE_EXPERTS

COMMON = {"return_report": True, "verbose": False}


def sem(report) -> list:
    return [a for a in report if a.step == "semantic"]


def applied(report) -> list:
    return [a for a in sem(report) if a.status == "automatic"]


def suggested(report) -> list:
    return [a for a in sem(report) if a.status == "suggested"]


def skipped(report) -> list:
    return [a for a in sem(report) if a.status == "skipped"]


def spelled_frame() -> pd.DataFrame:
    # Mixed so fix_dtypes can't pre-convert; the spelled word survives to semantic.
    return pd.DataFrame({"age": ["twenty", "30", "40", "35", "twenty", "40", "25", "19"]})


def kitchen_sink() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": ["twenty", "30", "40", "35", "twenty", "40", "25", "19"],
            "active": ["yes", "no", "y", "n", "unknown", "yes", "no", "y"],
            "price": ["$1,200.50", "$300", "free", "$50", "$300", "$25", "$300", "$50"],
            "weight": ["10 kg", "5 kg", "10 kg", "7 kg", "5 kg", "12 kg", "9 kg", "3 kg"],
            "gender": ["M", "female", "M", "F", "male", "Female ", "M", "f"],
            "customer_id": ["007", "00123", "105A", "A-100", "008", "00999", "010", "011"],
        }
    )


# 1. disabled by default ----------------------------------------------------- #
def test_disabled_by_default_keeps_behavior():
    df = kitchen_sink()
    base, base_report = fd.clean(df, **COMMON)
    assert sem(base_report) == []
    for mode in (None, "off"):
        out, report = fd.clean(df, semantic_mode=mode, **COMMON)
        assert sem(report) == []
        pd.testing.assert_frame_equal(out, base)


# 2. spelled number in numeric-like column ----------------------------------- #
def test_spelled_number_converts_in_numeric_column():
    out, report = fd.clean(spelled_frame(), semantic_mode="auto", **COMMON)
    assert pd.api.types.is_numeric_dtype(out["age"])
    assert 20 in list(out["age"])
    assert any(a.model_id == "semantic:spelled_number:v1" for a in applied(report))


# 3. spelled number NOT converted in free-text column ------------------------ #
def test_spelled_number_skipped_in_free_text():
    df = pd.DataFrame(
        {
            "notes": [
                "Room One please and thank you very much indeed",
                "another long free form note describing the situation",
                "yet more descriptive prose that is clearly free text",
                "Grade A material was used here according to the spec",
            ]
        }
    )
    out, report = fd.clean(df, semantic_mode="auto", **COMMON)
    assert list(out["notes"]) == list(df["notes"])
    assert sem(report) == []


# 4. boolean synonym normalization ------------------------------------------- #
def test_boolean_synonyms_normalized():
    out, report = fd.clean(kitchen_sink(), semantic_mode="auto", **COMMON)
    assert out["active"].tolist()[:4] == [True, False, True, False]
    assert "unknown" in out["active"].tolist()  # unparseable token left alone
    assert any(a.model_id == "semantic:boolean_synonym:v1" for a in applied(report))


# 5. category synonym proposal ----------------------------------------------- #
def test_category_synonym_proposed():
    _, report = fd.clean(kitchen_sink(), semantic_mode="assist", **COMMON)
    cats = [a for a in sem(report) if a.model_id == "semantic:category_synonym:v1"]
    assert cats
    assert all(a.status == "suggested" for a in cats)


# 6. currency string conversion ---------------------------------------------- #
def test_currency_string_converted():
    out, report = fd.clean(kitchen_sink(), semantic_mode="auto", **COMMON)
    assert 1200.5 in list(out["price"])
    assert "free" in list(out["price"])  # non-currency value untouched
    assert any(a.model_id == "semantic:currency_string:v1" for a in applied(report))


# 7. unit suffix conversion -------------------------------------------------- #
def test_unit_suffix_converted():
    out, report = fd.clean(kitchen_sink(), semantic_mode="auto", **COMMON)
    assert 10 in list(out["weight"])
    assert any(a.model_id == "semantic:unit_suffix:v1" for a in applied(report))


# 8. ID columns are protected ------------------------------------------------ #
def test_id_columns_protected():
    out, report = fd.clean(kitchen_sink(), semantic_mode="auto", **COMMON)
    assert list(out["customer_id"]) == list(kitchen_sink()["customer_id"])
    assert any(a.column == "customer_id" for a in skipped(report))


def test_explicit_id_column_protected():
    df = spelled_frame().rename(columns={"age": "account"})
    out, report = fd.clean(df, semantic_mode="auto", id_columns=("account",), **COMMON)
    assert "twenty" in list(out["account"])  # unchanged
    assert any(a.column == "account" and a.status == "skipped" for a in sem(report))


# 9. preserve_columns are protected ------------------------------------------ #
def test_preserve_columns_protected():
    out, report = fd.clean(
        spelled_frame(), semantic_mode="auto", preserve_columns=("age",), **COMMON
    )
    assert "twenty" in list(out["age"])
    assert any(a.column == "age" and a.status == "skipped" for a in sem(report))


# 10. target_column is protected --------------------------------------------- #
def test_target_column_protected():
    out, report = fd.clean(
        spelled_frame(), semantic_mode="auto", target_column="age", **COMMON
    )
    assert "twenty" in list(out["age"])
    assert any(a.column == "age" and a.status == "skipped" for a in sem(report))


# 11. assist records suggestions but does not mutate ------------------------- #
def test_assist_records_without_mutating():
    df = kitchen_sink()
    base, _ = fd.clean(df, **COMMON)
    out, report = fd.clean(df, semantic_mode="assist", **COMMON)
    assert suggested(report)
    assert applied(report) == []
    pd.testing.assert_frame_equal(out, base)


# 12. review mode applies only safe deterministic proposals ------------------ #
def test_review_applies_only_low_risk():
    out, report = fd.clean(kitchen_sink(), semantic_mode="review", **COMMON)
    assert all(a.risk == "low" for a in applied(report))
    # category (medium-risk) is suggested, not applied -> gender unchanged.
    assert "M" in list(out["gender"])
    assert any(a.model_id == "semantic:category_synonym:v1" for a in suggested(report))


# 13. auto mode applies high-confidence low-risk proposals ------------------- #
def test_auto_applies_high_confidence():
    out, report = fd.clean(kitchen_sink(), semantic_mode="auto", **COMMON)
    assert applied(report)
    assert "male" in list(out["gender"]) and "M" not in list(out["gender"])


# 14. low-confidence proposals are suggested or skipped ---------------------- #
def test_low_confidence_not_applied():
    out, report = fd.clean(
        spelled_frame(),
        semantic_mode="auto",
        semantic_review_threshold=0.99,
        semantic_auto_threshold=0.99,
        **COMMON,
    )
    assert applied(report) == []
    assert skipped(report)
    assert "twenty" in list(out["age"])  # nothing mutated


# 15. report carries the full audit fields ----------------------------------- #
def test_report_fields_present():
    _, report = fd.clean(kitchen_sink(), semantic_mode="auto", **COMMON)
    actions = sem(report)
    assert actions
    for a in actions:
        assert a.step == "semantic"
        assert 0.0 <= a.confidence <= 1.0
        assert a.risk in ("low", "medium", "high")
        assert a.status in ("automatic", "suggested", "skipped", "approved")
        assert a.model_id.startswith("semantic:")
        assert isinstance(a.human_review, bool)
        assert a.rationale
    # serialization carries the new fields too.
    payload = report.to_dict()["actions"]
    assert any("status" in a and "human_review" in a for a in payload)


# 16. return_report surfaces semantic actions -------------------------------- #
def test_return_report_includes_semantic():
    result = fd.clean(kitchen_sink(), semantic_mode="assist", return_report=True, verbose=False)
    assert isinstance(result, tuple)
    _, report = result
    assert sem(report)


# 17. invalid config values raise clean errors ------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"semantic_mode": "bogus"},
        {"semantic_mode": "auto", "semantic_auto_threshold": 1.5},
        {"semantic_mode": "auto", "semantic_review_threshold": -0.1},
        {"semantic_review_threshold": 0.9, "semantic_auto_threshold": 0.5},
        {"semantic_mode": "auto", "semantic_max_distinct_values": 0},
        {"semantic_mode": "auto", "semantic_sample_size": 0},
        {"semantic_mode": "auto", "semantic_privacy_policy": "nope"},
    ],
)
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        fd.clean(spelled_frame(), **kwargs, **COMMON)


# 18. semantic interacts safely with the missing/outlier engine -------------- #
def test_engine_imputes_after_semantic():
    # >= 30 rows and a second column so the engine actually imputes the NaN
    # (rather than the row being dropped as structurally empty).
    words = ["twenty", "30", "40", "35", "25", "19", "22", "31"]
    age = [words[i % len(words)] for i in range(40)]
    age[5] = None
    age[17] = None
    df = pd.DataFrame({"age": age, "other": list(range(100, 140))})
    out, report = fd.clean(df, semantic_mode="auto", strategy="balanced", **COMMON)
    assert pd.api.types.is_numeric_dtype(out["age"])
    assert int(out["age"].isna().sum()) == 0
    assert applied(report)  # semantic ran
    assert any(a.step == "missing" for a in report)  # engine ran after


# 19. semantic cleaning is deterministic and repeatable ---------------------- #
def test_deterministic_and_repeatable():
    df = kitchen_sink()
    out1, r1 = fd.clean(df, semantic_mode="auto", **COMMON)
    out2, r2 = fd.clean(df, semantic_mode="auto", **COMMON)
    pd.testing.assert_frame_equal(out1, out2)
    assert r1.to_dict()["actions"] == r2.to_dict()["actions"]


# 20. native engine path is handled explicitly ------------------------------- #
def test_native_engine_falls_back_with_event():
    df = kitchen_sink()
    out, report = fd.clean(df, engine="polars", semantic_mode="assist", **COMMON)
    assert report.backend == "pandas"
    assert any(e["fallback_step"] == "semantic" for e in report.fallback_events)
    assert sem(report)


# Context hints: allowed_values mapping and mutable override ----------------- #
def test_allowed_values_mapping():
    # allowed_values canonicalizes case/whitespace variants of allowed entries.
    df = pd.DataFrame(
        {"country": ["united states", "UNITED STATES", "United States", "india", "INDIA", "India"]}
    )
    ctx = {"columns": {"country": {"allowed_values": ["United States", "India"]}}}
    out, report = fd.clean(df, semantic_mode="auto", semantic_context=ctx, **COMMON)
    assert set(out["country"]) == {"United States", "India"}
    assert applied(report)


def test_mutable_override_allows_identifier_repair():
    df = pd.DataFrame({"order_id": ["twenty", "30", "40", "35", "twenty", "40", "25", "19"]})
    ctx = {"columns": {"order_id": {"semantic_type": "number", "mutable": True}}}
    out, _ = fd.clean(df, semantic_mode="auto", semantic_context=ctx, **COMMON)
    assert pd.api.types.is_numeric_dtype(out["order_id"])


# Preview: suggest_plan surfaces semantic proposal counts -------------------- #
def test_suggest_plan_shows_semantic_proposals():
    plan = fd.suggest_plan(kitchen_sink(), semantic_mode="assist")
    frame = plan.to_frame()
    assert "semantic_proposals" in frame.columns
    assert frame["semantic_proposals"].sum() > 0
    assert any(
        v["semantic_proposals"] > 0 for v in plan.to_dict()["columns"].values()
    )


def test_suggest_plan_semantic_under_conservative():
    # engine off (conservative) but semantic still previewed.
    plan = fd.suggest_plan(spelled_frame(), strategy="conservative", semantic_mode="assist")
    assert plan.column_plans["age"].semantic_proposals > 0


# --------------------------------------------------------------------------- #
# Unit tests for the semantic internals (parsers, types, profiler, policy).
# --------------------------------------------------------------------------- #
from freshdata.config import CleanConfig  # noqa: E402
from freshdata.semantic.context import build_semantic_context  # noqa: E402
from freshdata.semantic.experts import (  # noqa: E402
    _value_counts,
    column_name_is_identifier,
    is_plain_number,
    looks_like_identifier_value,
    parse_boolean,
    parse_currency,
    parse_number_words,
    parse_unit,
)
from freshdata.semantic.policy import decide  # noqa: E402
from freshdata.semantic.profiler import plan_semantic, profile_semantic_issues  # noqa: E402
from freshdata.semantic.scoring import confidence_from_evidence, risk_for  # noqa: E402
from freshdata.semantic.types import (  # noqa: E402
    SemanticEvidence,
    SemanticProposal,
    SemanticProposalSet,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("zero", 0),
        ("one", 1),
        ("twenty", 20),
        ("thirty five", 35),
        ("thirty-five", 35),
        ("one hundred", 100),
        ("two thousand five hundred", 2500),
        ("ninety", 90),
        ("one and", 1),
        ("room", None),
        ("oneplus", None),
        ("", None),
        ("and", None),
    ],
)
def test_parse_number_words(text, expected):
    assert parse_number_words(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$1,200.50", 1200.5),
        ("₹2,000", 2000.0),
        ("EUR 10.5", 10.5),
        ("1200", None),  # no currency marker
        ("$", None),
        ("free", None),
        ("$1.2.3", None),  # malformed -> ValueError path
        ("USD .", None),  # strips to "." -> rejected
    ],
)
def test_parse_currency(text, expected):
    assert parse_currency(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("10 kg", (10.0, "kg")), ("5 km", (5.0, "km")), ("42 ms", (42.0, "ms")), ("abc", None)],
)
def test_parse_unit(text, expected):
    assert parse_unit(text) == expected


def test_parse_boolean_and_plain_number():
    assert parse_boolean("yes") is True
    assert parse_boolean("n") is False
    assert parse_boolean("maybe") is None
    assert is_plain_number("1,200") and is_plain_number(3) and is_plain_number(2.5)
    assert not is_plain_number("abc") and not is_plain_number(True) and not is_plain_number("")


@pytest.mark.parametrize(
    "value,flagged",
    [
        ("007", True),
        ("00123", True),
        ("105A", True),
        ("A-100", True),
        ("D@vid", True),
        ("twenty", False),
        ("male", False),
        ("   ", False),  # empty after strip
        (123, False),
    ],
)
def test_identifier_value_guard(value, flagged):
    assert looks_like_identifier_value(value) is flagged


def test_identifier_name_guard():
    assert column_name_is_identifier("customer_id")
    assert column_name_is_identifier("sku")
    assert not column_name_is_identifier("age")


def test_value_counts_does_not_raise_on_unhashable():
    s = pd.Series([{1, 2}, {3, 4}])  # sets are unhashable
    result = _value_counts(s)
    assert hasattr(result, "empty")  # returns a Series either way, never raises


def test_proposal_set_helpers():
    ev = (SemanticEvidence("pattern", "x", 0.0),)
    p1 = SemanticProposal("a", "x", 1, "spelled_number", "e", 0.9, "low", "r", ev, count=2)
    p2 = SemanticProposal("a", "y", 2, "spelled_number", "e", 0.9, "low", "r", ev, count=1)
    p3 = SemanticProposal("b", "z", 3, "spelled_number", "e", 0.9, "low", "r", ev, count=4)
    ps = SemanticProposalSet()
    ps.add(p1)
    ps.extend([p2, p3])
    assert len(ps) == 3
    assert set(ps.by_column()) == {"a", "b"}
    assert ps.counts_by_column() == {"a": 3, "b": 4}


def test_scoring_helpers():
    assert risk_for("unsafe_ambiguous", 0.99) == "high"
    assert risk_for("spelled_number", 0.5) == "high"
    assert risk_for("spelled_number", 0.95) == "low"
    assert confidence_from_evidence(0.9, (SemanticEvidence("k", "d", 0.5),)) == 0.999  # capped


def test_profile_semantic_issues_public():
    cfg = CleanConfig(semantic_mode="assist", verbose=False)
    df = spelled_frame()
    ctx = build_semantic_context(df, cfg)
    issues = profile_semantic_issues(df, ctx)
    assert any(i.issue_type == "spelled_number" for i in issues)
    assert plan_semantic(df, cfg).proposals  # convenience wrapper


def test_decide_off_mode_skips():
    cfg = CleanConfig(semantic_mode="assist", verbose=False)
    df = spelled_frame()
    proposals = list(plan_semantic(df, cfg))
    target = next(p for p in proposals if p.issue_type == "spelled_number")
    off_ctx = build_semantic_context(df, CleanConfig(verbose=False))  # mode="off"
    d = decide(target, cfg, off_ctx)
    assert d.action == "skip"


def test_sampling_path_with_small_sample_size():
    # semantic_sample_size smaller than the row count exercises the sampling path.
    _, report = fd.clean(spelled_frame(), semantic_mode="assist", semantic_sample_size=3, **COMMON)
    assert any(a.model_id == "semantic:spelled_number:v1" for a in sem(report))


def test_code_value_skipped_in_numeric_column():
    df = pd.DataFrame({"qty": ["twenty", "30", "105A", "40", "35", "twenty", "25", "19"]})
    out, _ = fd.clean(df, semantic_mode="auto", **COMMON)
    assert "105A" in list(out["qty"])  # code-like value left intact


def test_unit_mismatch_value_skipped():
    readings = ["10 kg", "5 kg", "3 lb", "7 kg", "9 kg", "2 kg", "6 kg", "8 kg"]
    _, report = fd.clean(pd.DataFrame({"reading": readings}), semantic_mode="assist", **COMMON)
    units = [a for a in sem(report) if a.model_id == "semantic:unit_suffix:v1"]
    assert units  # kg values proposed
    assert all("lb" not in a.rationale for a in units)  # the lb value was skipped


def test_category_case_whitespace_canonicalization():
    df = pd.DataFrame({"tag": ["usa", "USA", "Usa", "usa", "USA", "usa", "Usa", "usa"]})
    _, report = fd.clean(df, semantic_mode="assist", **COMMON)
    cats = [a for a in sem(report) if a.model_id == "semantic:category_synonym:v1"]
    assert cats  # case variants fold to the dominant canonical


def test_mutable_id_column_allowed():
    df = pd.DataFrame({"order_id": ["twenty", "30", "40", "35", "25", "19", "22", "31"]})
    ctx = {"columns": {"order_id": {"semantic_type": "number", "mutable": True}}}
    out, _ = fd.clean(
        df, semantic_mode="auto", id_columns=("order_id",), semantic_context=ctx, **COMMON
    )
    assert pd.api.types.is_numeric_dtype(out["order_id"])


def test_malformed_semantic_context_ignored():
    # Non-mapping columns value is ignored rather than raising.
    df = spelled_frame()
    out, report = fd.clean(
        df, semantic_mode="assist", semantic_context={"dataset": "d", "columns": "oops"}, **COMMON
    )
    assert sem(report)  # still profiles using inferred roles


def test_boolean_downcast_to_bool_dtype():
    # fix_dtypes off, so a clean yes/no column reaches semantic as strings and is
    # fully normalized -> exercises the all-bool downcast path.
    df = pd.DataFrame({"flag": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"]})
    out, _ = fd.clean(df, semantic_mode="auto", fix_dtypes=False, **COMMON)
    assert out["flag"].dtype == bool


# 25. DatePhraseExpert -------------------------------------------------------- #
# fix_dtypes off throughout, exactly like spelled_frame()'s note: otherwise the
# core dtype-repair step (a pre-existing, unrelated engine feature) can already
# parse an obviously date-shaped column before the semantic layer ever sees it.
DATE_COMMON = {**COMMON, "fix_dtypes": False}


def date_frame(values: list[str]) -> pd.DataFrame:
    # A companion "id" column keeps literal duplicate strings from collapsing
    # into one row via drop_duplicates, so repeated raw values still reach the
    # semantic layer (and so count aggregation is exercised).
    return pd.DataFrame({"id": range(len(values)), "signup_date": values})


def test_date_phrase_expert_registered():
    assert any(e.issue_type == "date_phrase" for e in VALUE_EXPERTS)


def test_iso_date_normalizes_in_date_like_column():
    values = ["2026-07-01", "2026-06-15", "2025-12-31", "2026-01-01",
              "2026-03-10", "2026-07-01", "2026-06-15", "2025-12-31"]
    out, report = fd.clean(date_frame(values), semantic_mode="auto", **DATE_COMMON)
    assert pd.api.types.is_datetime64_any_dtype(out["signup_date"])
    assert pd.Timestamp("2026-07-01") in list(out["signup_date"])
    assert any(a.model_id == "semantic:date_phrase:v1" for a in applied(report))


def test_unambiguous_numeric_dates_normalize():
    # Every value has a token > 12, so day/month order is forced either way.
    values = ["25/12/2026", "13-01-2026", "25/12/2026", "13-01-2026",
              "25/12/2026", "13-01-2026", "25/12/2026", "13-01-2026"]
    out, report = fd.clean(date_frame(values), semantic_mode="auto", **DATE_COMMON)
    assert pd.api.types.is_datetime64_any_dtype(out["signup_date"])
    assert pd.Timestamp("2026-12-25") in list(out["signup_date"])
    assert pd.Timestamp("2026-01-13") in list(out["signup_date"])
    assert any(a.model_id == "semantic:date_phrase:v1" for a in applied(report))


def test_month_name_dates_normalize():
    values = ["July 1 2026", "1 July 2026", "June 15 2026", "15 June 2026",
              "July 1 2026", "1 July 2026", "June 15 2026", "15 June 2026"]
    out, report = fd.clean(date_frame(values), semantic_mode="auto", **DATE_COMMON)
    assert pd.api.types.is_datetime64_any_dtype(out["signup_date"])
    assert pd.Timestamp("2026-07-01") in list(out["signup_date"])
    assert any(a.model_id == "semantic:date_phrase:v1" for a in applied(report))


def test_relative_phrases_normalize_only_with_reference_date():
    values = ["today", "yesterday", "tomorrow", "today",
              "yesterday", "tomorrow", "today", "yesterday"]
    ctx = {"reference_date": "2026-07-01", "columns": {"signup_date": {"semantic_type": "date"}}}
    out, report = fd.clean(
        date_frame(values), semantic_mode="auto", semantic_context=ctx, **DATE_COMMON
    )
    assert pd.Timestamp("2026-07-01") in list(out["signup_date"])
    assert pd.Timestamp("2026-06-30") in list(out["signup_date"])
    assert pd.Timestamp("2026-07-02") in list(out["signup_date"])
    assert any(a.model_id == "semantic:date_phrase:v1" for a in applied(report))


def test_relative_phrases_without_reference_date_are_not_auto_applied():
    values = ["today", "yesterday", "tomorrow", "today",
              "yesterday", "tomorrow", "today", "yesterday"]
    ctx = {"columns": {"signup_date": {"semantic_type": "date"}}}
    out, report = fd.clean(
        date_frame(values), semantic_mode="auto", semantic_context=ctx, **DATE_COMMON
    )
    assert out["signup_date"].tolist() == values
    assert not applied(report)
    assert suggested(report) or skipped(report)


def test_ambiguous_numeric_dates_without_dayfirst_are_not_mutated():
    values = ["01/02/2026", "03/04/2026", "05/06/2026", "02/01/2026",
              "04/03/2026", "06/05/2026", "01/02/2026", "03/04/2026"]
    ctx = {"columns": {"signup_date": {"semantic_type": "date"}}}
    out, report = fd.clean(
        date_frame(values), semantic_mode="auto", semantic_context=ctx, **DATE_COMMON
    )
    assert out["signup_date"].tolist() == values
    assert not applied(report)
    reviewed = suggested(report) + skipped(report)
    assert reviewed and all(a.risk == "high" for a in reviewed)


def test_ambiguous_numeric_dates_with_dayfirst_apply_correctly():
    values = ["01/02/2026"] * 4 + ["03/04/2026"] * 4
    ctx = {"columns": {"signup_date": {"semantic_type": "date", "dayfirst": True}}}
    out, report = fd.clean(
        date_frame(values), semantic_mode="auto", semantic_context=ctx, **DATE_COMMON
    )
    assert pd.Timestamp("2026-02-01") in list(out["signup_date"])
    assert pd.Timestamp("2026-04-03") in list(out["signup_date"])
    assert any(a.model_id == "semantic:date_phrase:v1" for a in applied(report))


def test_ambiguous_numeric_dates_fall_back_to_global_dayfirst_config():
    # No per-column dayfirst hint: the global CleanConfig.dayfirst setting
    # (already used elsewhere for this exact ambiguity) should apply too.
    values = ["01/02/2026"] * 4 + ["03/04/2026"] * 4
    ctx = {"columns": {"signup_date": {"semantic_type": "date"}}}
    out, report = fd.clean(
        date_frame(values), semantic_mode="auto", dayfirst=True, semantic_context=ctx,
        **DATE_COMMON,
    )
    assert pd.Timestamp("2026-02-01") in list(out["signup_date"])
    assert pd.Timestamp("2026-04-03") in list(out["signup_date"])
    assert any(a.model_id == "semantic:date_phrase:v1" for a in applied(report))


def test_date_values_in_free_text_column_are_not_converted():
    values = [f"note {i}: contact us for more info please" for i in range(7)] + ["2026-07-01"]
    out, report = fd.clean(date_frame(values), semantic_mode="auto", **DATE_COMMON)
    assert out["signup_date"].tolist() == values
    assert not any(
        a.column == "signup_date" and a.model_id.startswith("semantic:date_phrase")
        for a in sem(report)
    )


def test_date_values_in_protected_columns_are_not_converted():
    values = ["2026-07-01", "2026-06-15", "2025-12-31", "2026-01-01",
              "2026-03-10", "2026-07-01", "2026-06-15", "2025-12-31"]
    ctx = {"columns": {"signup_date": {"semantic_type": "date"}}}
    out, report = fd.clean(
        date_frame(values), semantic_mode="auto", target_column="signup_date",
        semantic_context=ctx, **DATE_COMMON,
    )
    assert out["signup_date"].tolist() == values
    assert skipped(report)
    assert all(a.status == "skipped" for a in sem(report) if a.column == "signup_date")
