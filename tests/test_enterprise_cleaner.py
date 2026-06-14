"""Clustering, PII masking, semantic validation, and cleanlab-guard tests."""

import pandas as pd
import pytest

from freshdata.adapters.polars import is_polars_frame
from freshdata.enterprise import (
    ISO_COUNTRY_ALPHA2,
    PII_PATTERNS,
    APISemanticValidator,
    CallableValidator,
    ClusterConfig,
    MaskingRule,
    ReferenceSetValidator,
    RegexValidator,
    SemanticValidatorConfig,
    build_validator,
    cluster_column,
    detect_label_issues,
    detect_outliers,
    iso_country_validator,
    mask_dataframe,
    merge_clusters,
    run_semantic_validation,
    validate_columns,
)
from freshdata.enterprise.cleaner import _fingerprint_str, _ngram_str, _pick_canonical

# =====================================================================
# Clustering
# =====================================================================

def test_fingerprint_merges_case_punct_wordorder_variants():
    df = pd.DataFrame({"v": ["Acme Inc", "ACME  inc", "acme, inc.", "Globex", "globex"]})
    out, results = merge_clusters(df, ["v"])
    assert out["v"].nunique() == 2
    assert results[0].n_clusters == 2
    assert results[0].n_cells_merged == 3


def test_fingerprint_parity_pandas_vs_polars():
    pl = pytest.importorskip("polars")
    data = {"v": ["Acme Inc", "ACME  inc", "acme inc", "Globex"]}
    pandas_result = cluster_column(pd.DataFrame(data), "v")
    polars_result = cluster_column(pl.DataFrame(data), "v")
    assert pandas_result.mapping == polars_result.mapping
    assert pandas_result.mapping  # something actually merged


def test_polars_in_polars_out():
    pl = pytest.importorskip("polars")
    out, _ = merge_clusters(pl.DataFrame({"v": ["aa", "AA", "bb"]}), ["v"])
    assert is_polars_frame(out)
    assert len(set(out["v"].to_list())) == 2


def test_ngram_merges_what_fingerprint_misses():
    df = pd.DataFrame({"v": ["ab cd", "ab cd", "abcd"]})
    out_f, _ = merge_clusters(df, ["v"], ClusterConfig(columns=("v",), method="fingerprint"))
    assert out_f["v"].nunique() == 2  # fingerprint keeps "ab cd" and "abcd" apart
    out_n, _ = merge_clusters(df, ["v"], ClusterConfig(columns=("v",), method="ngram"))
    assert out_n["v"].nunique() == 1  # n-grams collapse them


def test_fingerprint_ngram_runs_two_passes():
    df = pd.DataFrame({"v": ["Acme Inc", "ACME inc"]})
    _, results = merge_clusters(
        df, ["v"], ClusterConfig(columns=("v",), method="fingerprint_ngram")
    )
    assert [r.method for r in results] == ["fingerprint", "ngram"]


@pytest.mark.parametrize(
    "policy, expected",
    [("most_frequent", "bb"), ("longest", "cccc"), ("shortest", "bb"), ("first", "aaa")],
)
def test_pick_canonical_policies(policy, expected):
    members = [("aaa", 1), ("bb", 5), ("cccc", 2)]
    assert _pick_canonical(members, policy) == expected


def test_canonical_most_frequent_wins():
    df = pd.DataFrame({"v": ["Foo", "Foo", "Foo", "foo", "FOO"]})
    result = cluster_column(df, "v", config=ClusterConfig(canonical="most_frequent"))
    assert result.clusters[0].canonical == "Foo"


def test_min_cluster_size_gate_blocks_small_clusters():
    result = cluster_column(
        pd.DataFrame({"v": ["a", "A"]}), "v", config=ClusterConfig(min_cluster_size=10)
    )
    assert result.n_clusters == 0
    assert result.mapping == {}


def test_cluster_column_rejects_non_text():
    with pytest.raises(ValueError, match="text column"):
        cluster_column(pd.DataFrame({"n": [1, 2, 3]}), "n")


def test_merge_clusters_autodetects_text_and_skips_missing():
    df = pd.DataFrame({"v": ["a", "A", "b"], "num": [1, 2, 3]})
    _, results = merge_clusters(df)
    assert {r.column for r in results} == {"v"}
    _, missing = merge_clusters(df, ["does_not_exist"])
    assert missing == []


def test_apply_mapping_noop_when_all_distinct():
    df = pd.DataFrame({"v": ["a", "b", "c"]})
    out, results = merge_clusters(df, ["v"])
    assert out["v"].tolist() == ["a", "b", "c"]
    assert results[0].mapping == {}


def test_cluster_result_serialization():
    result = cluster_column(pd.DataFrame({"v": ["Foo", "foo", "Foo"]}), "v")
    payload = result.to_dict()
    assert payload["column"] == "v"
    assert payload["clusters"][0]["canonical"] == "Foo"
    assert repr(result).startswith("<ClusterResult")


def test_fingerprint_and_ngram_string_helpers():
    assert _fingerprint_str("I.B.M Corp") == "corp ibm"
    assert _ngram_str("abcd", 2) == "abbccd"
    assert _ngram_str("ab", 3) == "ab"  # shorter than n -> cleaned value


# =====================================================================
# PII masking
# =====================================================================

def test_hash_is_deterministic_salt_sensitive_and_null_safe():
    df = pd.DataFrame({"e": ["a@x.com", "b@y.io", None]})
    rule = MaskingRule(name="h", columns=("e",), strategy="hash", salt="s1", hash_length=16)
    out1, _ = mask_dataframe(df, [rule])
    out2, _ = mask_dataframe(df, [rule])
    out3, _ = mask_dataframe(
        df, [MaskingRule(name="h", columns=("e",), strategy="hash", salt="s2")]
    )
    assert out1["e"].iloc[0] == out2["e"].iloc[0]          # deterministic
    assert out1["e"].iloc[0] != out3["e"].iloc[0]          # salt changes output
    assert len(out1["e"].iloc[0]) == 16                    # truncated length
    assert pd.isna(out1["e"].iloc[2])                      # null preserved


def test_redact_and_partial():
    out_r, _ = mask_dataframe(
        pd.DataFrame({"e": ["secret", None]}),
        [MaskingRule(name="r", columns=("e",), strategy="redact", placeholder="X")],
    )
    assert out_r["e"].iloc[0] == "X" and pd.isna(out_r["e"].iloc[1])

    out_p, _ = mask_dataframe(
        pd.DataFrame({"c": ["1234567890"]}),
        [MaskingRule(name="p", columns=("c",), strategy="partial", visible=4, placeholder="***")],
    )
    assert out_p["c"].iloc[0] == "***7890"


def test_regex_scrub_email_counts_changed_rows():
    df = pd.DataFrame({"t": ["mail a@x.com please", "nothing here"]})
    out, report = mask_dataframe(
        df,
        [MaskingRule(name="s", columns=("t",), strategy="regex_scrub",
                     scrub_patterns=("email",), placeholder="[E]")],
    )
    assert "[E]" in out["t"].iloc[0]
    assert "@" not in out["t"].iloc[0]
    assert report.cells_masked["t"] == 1


def test_drop_strategy_removes_column():
    out, report = mask_dataframe(
        pd.DataFrame({"e": ["x"], "keep": ["y"]}),
        [MaskingRule(name="d", columns=("e",), strategy="drop")],
    )
    assert "e" not in out.columns and "keep" in out.columns
    assert report.columns["e"] == "drop"


def test_mask_by_name_pattern():
    df = pd.DataFrame({"user_ssn": ["1"], "user_name": ["bob"]})
    out, report = mask_dataframe(
        df, [MaskingRule(name="byname", pattern=r"_ssn$", strategy="redact", placeholder="X")]
    )
    assert out["user_ssn"].iloc[0] == "X" and out["user_name"].iloc[0] == "bob"
    assert "user_ssn" in report.columns and "user_name" not in report.columns


def test_mask_report_serialization():
    _, report = mask_dataframe(
        pd.DataFrame({"e": ["a", "b"]}),
        [MaskingRule(name="r", columns=("e",), strategy="redact")],
    )
    payload = report.to_dict()
    assert payload["total_cells_masked"] == 2
    assert payload["columns"]["e"] == "redact"
    assert "r" in payload["rules_applied"]
    assert repr(report).startswith("<MaskReport")


def test_masking_polars_all_strategies():
    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"e": ["a@x.com", None], "c": ["123456", "999999"]})
    out_h, _ = mask_dataframe(frame, [MaskingRule(name="h", columns=("e",), strategy="hash")])
    assert is_polars_frame(out_h) and out_h["e"][1] is None and len(out_h["e"][0]) == 16
    out_r, _ = mask_dataframe(
        frame, [MaskingRule(name="r", columns=("e",), strategy="redact", placeholder="X")]
    )
    assert out_r["e"][0] == "X"
    out_p, _ = mask_dataframe(
        frame,
        [MaskingRule(name="p", columns=("c",), strategy="partial", visible=2, placeholder="**")],
    )
    assert out_p["c"][0] == "**56"
    out_s, report = mask_dataframe(
        frame, [MaskingRule(name="s", columns=("e",), strategy="regex_scrub",
                            scrub_patterns=("email",), placeholder="[E]")]
    )
    assert out_s["e"][0] == "[E]" and report.cells_masked["e"] == 1
    out_d, _ = mask_dataframe(frame, [MaskingRule(name="d", columns=("c",), strategy="drop")])
    assert "c" not in out_d.columns


def test_pii_patterns_present():
    assert {"email", "phone", "ssn", "credit_card", "ip", "iban"} <= set(PII_PATTERNS)


# =====================================================================
# Semantic validation
# =====================================================================

def test_reference_validator_case_handling_and_nulls():
    insensitive = ReferenceSetValidator("c", ["US", "CA"])
    assert insensitive.validate(["US", "us", "XX", None]) == [True, True, False, True]
    sensitive = ReferenceSetValidator("c", ["US"], case_sensitive=True)
    assert sensitive.validate(["US", "us"]) == [True, False]


def test_regex_validator_fullmatch_and_ignorecase():
    v = RegexValidator("zip", r"\d{5}")
    assert v.validate(["12345", "1234", "abcde", None]) == [True, False, False, True]
    assert RegexValidator("alpha", r"[a-z]+", case_sensitive=False).validate(["ABC"]) == [True]


def test_callable_validator():
    v = CallableValidator("pos", lambda x: x > 0)
    assert v.validate([1, -1, None]) == [True, False, True]


def test_api_validator_caches_and_skips_nulls():
    calls = []

    def fake_get(value):
        calls.append(value)
        return value.startswith("ok")

    v = APISemanticValidator("api", "http://example/validate", http_get=fake_get)
    assert v.validate(["ok1", "bad", "ok1", None]) == [True, False, True, True]
    assert calls == ["ok1", "bad"]  # repeat cached, null never queried


def test_build_validator_dispatch():
    ref = build_validator(SemanticValidatorConfig(name="r", kind="reference", reference=("US",)))
    rgx = build_validator(SemanticValidatorConfig(name="g", kind="regex", regex=r"\d+"))
    api = build_validator(SemanticValidatorConfig(name="a", kind="api", api_url="http://x"))
    assert isinstance(ref, ReferenceSetValidator)
    assert isinstance(rgx, RegexValidator)
    assert isinstance(api, APISemanticValidator)


def test_validate_columns_report_and_skip_missing():
    df = pd.DataFrame({"country": ["US", "XX", "CA", None], "other": [1, 2, 3, 4]})
    report = validate_columns(df, {"country": ReferenceSetValidator("iso", ["US", "CA"]),
                                   "missing": ReferenceSetValidator("x", ["1"])})
    cv = report.columns["country"]
    assert cv.n_invalid == 1 and cv.n_checked == 3
    assert "XX" in cv.invalid_samples
    assert report.n_invalid_total == 1
    assert cv.to_dict()["validator"] == "iso"
    assert report.to_dict()["n_invalid_total"] == 1
    assert repr(report).startswith("<ValidationReport")
    assert "missing" not in report.columns  # column absent from frame


def test_run_semantic_validation_from_configs():
    df = pd.DataFrame({"country": ["US", "ZZ"]})
    report = run_semantic_validation(
        df,
        [SemanticValidatorConfig(name="iso", kind="reference", columns=("country",),
                                 reference=("US", "CA"))],
    )
    assert report.columns["country"].n_invalid == 1


def test_validate_columns_polars():
    pl = pytest.importorskip("polars")
    report = validate_columns(pl.DataFrame({"country": ["US", "XX"]}),
                              {"country": iso_country_validator()})
    assert report.columns["country"].n_invalid == 1


def test_iso_country_validator_and_set():
    v = iso_country_validator()
    assert v.validate(["US", "us", "ZZ"]) == [True, True, False]
    assert "US" in ISO_COUNTRY_ALPHA2 and len(ISO_COUNTRY_ALPHA2) > 200


# =====================================================================
# Optional cleanlab wrapper
# =====================================================================

def test_cleanlab_wrappers_raise_clear_error_when_absent():
    try:
        import cleanlab  # noqa: F401, PLC0415
        pytest.skip("cleanlab is installed; the guard path cannot be exercised")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="cleanlab"):
        detect_label_issues([0, 1], [[0.9, 0.1], [0.2, 0.8]])
    with pytest.raises(ImportError, match="cleanlab"):
        detect_outliers([[1, 2], [3, 4]])
