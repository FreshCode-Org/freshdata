"""clean_enterprise reports must not carry raw values of masked columns."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from freshdata.enterprise import (
    ClusterConfig,
    EnterpriseConfig,
    MaskingRule,
    PIIDetectionConfig,
    SemanticValidatorConfig,
    clean_enterprise,
)

EMAILS = ["john.doe@acme.com", "John.Doe@acme.com", "jane@x.org"]


def _frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": EMAILS * n,
            "city": ["Paris", "paris", "Rome"] * n,
            "v": range(3 * n),
        }
    )


def _leaks(res, values) -> list[str]:
    text = res.to_json()
    return sorted(v for v in set(values) if v in text)


def _email_results(res):
    return [r for r in res.cluster_results if r.column == "email"]


def test_poc_hashed_column_is_not_in_the_report():
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("email",), strategy="hash"),),
        enable_clustering=True,
        clustering=ClusterConfig(),
    )
    res = clean_enterprise(_frame(), enterprise=ec)
    assert sorted(set(res.data["email"]) & set(EMAILS)) == []
    assert [e for e in set(EMAILS) if e in json.dumps(res.to_dict()["clusters"])] == []
    assert _leaks(res, EMAILS) == []


@pytest.mark.parametrize(
    ("rule_kwargs", "consistent"),
    [
        ({"strategy": "hash"}, True),
        ({"strategy": "redact"}, True),
        ({"strategy": "partial"}, True),
        ({"strategy": "tokenize", "key": "k" * 32}, False),
        ({"strategy": "surrogate", "key": "k" * 32}, False),
        ({"strategy": "drop"}, False),
    ],
    ids=["hash", "redact", "partial", "tokenize", "surrogate", "drop"],
)
def test_masked_column_clusters_are_masked_on_the_result(rule_kwargs, consistent):
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("email",), **rule_kwargs),),
        enable_clustering=True,
        clustering=ClusterConfig(columns=("email",)),
    )
    res = clean_enterprise(_frame(), enterprise=ec)
    assert _leaks(res, EMAILS) == []

    results = _email_results(res)
    assert results and all(r.redacted and r.mapping == {} for r in results)
    clusters = [c for r in results for c in r.clusters]
    assert clusters
    assert all(c.key == "<redacted>" for c in clusters)
    shown = {c.canonical for c in clusters} | {v for c in clusters for v in c.variants}
    assert not shown & set(EMAILS)
    if consistent:
        # Deterministic rules reproduce the token in the masked data.
        assert {c.canonical for c in clusters} <= set(res.data["email"])
    else:
        assert shown == {"<redacted>"}
    assert all(r.to_dict()["redacted"] is True for r in results)


def test_fingerprint_ngram_results_are_all_redacted():
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("email",), strategy="hash"),),
        enable_clustering=True,
        clustering=ClusterConfig(method="fingerprint_ngram"),
    )
    res = clean_enterprise(_frame(), enterprise=ec)
    results = _email_results(res)
    assert {r.method for r in results} == {"fingerprint", "ngram"}
    assert all(r.redacted for r in results)
    assert _leaks(res, EMAILS) == []


def test_pattern_rule_selects_columns_for_redaction():
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", pattern="^em", strategy="redact"),),
        enable_clustering=True,
        clustering=ClusterConfig(),
    )
    res = clean_enterprise(_frame(), enterprise=ec)
    assert all(r.redacted for r in _email_results(res))
    assert _leaks(res, EMAILS) == []


def test_detection_scrubbed_free_text_column_is_redacted():
    notes = ["contact john.doe@acme.com today", "Contact John.Doe@acme.com today", "no pii here"]
    df = pd.DataFrame({"note": notes * 4, "v": range(12)})
    ec = EnterpriseConfig(
        enable_privacy_detection=True,
        privacy=PIIDetectionConfig(),
        enable_clustering=True,
        clustering=ClusterConfig(),
    )
    res = clean_enterprise(df, enterprise=ec)
    assert res.privacy_report is not None and "note" in res.privacy_report.columns_changed
    results = [r for r in res.cluster_results if r.column == "note"]
    assert results and all(r.redacted for r in results)
    assert _leaks(res, ["john.doe@acme.com", "John.Doe@acme.com"]) == []


def test_semantic_validation_samples_are_masked_for_masked_columns():
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("email",), strategy="hash"),),
        semantic=(
            SemanticValidatorConfig(name="nope", columns=("email",), kind="regex", regex="^zzz$"),
            SemanticValidatorConfig(name="cities", columns=("city",), kind="reference",
                                    reference=("Rome",)),
        ),
    )
    res = clean_enterprise(_frame(), enterprise=ec)
    email_cv = res.validation_report.columns["email"]
    assert email_cv.invalid_samples
    assert not set(email_cv.invalid_samples) & set(EMAILS)
    assert set(email_cv.invalid_samples) <= set(res.data["email"])
    assert email_cv.n_invalid == 9  # counts kept
    # Unmasked columns keep their raw samples.
    assert "Paris" in res.validation_report.columns["city"].invalid_samples
    assert _leaks(res, EMAILS) == []


def test_unmasked_columns_keep_raw_clusters_and_counts():
    cfg = ClusterConfig()
    plain = clean_enterprise(
        _frame(), enterprise=EnterpriseConfig(enable_clustering=True, clustering=cfg)
    )
    masked = clean_enterprise(
        _frame(),
        enterprise=EnterpriseConfig(
            masking=(MaskingRule(name="m", columns=("email",), strategy="hash"),),
            enable_clustering=True,
            clustering=cfg,
        ),
    )
    city = [r for r in masked.cluster_results if r.column == "city"]
    assert city and not any(r.redacted for r in city)
    assert city[0].mapping
    assert {c.canonical for r in city for c in r.clusters} <= {"Paris", "paris"}
    assert masked.cells_merged == plain.cells_merged
    assert [r.n_clusters for r in masked.cluster_results] == [
        r.n_clusters for r in plain.cluster_results
    ]


def test_masking_disabled_leaves_clusters_raw():
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("email",), strategy="hash"),),
        enable_masking=False,
        enable_clustering=True,
        clustering=ClusterConfig(),
    )
    res = clean_enterprise(_frame(), enterprise=ec)
    assert not any(r.redacted for r in res.cluster_results)


def test_polars_input():
    pl = pytest.importorskip("polars")
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("email",), strategy="hash"),),
        enable_clustering=True,
        clustering=ClusterConfig(),
    )
    res = clean_enterprise(pl.from_pandas(_frame()), enterprise=ec)
    assert isinstance(res.data, pl.DataFrame)
    assert all(r.redacted for r in _email_results(res))
    assert _leaks(res, EMAILS) == []


@pytest.mark.parametrize("strategy", ["hash", "drop"])
def test_coerced_cells_of_masked_date_like_column(strategy):
    born = pd.date_range("1970-01-01", periods=99, freq="D").strftime("%Y-%m-%d").tolist()
    amount = [f"{i}.5" for i in range(99)]
    df = pd.DataFrame(
        {
            "born": born + ["31/31/XYZZY"],
            "amount": amount + ["12abcQWERTY"],
            "open": amount + ["7xyzOPEN"],
        }
    )
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("born", "amount"), strategy=strategy),)
    )
    res = clean_enterprise(df, enterprise=ec)
    coerced = res.clean_report.coerced_cells
    assert set(coerced) >= {"born", "amount", "open"}
    assert "XYZZY" not in json.dumps(coerced["born"], default=str)
    assert "QWERTY" not in json.dumps(coerced["amount"], default=str)
    if strategy == "drop":
        assert list(coerced["born"].values()) == ["<redacted>"]
    assert _leaks(res, ["XYZZY", "QWERTY"]) == []
    assert not [w for w in res.clean_report.warnings if "XYZZY" in w or "QWERTY" in w]
    # The unmasked column keeps its reviewable original.
    assert list(coerced["open"].values()) == ["7xyzOPEN"]


def test_strict_rule_on_column_absent_from_reports_does_not_raise():
    # "email" is masked but appears in no cluster, validation or coercion
    # report; report redaction must not re-apply ``strict`` to that subset.
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("Email",), strategy="hash", strict=True),),
    )
    df = pd.DataFrame({"email": ["a@x.com", "b@y.io"], "v": [1, 2]})
    res = clean_enterprise(df, enterprise=ec)
    assert "a@x.com" not in set(res.data["email"])
    assert _leaks(res, ["a@x.com", "b@y.io"]) == []


def test_strict_rule_on_column_missing_from_frame_still_raises():
    ec = EnterpriseConfig(
        masking=(MaskingRule(name="m", columns=("nope",), strategy="hash", strict=True),),
    )
    with pytest.raises(ValueError, match="not found in dataframe"):
        clean_enterprise(pd.DataFrame({"email": ["a@x.com"]}), enterprise=ec)
