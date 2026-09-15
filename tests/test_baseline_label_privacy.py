"""Baseline category labels must not be recoverable without a secret."""

from __future__ import annotations

import hashlib
import json
import re
import warnings

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import EnterpriseConfig, clean_enterprise
from freshdata.enterprise import interface as enterprise_interface
from freshdata.enterprise.contracts import (
    BASELINE_KEY_ENV,
    ColumnContract,
    DataContract,
    DatasetBaseline,
    build_baseline,
    compare_to_baseline,
    load_baseline,
    save_baseline,
)

GUESSES = ["HIV", "diabetes", "depression", "asthma", "cancer"]


@pytest.fixture(autouse=True)
def _no_ambient_key(monkeypatch):
    monkeypatch.delenv(BASELINE_KEY_ENV, raising=False)


def _sha1_label(value: str) -> str:
    return "h:" + hashlib.sha1(value.encode()).hexdigest()[:16]


def _abc(a: int, b: int, c: int) -> pd.DataFrame:
    return pd.DataFrame({"cat": ["a"] * a + ["b"] * b + ["c"] * c})


def _psi_findings(report, column="cat"):
    return [f for f in report.findings if f.check_id == "drift.psi" and f.column == column]


def _skipped(report):
    return [f for f in report.findings if f.check_id == "drift.categorical_drift_skipped"]


def test_poc_dictionary_attack_recovers_nothing(tmp_path):
    df = pd.DataFrame({"diagnosis": ["HIV", "HIV", "diabetes", "depression"] * 10})
    path = tmp_path / "b.json"
    save_baseline(fd.build_baseline(df, name="b"), path)
    raw = path.read_text(encoding="utf-8")
    stored = json.loads(raw)
    freqs = stored["columns"]["diagnosis"]["frequencies"]

    assert {g: freqs[_sha1_label(g)] for g in GUESSES if _sha1_label(g) in freqs} == {}
    for guess in GUESSES:
        assert guess not in raw
        assert hashlib.sha1(guess.encode()).hexdigest()[:16] not in raw
    assert "h:" not in raw
    assert stored["schema_version"] == "freshdata-baseline-v2"
    col = stored["columns"]["diagnosis"]
    assert col["metadata"]["label_mode"] == "rank"
    assert col["top_values"] == []
    assert freqs == {"r:0000": 0.5, "r:0001": 0.25, "r:0002": 0.25}


def test_rank_mode_identical_data_has_no_drift():
    base = build_baseline(_abc(60, 30, 10), name="b")
    report = compare_to_baseline(_abc(60, 30, 10), base)
    assert report.distribution_drift["cat"]["psi"] == pytest.approx(0.0, abs=1e-6)
    assert not _psi_findings(report)
    assert not _skipped(report)
    assert report.passed


def test_rank_mode_flags_a_shape_shift():
    base = build_baseline(_abc(60, 30, 10), name="b")
    report = compare_to_baseline(_abc(34, 33, 33), base)
    assert _psi_findings(report)


def test_rank_mode_cannot_see_a_label_swap():
    # Documented trade-off of the label-free profile.
    base = build_baseline(_abc(60, 30, 10), name="b")
    assert not _psi_findings(compare_to_baseline(_abc(10, 30, 60), base))


def test_rank_mode_new_categories_count_against_the_profile():
    base = build_baseline(_abc(50, 50, 0), name="b")
    current = pd.DataFrame({"cat": [f"v{i % 10}" for i in range(100)]})
    assert _psi_findings(compare_to_baseline(current, base))


def test_keyed_mode_is_label_aware_with_the_same_key():
    base = build_baseline(_abc(60, 30, 10), name="b", label_key="s3cret")
    assert not _psi_findings(compare_to_baseline(_abc(60, 30, 10), base, label_key="s3cret"))
    swapped = compare_to_baseline(_abc(10, 30, 60), base, label_key="s3cret")
    assert _psi_findings(swapped)
    assert not _skipped(swapped)


def test_keyed_labels_are_hmac_and_key_is_not_stored(tmp_path):
    key = "correct horse battery staple"
    base = build_baseline(_abc(60, 30, 10), name="b", label_key=key)
    col = base.columns["cat"]
    assert col.metadata["label_mode"] == "hmac-sha256"
    assert col.metadata["labels_hashed"] is True
    assert all(re.fullmatch(r"k:[0-9a-f]{32}", v) for v in col.top_values)
    assert set(col.frequencies) == set(col.top_values)

    path = tmp_path / "b.json"
    save_baseline(base, path)
    raw = path.read_text(encoding="utf-8")
    assert key not in raw
    assert key.encode().hex() not in raw
    for value in ("a", "b", "c"):
        assert _sha1_label(value) not in raw


def test_label_key_id_is_stable_and_key_specific():
    ids = {
        build_baseline(_abc(5, 3, 1), name="x", label_key="k1")
        .columns["cat"]
        .metadata["label_key_id"],
        build_baseline(_abc(9, 1, 1), name="y", label_key=b"k1")
        .columns["cat"]
        .metadata["label_key_id"],
    }
    assert len(ids) == 1
    other = build_baseline(_abc(5, 3, 1), name="x", label_key="k2")
    assert other.columns["cat"].metadata["label_key_id"] not in ids


@pytest.mark.parametrize(
    ("compare_key", "reason"),
    [(None, "missing_label_key"), ("wrong", "label_key_mismatch")],
)
def test_keyed_baseline_without_the_key_skips_categorical_psi(compare_key, reason):
    base = build_baseline(_abc(60, 30, 10), name="b", label_key="right")
    report = compare_to_baseline(_abc(10, 30, 60), base, label_key=compare_key)
    skipped = _skipped(report)
    assert len(skipped) == 1
    assert skipped[0].column == "cat"
    assert skipped[0].level == "warning"
    assert skipped[0].details["reason"] == reason
    assert not _psi_findings(report)
    assert "psi" not in report.distribution_drift.get("cat", {})
    assert report.passed  # a warning, not an error


def test_label_key_from_environment(monkeypatch):
    monkeypatch.setenv(BASELINE_KEY_ENV, "env-key")
    base = build_baseline(_abc(60, 30, 10), name="b")
    assert base.columns["cat"].metadata["label_mode"] == "hmac-sha256"
    assert _psi_findings(compare_to_baseline(_abc(10, 30, 60), base))
    monkeypatch.delenv(BASELINE_KEY_ENV)
    assert _skipped(compare_to_baseline(_abc(10, 30, 60), base))


def test_empty_label_key_is_rejected():
    with pytest.raises(ValueError, match="label_key"):
        build_baseline(_abc(1, 1, 1), name="b", label_key="")


def _v1_literal() -> dict:
    freqs = {_sha1_label("a"): 0.6, _sha1_label("b"): 0.3, _sha1_label("c"): 0.1}
    return {
        "schema_version": "freshdata-baseline-v1",
        "name": "legacy",
        "version": "1.0.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "freshdata_version": "2.0.0",
        "row_count": 100,
        "column_order": ["cat"],
        "columns": {
            "cat": {
                "name": "cat",
                "dtype": "object",
                "missing_ratio": 0.0,
                "cardinality": 3,
                "n_unique": 3,
                "n_rows": 100,
                "sample_values": [],
                "top_values": list(freqs),
                "frequencies": freqs,
                "metadata": {"labels_hashed": True},
            }
        },
        "contract": None,
        "trust_score": None,
        "metadata": {},
    }


def test_v1_baseline_loads_with_a_warning_and_compares_as_before(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(_v1_literal()), encoding="utf-8")
    with pytest.warns(UserWarning, match="reversible by dictionary"):
        base = load_baseline(path)

    assert not _psi_findings(compare_to_baseline(_abc(60, 30, 10), base))
    # The legacy comparison is label-aware, as it was.
    assert _psi_findings(compare_to_baseline(_abc(10, 30, 60), base))
    # Re-saving keeps the v1 tag, so the next load warns again.
    assert base.to_dict()["schema_version"] == "freshdata-baseline-v1"
    with pytest.warns(UserWarning, match="rebuild the baseline"):
        DatasetBaseline.from_dict(base.to_dict())


def test_unknown_schema_version_still_raises():
    literal = _v1_literal()
    literal["schema_version"] = "freshdata-baseline-v9"
    with pytest.raises(ValueError, match="unsupported baseline schema_version"):
        DatasetBaseline.from_dict(literal)


@pytest.mark.parametrize("label_key", [None, "k"])
def test_v2_round_trip_without_warning(tmp_path, label_key):
    df = _abc(60, 30, 10)
    base = build_baseline(df, name="b", label_key=label_key)
    path = tmp_path / "b.json"
    save_baseline(base, path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded = load_baseline(path)
    assert loaded.to_dict()["columns"] == base.to_dict()["columns"]
    report = compare_to_baseline(df, loaded, label_key=label_key)
    assert report.passed
    assert not _psi_findings(report)
    assert not _skipped(report)


def test_include_samples_keeps_raw_labels():
    base = build_baseline(_abc(60, 30, 10), name="b", include_samples=True, label_key="k")
    col = base.columns["cat"]
    assert col.metadata["label_mode"] == "raw"
    assert col.metadata["labels_hashed"] is False
    assert "label_key_id" not in col.metadata
    assert col.top_values == ("a", "b", "c")
    assert col.frequencies == {"a": 0.6, "b": 0.3, "c": 0.1}
    assert _psi_findings(compare_to_baseline(_abc(10, 30, 60), base))


def test_clean_enterprise_inline_baseline_uses_an_ephemeral_key(monkeypatch):
    seen = []
    real = enterprise_interface.build_baseline

    def spy(*args, **kwargs):
        seen.append(kwargs.get("label_key"))
        return real(*args, **kwargs)

    monkeypatch.setattr(enterprise_interface, "build_baseline", spy)
    df = pd.DataFrame({"cat": ["a"] * 60 + ["b"] * 30 + ["c"] * 10, "n": range(100)})
    contract = DataContract(name="c", columns=(ColumnContract(name="cat"),))
    ec = EnterpriseConfig(enable_contracts=True)

    res = clean_enterprise(df, enterprise=ec, contract=contract)
    res2 = clean_enterprise(df, enterprise=ec, contract=contract)

    assert len(seen) == 2
    assert all(isinstance(k, bytes) and len(k) == 32 for k in seen)
    assert seen[0] != seen[1]
    assert res.drift_report is not None
    assert not _skipped(res.drift_report)
    assert "psi" in res.drift_report.distribution_drift["cat"]
    assert res2.drift_report is not None and not _skipped(res2.drift_report)


@pytest.mark.parametrize("label_key", [None, "k"])
def test_bool_and_nullable_string_columns(label_key):
    df = pd.DataFrame(
        {
            "flag": pd.Series([True, False, True] * 40),
            "name": pd.Series(["x", None, "y"] * 40, dtype="string"),
        }
    )
    base = build_baseline(df, name="b", label_key=label_key)
    for col in ("flag", "name"):
        expected = "rank" if label_key is None else "hmac-sha256"
        assert base.columns[col].metadata["label_mode"] == expected
    report = compare_to_baseline(df, base, label_key=label_key)
    assert report.passed
    assert not _skipped(report)
    for col in ("flag", "name"):
        assert report.distribution_drift[col]["psi"] == pytest.approx(0.0, abs=1e-6)
