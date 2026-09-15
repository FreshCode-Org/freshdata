"""Privacy engine correctness: missing values, categorical k-anonymity, labels and metadata.

* #243: ``pd.NA``/``NaT`` cells stay missing under every strategy and action.
* #244: categorical quasi-identifiers do not produce empty equivalence classes.
* #265: duplicate column labels raise a clear ``ValueError``.
* #281: fpe audit metadata follows the mode each cell actually used.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from freshdata.enterprise import (
    EnterpriseConfig,
    KAnonymityConfig,
    MaskingRule,
    PIIDetectionConfig,
    PrivacyPolicy,
    PrivacyRule,
    anonymize,
    apply_privacy_policy,
    check_k_anonymity,
    clean_enterprise,
    detect_pii,
    load_compliance_pack,
)

KEY = "unit-test-key"


@pytest.fixture
def no_pyffx(monkeypatch):
    """Force the surrogate fallback of ``fpe`` whether or not pyffx is installed."""
    monkeypatch.setitem(sys.modules, "pyffx", None)


# --------------------------------------------------------------------------
# #243: missing cells are passed through, not masked
# --------------------------------------------------------------------------

#: dtype label -> column with a missing cell at position 1 and two real values.
_NULLABLE_COLUMNS = {
    "string": lambda: pd.array(["alice@example.com", None, "bob@example.com"], dtype="string"),
    "Int64": lambda: pd.array([1234567890, None, 9876543210], dtype="Int64"),
    "boolean": lambda: pd.array([True, None, False], dtype="boolean"),
    "datetime64[ns]": lambda: pd.to_datetime(["2024-01-02", None, "2024-03-04"]),
    "object_pd_NA": lambda: pd.Series(
        ["alice@example.com", pd.NA, "bob@example.com"], dtype=object
    ),
}

_RULES = {
    "hash": MaskingRule(name="r", columns=("c",), strategy="hash", salt="fixed-salt"),
    "redact": MaskingRule(name="r", columns=("c",), strategy="redact"),
    "partial": MaskingRule(name="r", columns=("c",), strategy="partial"),
    "regex_scrub": MaskingRule(
        name="r", columns=("c",), strategy="regex_scrub", scrub_patterns=(), regexes=(r".+",)
    ),
    "tokenize": MaskingRule(name="r", columns=("c",), strategy="tokenize", key=KEY),
    "surrogate": MaskingRule(name="r", columns=("c",), strategy="surrogate"),
    "fpe": MaskingRule(name="r", columns=("c",), strategy="fpe", key=KEY),
}


@pytest.mark.parametrize("dtype", list(_NULLABLE_COLUMNS))
@pytest.mark.parametrize("strategy", list(_RULES))
def test_rule_strategies_keep_missing_cells_missing(dtype, strategy, no_pyffx):
    df = pd.DataFrame({"c": _NULLABLE_COLUMNS[dtype]()})
    out, report = anonymize(df, rules=(_RULES[strategy],))
    assert pd.isna(out["c"].iloc[1])
    assert not pd.isna(out["c"].iloc[0]) and not pd.isna(out["c"].iloc[2])
    assert report.cells_changed == 2
    assert sorted(e.row for e in report.events) == [0, 2]


@pytest.mark.parametrize("dtype", list(_NULLABLE_COLUMNS))
def test_drop_counts_only_present_cells(dtype):
    df = pd.DataFrame({"c": _NULLABLE_COLUMNS[dtype](), "keep": [1, 2, 3]})
    out, report = anonymize(df, rules=(MaskingRule(name="r", columns=("c",), strategy="drop"),))
    assert "c" not in out.columns
    assert report.cells_changed == 2


_POLICY_CASES = [
    (dtype, action)
    for action in ("tokenize", "pseudonymize", "redact", "quarantine")
    for dtype in _NULLABLE_COLUMNS
    # quarantine writes a string placeholder with Series.where, which masked
    # Int64/boolean arrays reject; that is separate from missing-value handling.
    if not (action == "quarantine" and dtype in ("Int64", "boolean"))
]


@pytest.mark.parametrize(("dtype", "action"), _POLICY_CASES)
def test_policy_actions_keep_missing_cells_missing(dtype, action, no_pyffx):
    df = pd.DataFrame({"c": _NULLABLE_COLUMNS[dtype]()})
    rule = PrivacyRule(id="r", action=action, columns=("c",), classification="s")
    out, report = apply_privacy_policy(df, PrivacyPolicy(name="p", rules=(rule,), key=KEY))
    assert pd.isna(out["c"].iloc[1])
    assert not pd.isna(out["c"].iloc[0]) and not pd.isna(out["c"].iloc[2])
    assert report.cells_changed == 2


@pytest.mark.parametrize("dtype", ["string", "object_pd_NA"])
def test_detection_path_keeps_missing_cells_missing(dtype):
    values = ["call 123-45-6789 today", None, "ssn 987-65-4321"]
    series = (
        pd.array(values, dtype="string")
        if dtype == "string"
        else pd.Series([values[0], pd.NA, values[2]], dtype=object)
    )
    df = pd.DataFrame({"notes": series})
    out, report = anonymize(df, detection_config=PIIDetectionConfig())
    assert pd.isna(out["notes"].iloc[1])
    assert "<SSN>" in out["notes"].iloc[0] and "<SSN>" in out["notes"].iloc[2]
    assert report.entities_found == 2
    scan = detect_pii(df)
    assert sorted(e.metadata["row"] for e in scan.entities) == [0, 2]


def test_missing_masked_output_matches_object_none_control():
    rule = MaskingRule(name="h", columns=("c",), strategy="hash", salt="fixed-salt")
    nullable, _ = anonymize(
        pd.DataFrame({"c": pd.array(["a@b.com", None], dtype="string")}), rules=(rule,)
    )
    control, _ = anonymize(pd.DataFrame({"c": ["a@b.com", None]}), rules=(rule,))
    assert nullable["c"].iloc[0] == control["c"].iloc[0]


def test_issue_243_redact_repro():
    df = pd.DataFrame({"email": pd.array(["a@b.com", None], dtype="string")})
    out, report = anonymize(
        df, rules=(MaskingRule(name="r", columns=("email",), strategy="redact"),)
    )
    assert out["email"].iloc[0] == "***"
    assert pd.isna(out["email"].iloc[1])
    assert report.cells_changed == 1


def test_issue_243_tokenize_repro():
    df = pd.DataFrame({"email": pd.array(["a@b.com", None], dtype="string")})
    out, report = anonymize(
        df, rules=(MaskingRule(name="t", columns=("email",), strategy="tokenize", key="k"),)
    )
    assert out["email"].iloc[0].startswith("tok_")
    assert pd.isna(out["email"].iloc[1])
    assert report.cells_changed == 1


def test_issue_243_policy_repro():
    df = pd.DataFrame({"email": pd.array(["a@b.com", None], dtype="string")})
    policy = PrivacyPolicy(packs=(load_compliance_pack("hipaa"),), jurisdiction="US")
    out, report = apply_privacy_policy(df, policy)
    assert out["email"].iloc[0] == "<EMAIL>"
    assert pd.isna(out["email"].iloc[1])
    assert report.cells_changed == 1


# --------------------------------------------------------------------------
# #244: k-anonymity with categorical quasi-identifiers
# --------------------------------------------------------------------------


def _zip_sex_frame() -> pd.DataFrame:
    return pd.DataFrame({"zip": ["10001"] * 5 + ["20002"] * 5, "sex": ["F"] * 5 + ["M"] * 5})


def test_issue_244_categorical_matches_object():
    df = _zip_sex_frame()
    obj = check_k_anonymity(df, ["zip", "sex"], k=5)
    cat = check_k_anonymity(df.astype("category"), ["zip", "sex"], k=5)
    for report in (obj, cat):
        assert report.ok
        assert report.smallest_class_size == 5
        assert report.n_equivalence_classes == 2
        assert report.rows_violating_k == 0
        assert report.high_risk_groups == []
    assert cat.to_dict() == obj.to_dict()


def test_categorical_quasi_identifier_with_missing_value():
    df = pd.DataFrame({"zip": ["10001", "10001", np.nan, "20002"], "sex": ["F", "F", "M", "M"]})
    obj = check_k_anonymity(df, ["zip", "sex"], k=2)
    cat = check_k_anonymity(df.astype("category"), ["zip", "sex"], k=2)
    assert cat.n_equivalence_classes == obj.n_equivalence_classes == 3
    assert cat.smallest_class_size == obj.smallest_class_size == 1
    assert cat.rows_violating_k == obj.rows_violating_k == 2
    assert {"zip": None, "sex": "M"} in [g["group"] for g in cat.high_risk_groups]
    assert all(g["size"] > 0 for g in cat.high_risk_groups)


def test_unused_categories_are_not_classes():
    df = pd.DataFrame(
        {
            "zip": pd.Categorical(["10001"] * 3, categories=["10001", "20002", "30003"]),
            "sex": pd.Categorical(["F"] * 3, categories=["F", "M", "X"]),
        }
    )
    report = check_k_anonymity(df, ["zip", "sex"], k=3)
    assert report.ok
    assert report.n_equivalence_classes == 1
    assert report.smallest_class_size == 3


def test_mixed_categorical_and_object_quasi_identifiers():
    df = _zip_sex_frame()
    df["zip"] = df["zip"].astype("category")
    report = check_k_anonymity(df, ["zip", "sex"], k=5)
    assert report.ok
    assert report.n_equivalence_classes == 2
    assert report.smallest_class_size == 5


def test_clean_enterprise_k_anonymity_with_categorical_quasi_identifiers():
    df = _zip_sex_frame().astype("category")
    df["value"] = range(len(df))
    config = EnterpriseConfig(
        k_anonymity=KAnonymityConfig(enabled=True, quasi_identifiers=("zip", "sex"), k=5)
    )
    result = clean_enterprise(df, enterprise=config)
    report = result.k_anonymity_report
    assert report is not None
    assert report.ok
    assert report.n_equivalence_classes == 2
    assert report.smallest_class_size == 5


# --------------------------------------------------------------------------
# #265 part 1: duplicate column labels in detect_pii / anonymize
# --------------------------------------------------------------------------


def _duplicate_email_frame() -> pd.DataFrame:
    return pd.DataFrame([["a@b.com", "c@d.com"]], columns=["email", "email"])


def test_issue_265_detect_pii_rejects_duplicate_labels():
    with pytest.raises(ValueError, match=r"detect_pii requires unique column labels.*'email'"):
        detect_pii(_duplicate_email_frame())


def test_issue_265_anonymize_detection_rejects_duplicate_labels():
    df = _duplicate_email_frame()
    with pytest.raises(ValueError, match=r"requires unique column labels.*'email'"):
        anonymize(df, detection_config=PIIDetectionConfig())
    assert df.iloc[0].tolist() == ["a@b.com", "c@d.com"]


def test_anonymize_rule_targeting_duplicated_label_raises():
    df = _duplicate_email_frame()
    rule = MaskingRule(name="r", columns=("email",), strategy="redact")
    with pytest.raises(ValueError, match=r"rule 'r'.*'email'"):
        anonymize(df, rules=(rule,))


def test_anonymize_rule_on_unique_column_with_duplicates_elsewhere():
    df = pd.DataFrame([["x", "y", "a@b.com"]], columns=["dup", "dup", "email"])
    rule = MaskingRule(name="r", columns=("email",), strategy="redact")
    out, report = anonymize(df, rules=(rule,))
    assert out.columns.tolist() == ["dup", "dup", "email"]
    assert out.iloc[0].tolist() == ["x", "y", "***"]
    assert report.cells_changed == 1


def test_anonymize_disabled_detection_ignores_duplicates_elsewhere():
    df = pd.DataFrame([["x", "y", "a@b.com"]], columns=["dup", "dup", "email"])
    rule = MaskingRule(name="r", columns=("email",), strategy="redact")
    out, _ = anonymize(df, rules=(rule,), detection_config=PIIDetectionConfig(enabled=False))
    assert out.iloc[0].tolist() == ["x", "y", "***"]


# --------------------------------------------------------------------------
# #281 parts 1-2: fpe audit metadata follows the mode actually used
# --------------------------------------------------------------------------

_SURROGATE_MODE = "surrogate_format_preserving_not_crypto_fpe"


class _StubInteger:
    """Stand-in for ``pyffx.Integer`` that cannot encrypt the number 999."""

    def __init__(self, key, length):
        self.length = length

    def encrypt(self, n):
        if n == 999:
            raise ValueError("stub cannot encrypt 999")
        return (n * 7 + 3) % (10**self.length)


@pytest.fixture
def stub_pyffx(monkeypatch):
    module = types.ModuleType("pyffx")
    module.Integer = _StubInteger
    monkeypatch.setitem(sys.modules, "pyffx", module)


def test_issue_281_surrogate_fallback_is_not_reported_reversible(no_pyffx):
    df = pd.DataFrame({"ssn": ["123-45-6789"]})
    rule = MaskingRule(name="r", columns=("ssn",), strategy="fpe", reversible=True, key="k")
    out, report = anonymize(df, rules=(rule,))
    assert out["ssn"].iloc[0] == "119-85-2634"
    assert [e.reversible for e in report.events] == [False]
    assert report.metadata == {"fpe_mode": _SURROGATE_MODE}


def test_issue_281_mixed_modes_in_one_column_are_counted(stub_pyffx):
    df = pd.DataFrame({"x": ["abc-def", "123-45-6789", "999"]})
    rule = MaskingRule(name="r", columns=("x",), strategy="fpe", reversible=True, key="k")
    out, report = anonymize(df, rules=(rule,))
    assert out["x"].iloc[0] == "xhs-rps"
    assert out["x"].iloc[1] == "864-19-7526"
    assert report.metadata["fpe_mode"] == "mixed"
    assert report.metadata["fpe_modes"] == {"x": {_SURROGATE_MODE: 2, "crypto_fpe": 1}}
    assert {e.row: e.reversible for e in report.events} == {0: False, 1: True, 2: False}
    assert "fpe_mode: mixed" in report.summary()


def test_mixed_modes_across_columns_are_counted_per_column(stub_pyffx):
    df = pd.DataFrame({"a": ["123-45-6789"], "b": ["123-45-6789"]})
    rules = (
        MaskingRule(name="fa", columns=("a",), strategy="fpe", key="k"),
        MaskingRule(name="sb", columns=("b",), strategy="surrogate"),
    )
    _, report = anonymize(df, rules=rules)
    assert report.metadata == {
        "fpe_mode": "mixed",
        "fpe_modes": {"a": {"crypto_fpe": 1}, "b": {_SURROGATE_MODE: 1}},
    }


def test_crypto_fpe_single_mode_reports_mode_string(stub_pyffx):
    df = pd.DataFrame({"x": ["123-45-6789", "555-12-3456"]})
    rule = MaskingRule(name="r", columns=("x",), strategy="fpe", key="k")
    _, report = anonymize(df, rules=(rule,))
    assert report.metadata == {"fpe_mode": "crypto_fpe"}
    assert [e.reversible for e in report.events] == [False, False]


def test_reversible_tokenize_events_stay_reversible():
    df = pd.DataFrame({"email": ["a@b.com", "c@d.com"]})
    rule = MaskingRule(name="t", columns=("email",), strategy="tokenize", reversible=True, key=KEY)
    _, report = anonymize(df, rules=(rule,))
    assert [e.reversible for e in report.events] == [True, True]
    assert "fpe_mode" not in report.metadata


def test_single_mode_report_is_unchanged(no_pyffx):
    df = pd.DataFrame({"ssn": ["123-45-6789", "987-65-4321"], "acct": ["1234567890", None]})
    rules = (
        MaskingRule(
            name="s", columns=("ssn",), strategy="surrogate", preserve_format=True, visible=4
        ),
        MaskingRule(name="f", columns=("acct",), strategy="fpe", key="K", preserve_format=True),
        MaskingRule(name="d", columns=("ssn",), strategy="drop"),
    )
    _, report = anonymize(df, rules=rules)
    assert report.metadata == {"fpe_mode": _SURROGATE_MODE}
    assert report.summary().endswith(f"fpe_mode: {_SURROGATE_MODE}")
    assert all(e.reversible is False for e in report.events)


def test_rules_without_fpe_add_no_fpe_metadata():
    df = pd.DataFrame({"a": ["x"], "b": ["y"]})
    rules = (
        MaskingRule(name="r", columns=("a",), strategy="redact"),
        MaskingRule(name="d", columns=("b",), strategy="drop"),
    )
    _, report = anonymize(df, rules=rules)
    assert report.metadata == {}
