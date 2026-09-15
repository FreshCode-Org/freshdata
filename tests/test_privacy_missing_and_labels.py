"""Privacy engine correctness: missing values, categorical k-anonymity, labels and metadata.

* #243: ``pd.NA``/``NaT`` cells stay missing under every strategy and action.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from freshdata.enterprise import (
    MaskingRule,
    PIIDetectionConfig,
    PrivacyPolicy,
    PrivacyRule,
    anonymize,
    apply_privacy_policy,
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
