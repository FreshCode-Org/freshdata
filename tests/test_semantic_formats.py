"""Phase-2 deterministic format experts: email, phone (IN), reference lists."""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd
from freshdata.semantic.formats import (
    EmailExpert,
    PhoneExpert,
    ReferenceExpert,
    normalize_phone_in,
    repair_email,
)
from freshdata.semantic.memory import build_semantic_metadata
from freshdata.semantic.types import SemanticColumnInfo


def _info(name="col", **overrides) -> SemanticColumnInfo:
    base = {
        "name": name, "role": "categorical", "n_nonnull": 4, "nunique": 4,
        "high_cardinality": False, "preserve": False, "free_text": False,
        "numeric_like": False, "boolean_like": False, "money_like": False,
        "unit_like": False, "identifier_like": False,
    }
    base.update(overrides)
    return SemanticColumnInfo(**base)


# --------------------------------------------------------------------------- #
# repair_email / EmailExpert
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" Bob@GMAIL.COM ", "Bob@gmail.com"),
        ("alice @ example.com", "alice@example.com"),
        ("bob@@gmail.com", "bob@gmail.com"),
        ("<carol@x.org>", "carol@x.org"),
        ("'dave@y.io',", "dave@y.io"),
        ("erin@MiXeD.Co.IN", "erin@mixed.co.in"),
    ],
)
def test_repair_email_safe_cases(raw, expected):
    repaired = repair_email(raw)
    assert repaired is not None and repaired[0] == expected


@pytest.mark.parametrize(
    "raw",
    [
        "bob[at]gmail.com",
        "asdf",
        "bob@gmail",          # no TLD
        "bob@@@gmail.com",    # triple @
        "admin@localhost",    # no dot-TLD
        "a b@x.com",          # whitespace inside local part
    ],
)
def test_repair_email_refuses_ambiguous(raw):
    assert repair_email(raw) is None


def test_email_expert_proposals_and_flags():
    info = _info("email_addr", semantic_type="email", email_like=True)
    series = pd.Series(
        [" Bob@GMAIL.COM ", "bob[at]gmail.com", "fine@ok.com", "fine@ok.com"]
    )
    proposals = EmailExpert().propose(series, info)
    by_raw = {p.raw_value: p for p in proposals}
    assert "fine@ok.com" not in by_raw  # valid values untouched
    good = by_raw[" Bob@GMAIL.COM "]
    assert good.proposed_value == "Bob@gmail.com"
    assert good.confidence >= 0.95 and good.risk == "low"
    flagged = by_raw["bob[at]gmail.com"]
    assert flagged.proposed_value is None and flagged.risk == "high"
    meta_keys = {"issue_type", "raw_value", "proposed_value", "expert"}
    metadata = build_semantic_metadata(good, info)
    assert meta_keys <= set(metadata)
    assert metadata["issue_type"] == "email_format" and metadata["expert"] == "email"


def test_email_expert_distinct_values_only():
    info = _info("email_addr", semantic_type="email", email_like=True)
    series = pd.Series(["a@@b.com"] * 500)
    proposals = EmailExpert().propose(series, info)
    assert len(proposals) == 1 and proposals[0].count == 500


def test_email_end_to_end_auto():
    df = pd.DataFrame({"email_addr": ["x @ y.com", "ok@ok.com", "junk"]})
    out, report = fd.clean(
        df, context="Emails must be valid.", semantic_mode="auto",
        return_report=True, verbose=False,
    )
    assert out["email_addr"].tolist() == ["x@y.com", "ok@ok.com", "junk"]
    semantic = [a for a in report if a.step == "semantic"]
    assert any(a.status == "automatic" for a in semantic)
    assert any(a.status in ("skipped", "suggested") for a in semantic)  # "junk"


def test_protected_email_column_not_modified():
    df = pd.DataFrame({"email_addr": ["x @ y.com", "ok@ok.com"]})
    out = fd.clean(
        df,
        semantic_mode="auto",
        semantic_context={"columns": {"email_addr": {"semantic_type": "email",
                                                     "mutable": False}}},
        verbose=False,
    )
    assert out["email_addr"].equals(df["email_addr"])


# --------------------------------------------------------------------------- #
# normalize_phone_in / PhoneExpert
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("9876543210", "+919876543210"),
        ("98765 43210", "+919876543210"),
        ("09876543210", "+919876543210"),
        ("+91 98765 43210", "+919876543210"),
        ("91-9876543210", "+919876543210"),
        ("(98765) 43210", "+919876543210"),
    ],
)
def test_normalize_phone_in_safe(raw, expected):
    assert normalize_phone_in(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "12345",
        "999999999999999",
        "001234567890",
        "98765abc210",
        "5876543210",     # invalid mobile prefix
        "+929876543210",  # wrong country code
    ],
)
def test_normalize_phone_in_refuses(raw):
    assert normalize_phone_in(raw) is None


def test_phone_expert_requires_region():
    no_region = _info("mobile", semantic_type="phone", phone_like=True)
    assert not PhoneExpert().applies(no_region)
    with_region = _info("mobile", semantic_type="phone", phone_like=True, region="IN")
    assert PhoneExpert().applies(with_region)


def test_phone_expert_proposals_flags_and_region_metadata():
    info = _info("mobile", semantic_type="phone", phone_like=True, region="IN")
    series = pd.Series(["98765 43210", "12345", "+919876543210"])
    proposals = PhoneExpert().propose(series, info)
    by_raw = {p.raw_value: p for p in proposals}
    assert "+919876543210" not in by_raw  # already canonical
    ok = by_raw["98765 43210"]
    assert ok.proposed_value == "+919876543210" and ok.risk == "low"
    bad = by_raw["12345"]
    assert bad.proposed_value is None and bad.risk == "high"
    assert build_semantic_metadata(ok, info)["region"] == "IN"


def test_phone_end_to_end_from_context():
    df = pd.DataFrame({"mobile": ["09876543210", "98765 43210", "12345"]})
    out, report = fd.clean(
        df, context="Phone numbers are Indian.", semantic_mode="auto",
        return_report=True, verbose=False,
    )
    assert out["mobile"].tolist() == ["+919876543210", "+919876543210", "12345"]
    assert str(out["mobile"].dtype) == "object"  # canonical form stays text


def test_protected_phone_column_not_modified():
    df = pd.DataFrame({"mobile": ["98765 43210"]})
    out = fd.clean(
        df,
        semantic_mode="auto",
        semantic_context={"columns": {"mobile": {"semantic_type": "phone",
                                                 "region": "IN",
                                                 "mutable": False}}},
        verbose=False,
    )
    assert out["mobile"].equals(df["mobile"])


# --------------------------------------------------------------------------- #
# ReferenceExpert
# --------------------------------------------------------------------------- #

_ALLOWED = ("active", "inactive", "pending")


def _ref_info(**overrides):
    return _info("status", allowed_values=_ALLOWED, **overrides)


def test_reference_exact_values_untouched():
    proposals = ReferenceExpert().propose(
        pd.Series(["active", "pending"]), _ref_info()
    )
    assert proposals == []


def test_reference_case_whitespace_and_punctuation_repairs():
    series = pd.Series([" Active ", "INACTIVE", "pend-ing", "pend_ing"])
    proposals = ReferenceExpert().propose(series, _ref_info())
    mapping = {p.raw_value: p.proposed_value for p in proposals}
    assert mapping == {
        " Active ": "active",
        "INACTIVE": "inactive",
        "pend-ing": "pending",
        "pend_ing": "pending",
    }
    assert all(p.risk == "low" and p.confidence >= 0.95 for p in proposals)
    assert all(
        any("allowed values" in e.detail for e in p.evidence) for p in proposals
    )


def test_reference_fuzzy_typo_suggested_not_auto():
    proposals = ReferenceExpert().propose(pd.Series(["actve"]), _ref_info())
    (p,) = proposals
    assert p.proposed_value == "active"
    assert p.confidence < 0.95  # below every auto threshold
    df = pd.DataFrame({"status": ["actve", "active", "pending"]})
    out = fd.clean(
        df, semantic_mode="auto",
        semantic_context={"columns": {"status": {"allowed_values": list(_ALLOWED)}}},
        verbose=False,
    )
    assert out["status"].tolist()[0] == "actve"  # suggested, not applied


def test_reference_ambiguity_blocked():
    info = _info("grade", allowed_values=("AB", "AC"))
    proposals = ReferenceExpert().propose(pd.Series(["AD"]), info)
    (p,) = proposals
    assert p.proposed_value is None and p.risk == "high"
    assert "ambiguous" in p.rationale or "no allowed value" in p.rationale


def test_reference_unknown_value_flagged():
    proposals = ReferenceExpert().propose(pd.Series(["zzzzzz"]), _ref_info())
    (p,) = proposals
    assert p.proposed_value is None and p.risk == "high"
