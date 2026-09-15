"""Privacy policy classification: value coverage, rule priority, keys and labels.

* #246: classification reads every distinct value, not the first 200 cells.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from freshdata.enterprise import (
    PIIDetectionConfig,
    PrivacyPolicy,
    PrivacyRule,
    apply_privacy_policy,
    classify_columns,
    load_compliance_pack,
    privacy_policy,
)

KEY = "unit-test-key"


@pytest.fixture
def detect_calls(monkeypatch):
    """Count the detect_in_text calls made during classification."""
    calls: list[int] = []
    real = privacy_policy.detect_in_text

    def counting(text, **kwargs):
        calls.append(len(text))
        return real(text, **kwargs)

    monkeypatch.setattr(privacy_policy, "detect_in_text", counting)
    return calls


def _distinct_words(n: int, width: int) -> list[str]:
    """*n* distinct lowercase words of *width* characters that match no detector."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    words = []
    for i in range(n):
        stem = "".join(letters[(i // 26**k) % 26] for k in range(3))
        words.append(stem + "q" * (width - len(stem)))
    return words


# --------------------------------------------------------------------------
# #246: classify from all distinct values
# --------------------------------------------------------------------------

_PII = "patient email jane.roe@clinic.org ssn 123-45-6789"


@pytest.mark.parametrize("n_before", [10, 250, 10_000])
def test_issue_246_late_pii_is_classified(n_before):
    pol = PrivacyPolicy(packs=(load_compliance_pack("hipaa"),), jurisdiction="US")
    df = pd.DataFrame({"notes": ["follow-up scheduled"] * n_before + [_PII]})
    out, rep = apply_privacy_policy(df, pol)
    assert out["notes"].iloc[-1] == "patient email <EMAIL> ssn <SSN>"
    assert sorted(rep.classifications) == ["notes"]
    assert rep.metadata["classification_values_scanned"] == {"notes": 2}


def test_late_value_regex_match_is_classified():
    rule = PrivacyRule(id="ticket", action="redact", value_regexes=(r"\bTKT-\d{6}\b",))
    df = pd.DataFrame({"ref": [f"note {i}" for i in range(5_000)] + ["TKT-123456"]})
    cls = classify_columns(df, PrivacyPolicy(rules=(rule,)))
    assert cls["ref"].rule is rule
    assert cls["ref"].matched_by == "regex"


def test_late_context_keyword_is_classified():
    rule = PrivacyRule(id="conf", action="quarantine", context=("confidential",))
    df = pd.DataFrame({"memo": [f"memo {i}" for i in range(5_000)] + ["CONFIDENTIAL draft"]})
    out, rep = apply_privacy_policy(df, PrivacyPolicy(rules=(rule,)))
    assert rep.classifications["memo"]["matched_by"] == "context"
    assert set(out["memo"]) == {"<QUARANTINED>"}


def test_late_luhn_value_satisfies_requires_luhn():
    rule = PrivacyRule(id="pan", action="redact", requires_luhn=True, value_regexes=(r"\d{4}",))
    head = ["order 1234 5678 9012 3456"] * 300  # 16 digits, fails Luhn
    policy = PrivacyPolicy(rules=(rule,))
    assert "ref" not in classify_columns(pd.DataFrame({"ref": head}), policy)
    cls = classify_columns(pd.DataFrame({"ref": head + ["4111 1111 1111 1111"]}), policy)
    assert cls["ref"].rule is rule


def test_detect_calls_match_chunk_count(detect_calls):
    values = _distinct_words(640, 999)
    chars = len("\n".join(values))
    cls = classify_columns(pd.DataFrame({"words": values}), PrivacyPolicy())
    assert cls == {}
    assert len(detect_calls) == math.ceil(chars / privacy_policy._CLASSIFY_CHUNK_CHARS)
    assert max(detect_calls) <= privacy_policy._CLASSIFY_CHUNK_CHARS


def test_repeated_values_are_read_once(detect_calls):
    values = _distinct_words(640, 999)
    classify_columns(pd.DataFrame({"words": values * 5}), PrivacyPolicy())
    assert len(detect_calls) == 10
    detect_calls.clear()
    _, rep = apply_privacy_policy(pd.DataFrame({"same": ["repeated"] * 10_000}), PrivacyPolicy())
    assert len(detect_calls) == 1
    assert rep.metadata["classification_values_scanned"] == {"same": 1}


def test_detection_stops_once_every_entity_type_is_found(detect_calls):
    values = ["mail a@b.com", *_distinct_words(200, 999)]
    policy = PrivacyPolicy(detection_config=PIIDetectionConfig(entities=("EMAIL",)))
    cls = classify_columns(pd.DataFrame({"notes": values}), policy)
    assert cls["notes"].classification == "detected PII: EMAIL"
    assert len(detect_calls) == 1


def test_oversized_value_forms_its_own_chunk():
    limit = privacy_policy._CLASSIFY_CHUNK_CHARS
    big = "q" * (limit + 10)
    chunks = privacy_policy._pack_chunks(["a", big, "b", "c"], limit)
    assert chunks == ["a", big, "b\nc"]
    assert privacy_policy._pack_chunks([], limit) == []


def test_values_scanned_ignores_missing_cells():
    df = pd.DataFrame({"s": pd.array(["x", "x", None, "y"], dtype="string"), "n": [1, 1, 2, 2]})
    _, rep = apply_privacy_policy(df, PrivacyPolicy())
    assert rep.metadata["classification_values_scanned"] == {"s": 2, "n": 2}


def test_unhashable_cells_are_classified():
    df = pd.DataFrame({"tags": [["a"], ["a"], ["mail a@b.com"]]})
    _, rep = apply_privacy_policy(df, PrivacyPolicy())
    assert rep.classifications["tags"]["classification"] == "detected PII: EMAIL"
    assert rep.metadata["classification_values_scanned"] == {"tags": 2}
