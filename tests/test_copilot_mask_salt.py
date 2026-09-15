"""``analyze_dataset(mask_salt=...)``: reproducible or per-run sample tokens (#288)."""

from __future__ import annotations

import hashlib
import hmac
import html
import json

import pandas as pd
import pytest

from freshdata.enterprise.cleaner import _hash_value
from freshdata.experimental.ai_copilot import analyze_dataset

SALT = "Zq7-copilot-mask-salt-4f1e9b"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice Johnson", "Bob Smith"],
            "alias": ["Alice Johnson", "Carol White"],
            "v": [1, 2],
        }
    )


def _tokens(report) -> list[dict]:
    return report.model_context["sample_rows_masked"]


def test_default_runs_use_a_fresh_key_per_run() -> None:
    a, b = analyze_dataset(_frame()), analyze_dataset(_frame())
    assert _tokens(a) != _tokens(b)
    assert a.audit["model_context_sha256"] != b.audit["model_context_sha256"]
    assert a.audit["mask_salt_source"] == "per-run-random"


def test_same_mask_salt_reproduces_model_context_and_fingerprint() -> None:
    a = analyze_dataset(_frame(), mask_salt=SALT)
    b = analyze_dataset(_frame(), mask_salt=SALT)
    assert _tokens(a) == _tokens(b)
    assert a.model_context == b.model_context
    assert a.audit["model_context_sha256"] == b.audit["model_context_sha256"]
    assert a.audit["mask_salt_source"] == "caller"
    other = analyze_dataset(_frame(), mask_salt=SALT + "-other")
    assert other.audit["model_context_sha256"] != a.audit["model_context_sha256"]


def test_tokens_follow_the_documented_per_column_derivation() -> None:
    report = analyze_dataset(_frame(), mask_salt=SALT)
    for position, column in enumerate(["name", "alias"]):
        column_salt = hmac.new(
            SALT.encode("utf-8"), f"copilot-col:{position}".encode(), hashlib.sha256
        ).hexdigest()
        expected = [_hash_value(v, column_salt, 16) for v in _frame()[column]]
        assert [row[column] for row in _tokens(report)] == expected


def test_same_value_in_two_columns_gets_different_tokens() -> None:
    row = _tokens(analyze_dataset(_frame(), mask_salt=SALT))[0]
    assert row["name"] != row["alias"]


def test_mask_salt_never_appears_in_any_sink() -> None:
    prompts: list[str] = []

    def provider(prompt: str) -> str:
        prompts.append(prompt)
        return "ok"

    with pytest.warns(FutureWarning, match="experimental"):
        report = analyze_dataset(_frame(), mask_salt=SALT, provider=provider)
    rendered = report._repr_html_()
    sinks = {
        "prompt": prompts[0],
        "model_context": json.dumps(report.model_context, default=str),
        "audit": json.dumps(report.audit, default=str),
        "to_json": report.to_json(),
        "html": html.unescape(rendered),
        "str": str(report),
        "recommended_code": report.recommended_code,
    }
    for sink, text in sinks.items():
        assert SALT not in text, sink
        assert SALT.encode().hex() not in text, sink


def test_mask_salt_source_is_recorded_without_sample_rows() -> None:
    report = analyze_dataset(_frame(), privacy="schema_only", mask_salt=SALT)
    assert "sample_rows_masked" not in report.model_context
    assert report.audit["mask_salt_source"] == "caller"


@pytest.mark.parametrize(
    ("salt", "error"), [("", ValueError), (b"bytes", TypeError), (7, TypeError)]
)
def test_invalid_mask_salt_is_rejected(salt, error) -> None:
    with pytest.raises(error, match="mask_salt"):
        analyze_dataset(_frame(), mask_salt=salt)
