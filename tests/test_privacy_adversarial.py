"""Adversarial privacy suite (audit deliverable).

Two kinds of tests live here, on purpose:

1. **Leak checks** — planted PII must never appear in the copilot's
   ``model_context`` (the only payload a provider sees), in any privacy mode.
2. **Accepted-limitation pins** — the regex detector's documented blind spots
   (see docs/threat-model.md) asserted as *current behavior*. If detection
   improves, these fail and the threat model must be updated in the same PR —
   that is the point: docs and behavior stay in sync.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata.enterprise import detect_pii
from freshdata.experimental.ai_copilot import analyze_dataset

OBFUSCATED_EMAILS = ["john[at]corp[dot]com", "john (at) corp dot com", "john＠corp.com"]
INTL_PHONES = ["+44 20 7946 0958", "0049-30-123456", "(02) 9374 4000"]
HOMOGLYPH_EMAIL = "jоhn@corp.com"  # Cyrillic о
RTL_NAME = "דוד לוי"  # Hebrew name
FREE_TEXT_PII = "Call Miklós O'Hara-Kovács at +36 30 123 4567 re: acct 4111-1111-1111-1111"
EVIL_COLUMNS = ["=cmd()", "'; DROP TABLE users;--", "<script>alert(1)</script>"]


def _adversarial_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "contact": OBFUSCATED_EMAILS,
            "phone_intl": INTL_PHONES,
            "who": [HOMOGLYPH_EMAIL, RTL_NAME, "plain name"],
            "notes": [FREE_TEXT_PII, "ok", "fine"],
            "age": [34, 29, 61],
            "salary": [91000.0, 45500.0, 130250.0],
        }
    )


def _blob(report) -> str:
    return json.dumps(report.model_context, default=str)


# --------------------------------------------------------------------------- #
# 1-4. Leak checks: nothing string-typed reaches the model context raw
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("privacy", ["mask_pii_before_reasoning", "schema_only"])
def test_no_adversarial_string_reaches_model_context(privacy):
    report = analyze_dataset(_adversarial_df(), privacy=privacy)
    blob = _blob(report)
    planted = [*OBFUSCATED_EMAILS, *INTL_PHONES, HOMOGLYPH_EMAIL, RTL_NAME,
               "Miklós", "4111-1111-1111-1111"]
    for value in planted:
        assert value not in blob, f"leaked {value!r} in privacy={privacy}"


def test_prompt_never_contains_planted_strings():
    prompts: list[str] = []

    def provider(prompt: str) -> str:
        prompts.append(prompt)
        return "narrative"

    with pytest.warns(FutureWarning):  # provider hooks are experimental
        analyze_dataset(_adversarial_df(), provider=provider)
    assert prompts, "provider was never called"
    for value in [*OBFUSCATED_EMAILS, HOMOGLYPH_EMAIL, "Miklós"]:
        assert value not in prompts[0]


# --------------------------------------------------------------------------- #
# 5. Quasi-identifiers: numeric residue is REAL and documented, not hidden
# --------------------------------------------------------------------------- #


def test_numeric_quasi_identifiers_pass_through_in_default_mode():
    """Documented residual risk (docs/threat-model.md boundary 3): exact
    numeric values ARE visible in default-mode samples. If this ever changes,
    update the threat model and the analyze_dataset docstring together."""
    report = analyze_dataset(_adversarial_df())
    blob = _blob(report)
    assert "91000.0" in blob or "91000" in blob


def test_schema_only_hides_numeric_quasi_identifiers():
    report = analyze_dataset(_adversarial_df(), privacy="schema_only")
    assert "sample_rows_masked" not in report.model_context
    assert "91000" not in _blob(report)


# --------------------------------------------------------------------------- #
# 6. Malicious column names: never crash, labels sanitized in csv exports
# --------------------------------------------------------------------------- #


def test_malicious_column_names_do_not_crash_anything(tmp_path):
    df = pd.DataFrame({name: ["a", "b"] for name in EVIL_COLUMNS})
    out = fd.clean(df, verbose=False)
    assert len(out.columns) == len(EVIL_COLUMNS)
    detect_pii(df)
    report = analyze_dataset(df)
    assert report.summary


def test_formula_column_labels_sanitized_end_to_end(tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"=SUM(A1)": ["x", "y"], "safe": ["=1+1", "ok"]}).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    fd.clean_csv(src, output_path=out, sanitize_formulas=True, column_names=False)
    text = out.read_text()
    assert "'=1+1" in text
    assert not any(line.startswith("=") for line in text.splitlines())


# --------------------------------------------------------------------------- #
# 7. Detection-limitation pins (docs/threat-model.md boundary 5)
# --------------------------------------------------------------------------- #


def test_regex_detector_misses_obfuscated_emails_as_documented():
    scan = detect_pii(pd.DataFrame({"contact": OBFUSCATED_EMAILS}))
    hit_columns = {e["metadata"]["column"] for e in scan.to_dict()["entities"]}
    # Accepted limitation: obfuscated addresses are invisible to the regex
    # detector. The copilot compensates by masking ALL string-like columns.
    assert "contact" not in hit_columns


def test_regex_detector_misses_free_text_names_as_documented():
    scan = detect_pii(pd.DataFrame({"notes": ["Miklós O'Hara-Kovács lives here"]}))
    assert not any(
        e["entity_type"] in ("PERSON", "NAME") for e in scan.to_dict()["entities"]
    )


# --------------------------------------------------------------------------- #
# 8. Provider failure: fail closed for the report, honestly recorded
# --------------------------------------------------------------------------- #


def test_provider_exception_fails_closed():
    def bad_provider(prompt: str) -> str:
        raise RuntimeError("upstream 500")

    with pytest.warns(FutureWarning):
        report = analyze_dataset(_adversarial_df(), provider=bad_provider)
    assert report.narrative is None
    assert "RuntimeError" in report.audit["provider_error"]
    assert report.audit["engine"] == "deterministic-local"
    assert report.cleaning_plan.steps  # deterministic report intact
