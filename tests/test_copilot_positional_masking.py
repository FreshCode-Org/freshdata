"""Copilot sample masking works by column position, whatever the label type.

Regression tests for raw sample values reaching the provider prompt when
column labels are not ``str``: masking rules and the declared mask set were
matched by ``str(label)``, so declared ``must_mask`` / ``sensitive_columns``
entries with int, float or tuple labels were dropped from the mask set.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import pytest

from freshdata.enterprise import privacy as privacy_module
from freshdata.experimental import ai_copilot
from freshdata.experimental.ai_copilot import analyze_dataset

NAMES = ("Alice Johnson", "Bob Smith")
SSNS = ("123456789", "987654321")


def _run_with_prompt(frame: pd.DataFrame, **kwargs):
    prompts: list[str] = []

    def provider(prompt: str) -> str:
        prompts.append(prompt)
        return "ok"

    with pytest.warns(FutureWarning, match="experimental"):
        report = analyze_dataset(frame, provider=provider, **kwargs)
    assert len(prompts) == 1
    return report, prompts[0]


def _assert_no_raw(report, prompt: str, raws) -> None:
    rendered = report._repr_html_()
    sinks = {
        "prompt": prompt,
        "model_context": json.dumps(report.model_context, default=str, ensure_ascii=False),
        "to_json": report.to_json(),
        "html": rendered,
        "html_unescaped": html.unescape(rendered),
        "str": str(report),
    }
    for sink, text in sinks.items():
        leaked = [raw for raw in raws if raw in text]
        assert not leaked, f"raw value(s) {leaked} reached {sink}"


def _frame(labels) -> pd.DataFrame:
    columns = pd.Index(list(labels), tupleize_cols=False)
    return pd.DataFrame(
        [[NAMES[0], int(SSNS[0])], [NAMES[1], int(SSNS[1])]], columns=columns
    )


def test_advisory_poc_int_labels_default_mode() -> None:
    df = pd.DataFrame({0: list(NAMES), 1: [int(s) for s in SSNS]})
    report, prompt = _run_with_prompt(df)
    assert [n for n in NAMES if n in prompt] == []
    assert report.audit["sample_masked_columns"] == ["0"]
    _assert_no_raw(report, prompt, NAMES)


def test_advisory_poc_int_label_must_mask_and_sensitive_columns() -> None:
    df = pd.DataFrame({0: list(NAMES), 1: [int(s) for s in SSNS]})
    report, prompt = _run_with_prompt(
        df, context_policy={1: "must_mask"}, sensitive_columns=[1]
    )
    assert "123456789" not in prompt
    assert report.audit["sample_masked_columns"] == ["0", "1"]
    assert "1" in report.audit["masked_columns"]
    _assert_no_raw(report, prompt, NAMES + SSNS)


@pytest.mark.parametrize(
    "labels",
    [
        (0, 1),
        (1.5, 2.5),
        (("person", "name"), ("person", "ssn")),
    ],
    ids=["int", "float", "tuple"],
)
@pytest.mark.parametrize("declare", ["sensitive_columns", "must_mask"])
def test_declared_numeric_column_is_masked_for_any_label_type(labels, declare) -> None:
    df = _frame(labels)
    ssn_label = labels[1]
    kwargs = (
        {"sensitive_columns": [ssn_label]}
        if declare == "sensitive_columns"
        else {"context_policy": {ssn_label: "must_mask"}}
    )
    report, prompt = _run_with_prompt(df, **kwargs)
    assert report.audit["sample_masked_columns"] == sorted(str(label) for label in labels)
    rows = report.model_context["sample_rows_masked"]
    assert [sorted(row) for row in rows] == [sorted(str(label) for label in labels)] * 2
    _assert_no_raw(report, prompt, NAMES + SSNS)


def test_multiindex_columns_are_masked_positionally() -> None:
    columns = pd.MultiIndex.from_tuples([("person", "name"), ("person", "ssn")])
    df = pd.DataFrame([[NAMES[0], 123456789], [NAMES[1], 987654321]], columns=columns)
    report, prompt = _run_with_prompt(df, sensitive_columns=[("person", "ssn")])
    assert len(report.audit["sample_masked_columns"]) == 2
    _assert_no_raw(report, prompt, NAMES + SSNS)


def test_string_form_of_a_non_str_label_is_accepted() -> None:
    df = pd.DataFrame({0: list(NAMES), 1: [int(s) for s in SSNS]})
    report, prompt = _run_with_prompt(df, sensitive_columns=["1"])
    assert report.audit["sample_masked_columns"] == ["0", "1"]
    _assert_no_raw(report, prompt, SSNS)


def test_allow_unmasked_columns_matches_non_str_labels() -> None:
    df = pd.DataFrame({0: ["Paris", "Lyon"], 1: [1, 2]})
    for name in (0, "0"):
        report = analyze_dataset(df, allow_unmasked_columns=[name])
        assert report.audit["sample_masked_columns"] == []
        assert "Paris" in json.dumps(report.model_context)


@pytest.mark.parametrize("privacy", ["mask_pii_before_reasoning", "schema_only"])
def test_labels_colliding_as_strings_raise(privacy) -> None:
    df = pd.DataFrame([[NAMES[0], NAMES[1]]], columns=pd.Index([0, "0"], dtype=object))
    with pytest.raises(ValueError, match="unique when converted to str") as info:
        analyze_dataset(df, privacy=privacy)
    assert not [n for n in NAMES if n in str(info.value)]


def test_unknown_sensitive_columns_raise() -> None:
    df = pd.DataFrame({"a": [1, 2], 3: ["x", "y"]})
    with pytest.raises(ValueError, match="sensitive_columns contains unknown column"):
        analyze_dataset(df, sensitive_columns=["nope"])
    with pytest.raises(ValueError, match="sensitive_columns contains unknown column"):
        analyze_dataset(df, sensitive_columns=[4])
    analyze_dataset(df, sensitive_columns=[3, "a"])


def test_masking_that_changes_nothing_fails_closed(monkeypatch) -> None:
    def no_op_anonymize(df, **kwargs):
        return df.copy()

    monkeypatch.setattr(ai_copilot, "anonymize", no_op_anonymize)
    df = pd.DataFrame({0: list(NAMES), 1: [int(s) for s in SSNS]})
    with pytest.raises(RuntimeError, match="failed closed") as info:
        analyze_dataset(df, sensitive_columns=[1])
    message = str(info.value)
    assert not [raw for raw in NAMES + SSNS if raw in message]


def test_masking_that_skips_one_column_fails_closed(monkeypatch) -> None:
    real_anonymize = ai_copilot.anonymize

    def skip_last_rule(df, *, rules=(), **kwargs):
        if rules:
            rules = rules[:-1]
            if not rules:
                return df.copy()
        return real_anonymize(df, rules=rules, **kwargs)

    monkeypatch.setattr(ai_copilot, "anonymize", skip_last_rule)
    df = pd.DataFrame({"name": list(NAMES), "ssn": [int(s) for s in SSNS]})
    with pytest.raises(RuntimeError, match="position 1"):
        analyze_dataset(df, sensitive_columns=["ssn"])


def test_missing_values_in_masked_columns_do_not_trip_the_check() -> None:
    df = pd.DataFrame(
        {
            0: [NAMES[0], None, NAMES[1]],
            1: pd.Series([123456789, None, 987654321], dtype="Int64"),
            2: pd.to_datetime(["1980-02-03", None, "1975-06-07"]),
        }
    )
    report = analyze_dataset(df, sensitive_columns=[1])
    rows = report.model_context["sample_rows_masked"]
    assert [row["1"] is None for row in rows] == [False, True, False]
    assert [row["2"] is None for row in rows] == [False, True, False]


def test_detection_never_rewrites_a_hash_token(monkeypatch) -> None:
    # A token that happens to look like a Luhn-valid card number must stay a
    # token: free-text PII detection runs only on unmasked columns.
    monkeypatch.setattr(
        privacy_module, "_hash_value", lambda value, salt, length: "4111111111111111"
    )
    df = pd.DataFrame({"note": list(NAMES), "v": [1, 2]})
    report = analyze_dataset(df)
    assert [row["note"] for row in report.model_context["sample_rows_masked"]] == [
        "4111111111111111"
    ] * 2


def test_detection_pass_covers_exactly_the_unmasked_columns(monkeypatch) -> None:
    real_anonymize = ai_copilot.anonymize
    calls: list[tuple[list[str], int, bool]] = []

    def spy(df, *, rules=(), detection_config=None, **kwargs):
        calls.append(([str(c) for c in df.columns], len(rules), detection_config is not None))
        return real_anonymize(df, rules=rules, detection_config=detection_config, **kwargs)

    monkeypatch.setattr(ai_copilot, "anonymize", spy)
    df = pd.DataFrame({"city": ["Paris", "Lyon"], "note": list(NAMES), "v": [1, 2]})
    analyze_dataset(df, allow_unmasked_columns=["city"])
    assert calls == [(["__c0", "__c1", "__c2"], 1, False), (["__c0", "__c2"], 0, True)]
