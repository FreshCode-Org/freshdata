from __future__ import annotations

import dataclasses
import html
import json
import re
import unicodedata
import urllib.parse

import pandas as pd
import pytest
from benchmarks.truthbench.exact import encode_typed
from benchmarks.truthbench.privacy import Leak, SinkScanner

CANARY = "tb.person+7@example.invalid"


def _url_quote(value: str) -> str:
    return urllib.parse.quote(value)


def _html_escape(value: str) -> str:
    return html.escape(value)


def _json_dump(value: str) -> str:
    return json.dumps(value)


def _mutators() -> list:
    return [
        lambda s: s,
        lambda s: s.casefold(),
        lambda s: re.sub(r"\s+", "", s),
        lambda s: re.sub(r"[^\w\s]", "", s),
        lambda s: re.sub(r"\D", "", s),
        _url_quote,
        _html_escape,
        lambda s: s.encode("utf-8"),
        lambda s: unicodedata.normalize("NFKC", s),
        lambda s: unicodedata.normalize("NFC", s),
        lambda s: unicodedata.normalize("NFD", s),
        lambda s: s.replace(".", "\u200b.\u200b"),
        _json_dump,
    ]


@pytest.mark.parametrize("mutate", _mutators())
def test_scanner_finds_every_normalized_variant(mutate):
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    leaks = scanner.scan({"report": [mutate(CANARY)]})
    assert [(leak.canary_id, leak.path) for leak in leaks] == [("crm-email", "$.report[0]")]
    assert CANARY not in repr(leaks)


def test_scan_nested_dataclass_pandas_and_bytes_without_raw_values():
    @dataclasses.dataclass
    class Envelope:
        exception_text: str
        frame: pd.DataFrame

    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    sink = Envelope("failed for " + CANARY, pd.DataFrame({"email": [CANARY.encode()]}))
    leaks = scanner.scan(sink)
    assert {leak.path for leak in leaks} == {
        "$.exception_text",
        "$.frame.email[0]",
    }
    assert all(CANARY not in repr(leak) for leak in leaks)


def test_redact_returns_digest_marker_and_self_test():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY}, run_id="run-1")
    redacted = scanner.redact({"email": CANARY, "nested": [CANARY]})
    assert CANARY not in repr(redacted)
    assert scanner.scan(redacted) == []
    assert scanner.self_test(redacted) is True
    with pytest.raises(AssertionError):
        scanner.self_test({"email": CANARY})


def test_scanner_accepts_exact_typed_redaction_without_scanning_digest():
    scanner = SinkScanner.from_canaries({"short": "7"}, run_id="run-1")
    typed = encode_typed("7", sensitive=True, digest_key=b"test-key").to_dict()
    assert scanner.scan(typed) == []


def test_mapping_keys_are_scanned_without_echoing_sensitive_key():
    scanner = SinkScanner.from_canaries({"short": "7"})
    leaks = scanner.scan({"7": "safe"})
    assert [(leak.canary_id, leak.path) for leak in leaks] == [("short", "$[<key>]")]
    assert "'7'" not in repr(leaks)


def test_arbitrary_mapping_key_stringification_is_sanitized():
    class SensitiveKey:
        def __str__(self):
            return CANARY

    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    sink = {SensitiveKey(): "safe"}
    assert scanner.scan(sink)[0].path == "$[<key>]"
    redacted = scanner.redact(sink)
    assert CANARY not in repr(redacted)


def test_forged_redaction_marker_is_not_trusted():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    forged = "[REDACTED:" + CANARY + "]"
    with pytest.raises(AssertionError):
        scanner.self_test(forged)


def test_redact_removes_sensitive_mapping_keys():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    redacted = scanner.redact({CANARY: "safe"})
    assert CANARY not in repr(redacted)
    assert scanner.scan(redacted) == []


def test_dataframe_and_series_labels_are_scanned_and_redacted():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    frame = pd.DataFrame({CANARY: ["safe"]}, index=[CANARY])
    series = pd.Series(["safe"], index=[CANARY], name=CANARY)
    assert scanner.scan(frame)
    assert scanner.scan(series)
    assert scanner.scan(pd.Index([CANARY]))
    assert scanner.scan(pd.Index(["safe"], name=CANARY))
    assert scanner.scan(pd.Series(name=CANARY))
    redacted_frame = scanner.redact(frame)
    redacted_series = scanner.redact(series)
    assert CANARY not in repr(redacted_frame)
    assert CANARY not in repr(redacted_series)


def test_digest_alias_matches_digest_for():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    assert scanner.digest(CANARY) == scanner.digest_for(CANARY)


def test_named_sink_scanners_cover_report_surfaces():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    sinks = {
        "exception_text": CANARY,
        "clean_report": {"coerced_cells": {"email": CANARY}},
        "actions": [{"metadata": CANARY}],
        "findings": [{"observed_value": CANARY}],
        "plan": {"json": CANARY},
        "reports": {"validation": CANARY, "copilot": CANARY},
        "generated_code": CANARY,
        "stdout": CANARY,
        "stderr": CANARY,
        "markdown": CANARY,
        "html": CANARY,
        "json": CANARY,
        "failure_artifacts": [CANARY],
    }
    leaks = scanner.scan_sinks(sinks)
    assert len(leaks) == 14
    assert all(isinstance(leak, Leak) for leak in leaks)


def test_multiindex_labels_and_names_are_scanned_with_sanitized_paths():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    columns = pd.MultiIndex.from_tuples([("group", CANARY)], names=["level", CANARY])
    index = pd.MultiIndex.from_tuples([("row", CANARY)], names=[CANARY, "kind"])
    frame = pd.DataFrame([["safe"]], columns=columns, index=index)
    series = pd.Series(["safe"], index=index, name=("series", CANARY))

    leaks = scanner.scan(frame)
    leaks += scanner.scan(series)
    assert leaks
    assert all(CANARY not in leak.path for leak in leaks)
    assert any("[<column>]" in leak.path for leak in leaks)
    assert any("[<index>]" in leak.path for leak in leaks)
    assert any("[<name>]" in leak.path for leak in leaks)


def test_multiindex_redaction_preserves_tuple_structure_and_names():
    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    columns = pd.MultiIndex.from_tuples([("group", CANARY)], names=["level", CANARY])
    index = pd.MultiIndex.from_tuples([("row", CANARY)], names=[CANARY, "kind"])
    frame = pd.DataFrame([[CANARY]], columns=columns, index=index)
    redacted = scanner.redact(frame)

    assert isinstance(redacted.columns, pd.MultiIndex)
    assert isinstance(redacted.index, pd.MultiIndex)
    assert all(isinstance(label, tuple) for label in redacted.columns)
    assert all(isinstance(label, tuple) for label in redacted.index)
    assert CANARY not in repr(redacted)
    assert scanner.scan(redacted) == []


def test_hostile_non_string_labels_are_sanitized_and_redacted():
    class HostileLabel:
        def __str__(self):
            return CANARY

        def __repr__(self):
            return CANARY

    scanner = SinkScanner.from_canaries({"crm-email": CANARY})
    hostile = HostileLabel()
    frame = pd.DataFrame([["safe"]], columns=pd.Index([hostile]), index=pd.Index([hostile]))
    leaks = scanner.scan(frame)
    assert leaks
    assert all(CANARY not in leak.path for leak in leaks)
    redacted = scanner.redact(frame)
    assert CANARY not in repr(redacted)
    assert scanner.scan(redacted) == []
