"""Copilot TruthBench sinks are deterministic and the canary scanner is precise.

A TruthBench release gate once reported a digit-only phone canary in the
copilot ``rendered.html`` sink intermittently. Two things could produce that:
the digit-only scanner joining digits from different tokens of one long HTML
leaf, and masked sample tokens that differ on every run. These tests pin both.
"""

from __future__ import annotations

import random
import re
import secrets
from datetime import datetime, timezone

import pandas as pd
import pytest
from benchmarks.truthbench.fixtures import DOMAINS, build_fixture
from benchmarks.truthbench.privacy import SinkScanner
from benchmarks.truthbench.surfaces.copilot import CopilotAdapter

from freshdata.enterprise import privacy as privacy_module
from freshdata.experimental import ai_copilot
from freshdata.experimental.ai_copilot import analyze_dataset

PHONE = "555-0110"
PROSE = " ".join(["the quick brown fox jumps over the lazy dog"] * 40)


def _education():
    return build_fixture("education")


def _phone_canary_id(fixture) -> str:
    return next(k for k, v in sorted(fixture.pii_canaries.items()) if v == PHONE)


def _digits(text: str) -> str:
    return "".join(ch for ch in text if ch.isdigit())


# (a) scanner precision on long leaves ----------------------------------------


def test_contiguous_canary_digits_inside_a_hex_token_flag() -> None:
    fixture = _education()
    leaf = f"<td>{PROSE} token 3fa5550110c9 {PROSE}</td>"
    # One numeric run carries every canary digit, so this is a genuine match
    # (the scanner reports whichever normalised variant matches first).
    assert any(_digits(PHONE) in run for run in re.findall(r"\d+", leaf))
    leaks = CopilotAdapter().scanner_for(fixture).scan({"html": leaf})
    assert _phone_canary_id(fixture) in {x.canary_id for x in leaks}


def test_canary_digits_split_by_letters_in_a_long_leaf_do_not_flag() -> None:
    fixture = _education()
    leaf = f"<td>{PROSE} 3fa555b01c10 {PROSE}</td>"
    # The concatenated digit stream of the leaf does contain the canary;
    # only digits within one numeric run may match it.
    assert _digits(PHONE) in _digits(leaf)
    assert CopilotAdapter().scanner_for(fixture).scan({"html": leaf}) == []


# (b) a token carrying canary digits is caught every time ---------------------


def _all_fixture_scanner() -> SinkScanner:
    canaries: dict[str, object] = {}
    for domain in DOMAINS:
        canaries.update(build_fixture(domain).pii_canaries)
    return SinkScanner.from_canaries(canaries, key=b"copilot-determinism-test")


def test_hash_token_with_canary_digits_is_flagged_deterministically(monkeypatch) -> None:
    monkeypatch.setattr(
        privacy_module, "_hash_value", lambda value, salt, length: "ab5550110cdef012"
    )
    scanner = _all_fixture_scanner()
    phone_id = _phone_canary_id(_education())
    runs = []
    for _ in range(2):
        observation = CopilotAdapter().observe(build_fixture("crm"), {})
        assert observation.unexpected_exception is None
        leaks = scanner.scan(observation.audit_sinks)
        runs.append(sorted((x.canary_id, x.variant, x.path) for x in leaks))
    assert runs[0] == runs[1]
    paths = {path for canary, _, path in runs[0] if canary == phone_id}
    assert "$.rendered.html" in paths


# (c) pinned mask_salt: rendered sinks do not depend on process randomness ----


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return datetime(2026, 1, 2, 3, 4, 5, tzinfo=tz or timezone.utc)


def _seed_randomness(monkeypatch, seed: int) -> None:
    rng = random.Random(seed)

    def token_bytes(nbytes: int | None = None) -> bytes:
        n = 32 if nbytes is None else nbytes
        return rng.getrandbits(8 * n).to_bytes(n, "big")

    monkeypatch.setattr(secrets, "token_bytes", token_bytes)
    monkeypatch.setattr(secrets, "token_hex", lambda nbytes=None: token_bytes(nbytes).hex())


def _rendered_across_seeds(monkeypatch, domain: str, seeds) -> None:
    monkeypatch.setattr(ai_copilot, "datetime", _FrozenDatetime)
    fixture = build_fixture(domain)
    scanner = CopilotAdapter().scanner_for(fixture)
    outputs = []
    for seed in seeds:
        _seed_randomness(monkeypatch, seed)
        observation = CopilotAdapter().observe(fixture, {})
        assert observation.unexpected_exception is None
        sinks = observation.audit_sinks
        assert scanner.scan(sinks) == []
        outputs.append((sinks["rendered"]["html"], sinks["prompt"], sinks["model_context"]))
    assert all(output == outputs[0] for output in outputs[1:])


def test_seeded_randomness_reaches_the_default_key(monkeypatch) -> None:
    monkeypatch.setattr(ai_copilot, "datetime", _FrozenDatetime)
    frame = _education().frame
    htmls = []
    for seed in (11, 12, 11):
        _seed_randomness(monkeypatch, seed)
        htmls.append(analyze_dataset(frame)._repr_html_())
    assert htmls[0] == htmls[2]
    assert htmls[0] != htmls[1]


def test_pinned_mask_salt_renders_identically_across_seeds(monkeypatch) -> None:
    _rendered_across_seeds(monkeypatch, "education", seeds=(0, 1, 2))


@pytest.mark.large
@pytest.mark.parametrize("domain", DOMAINS)
def test_pinned_mask_salt_renders_identically_across_many_seeds(monkeypatch, domain) -> None:
    _rendered_across_seeds(monkeypatch, domain, seeds=range(25))


# (d) label and dtype sweep ---------------------------------------------------


def _text_dtypes():
    dtypes = {"object": object, "string": "string", "categorical": "category"}
    try:
        import pyarrow as pa  # noqa: PLC0415

        dtypes["arrow-string"] = pd.ArrowDtype(pa.string())
    except ImportError:
        pass
    return dtypes


LABELS = {
    "int": (0, 1, 2, 3),
    "float": (0.5, 1.5, 2.5, 3.5),
    "tuple": (("g", "phone"), ("g", "email"), ("g", "notes"), ("g", "n")),
}


@pytest.mark.parametrize("labels", sorted(LABELS))
@pytest.mark.parametrize("dtype", sorted(_text_dtypes()))
def test_label_and_dtype_sweep_finds_no_leak(labels, dtype) -> None:
    fixture = _education()
    scanner = CopilotAdapter().scanner_for(fixture)
    values = sorted(v for v in fixture.pii_canaries.values() if isinstance(v, str))
    assert PHONE in values
    others = [v for v in values if v != PHONE][:2]
    names = LABELS[labels]
    frame = pd.DataFrame(
        {
            0: pd.Series([PHONE, "n/a", "unknown"], dtype=_text_dtypes()[dtype]),
            1: pd.Series([others[0], "none", "missing"], dtype=_text_dtypes()[dtype]),
            2: pd.Series([others[1], "blank", "empty"], dtype=_text_dtypes()[dtype]),
            3: [5550110, 1, 2],
        }
    )
    frame.columns = pd.Index(list(names), tupleize_cols=False)
    prompts: list[str] = []

    def provider(prompt: str) -> str:
        prompts.append(prompt)
        return "ok"

    with pytest.warns(FutureWarning, match="experimental"):
        report = analyze_dataset(
            frame,
            provider=provider,
            sensitive_columns=[names[3]],
            mask_salt="sweep-salt",
        )
    sinks = {
        "rendered": {"html": report._repr_html_()},
        "prompt": prompts[0],
        "model_context": report.model_context,
    }
    assert scanner.scan(sinks) == []
