"""Golden corpus: 60 context texts snapshot to exact policy JSON.

Every text in ``tests/context/golden/*.txt`` must compile (against the fixed
``GOLDEN_SCHEMA``) to exactly the JSON in the sibling ``*.policy.json``.
Any diff is a reviewed change:

    FRESHDATA_REGEN_GOLDEN=1 pytest tests/context/test_context_golden.py

regenerates the snapshots; review the diff before committing.
"""

import json
import os
from pathlib import Path

import pytest

from freshdata.context import compile_context, parse_context, split_sentences

GOLDEN_DIR = Path(__file__).parent / "golden"

#: The one schema every golden text resolves against (post-normalization names).
GOLDEN_SCHEMA = [
    "cust_id",
    "full_name",
    "email_addr",
    "mobile",
    "age",
    "monthly_revenue",
    "city",
    "status",
    "signup_date",
    "order_id",
    "quantity",
    "price",
    "country",
    "gender",
    "zip_code",
]

_CASES = sorted(GOLDEN_DIR.glob("*.txt"))


def test_corpus_is_large_enough():
    assert len(_CASES) >= 60


@pytest.mark.parametrize("case", _CASES, ids=lambda p: p.stem)
def test_golden_policy_snapshot(case):
    text = case.read_text(encoding="utf-8")
    policy = compile_context(text, columns=GOLDEN_SCHEMA)
    got = policy.to_dict()
    snapshot_path = case.with_suffix(".policy.json")
    if os.environ.get("FRESHDATA_REGEN_GOLDEN"):
        snapshot_path.write_text(
            json.dumps(got, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
    expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert got == expected, (
        f"{case.name} compiled differently from its snapshot; if the change is "
        "intentional, regenerate with FRESHDATA_REGEN_GOLDEN=1 and review the diff"
    )


@pytest.mark.parametrize("case", _CASES, ids=lambda p: p.stem)
def test_golden_compile_is_deterministic(case):
    text = case.read_text(encoding="utf-8")
    assert compile_context(text, columns=GOLDEN_SCHEMA) == compile_context(
        text, columns=GOLDEN_SCHEMA
    )


def test_every_unmatched_sentence_is_surfaced():
    """Invariant: nothing is ever silently dropped across the whole corpus."""
    for case in _CASES:
        text = case.read_text(encoding="utf-8")
        result = parse_context(text)
        accounted = len(result.candidates) + len(result.unparsed)
        assert accounted == len(split_sentences(text)), case.name
