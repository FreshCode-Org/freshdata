from __future__ import annotations

import pandas as pd
from benchmarks.performance.semantic_probe import (
    _patched_context_operations,
    _ProbeCollector,
    probe_context_build,
)

from freshdata.config import CleanConfig
from freshdata.semantic import context as semantic_context


def test_probe_counts_only_within_one_context_build() -> None:
    frame = pd.DataFrame(
        {
            "left": ["yes", "no", "2024-01-01"],
            "right": ["yes", "no", "2024-01-01"],
        }
    )
    config = CleanConfig(semantic_mode="assist", verbose=False)

    _context, first = probe_context_build(frame, config)
    _context, second = probe_context_build(frame, config)

    assert set(first.by_operation) == {
        "is_plain_number",
        "parse_number_words",
        "parse_boolean",
        "parse_currency",
        "parse_unit",
        "email_value",
        "looks_like_date_value",
    }
    assert first.by_operation["parse_boolean"].theoretical_hits >= 1
    assert second.total_theoretical_hits == first.total_theoretical_hits


def test_probe_uses_exact_types_and_bypasses_unsafe_values() -> None:
    frame = pd.DataFrame(
        {
            "ints": pd.Series([1], dtype=object),
            "bools": pd.Series([True], dtype=object),
            "text": pd.Series(["1"], dtype=object),
        }
    )
    _context, result = probe_context_build(
        frame, CleanConfig(semantic_mode="assist", verbose=False)
    )

    numeric = result.by_operation["is_plain_number"]
    assert {type(value) for value in numeric.eligible_values} == {int, bool, str}
    assert numeric.unique_keys == 3


def test_direct_probe_bypasses_unhashable_list_without_hashing() -> None:
    collector = _ProbeCollector()
    value = [1, 2, 3]
    collector.record("is_plain_number", value)

    result = collector.finish_build()
    numeric = result.by_operation["is_plain_number"]
    assert numeric.bypassed_calls == 1
    assert numeric.eligible_calls == 0
    assert numeric.unique_keys == 0


def test_parse_boolean_probe_keeps_post_stringification_value() -> None:
    probe = _ProbeCollector()
    value = 1
    with _patched_context_operations(probe):
        semantic_context.parse_boolean(str(value))
    result = probe.finish_build()

    assert result.by_operation["parse_boolean"].eligible_values == (str(value),)
