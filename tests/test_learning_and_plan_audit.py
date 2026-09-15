"""Regression tests for learned-memory types, merge isolation and plan audit.

* #256 — learned clean values (numpy/pandas scalars) must be JSON-native in
  the embedded ``CleaningMemory`` so profiles save and round-trip.
* #257 — ``LearningProfile.merge`` must not mutate or share a parent's memory.
* #258 — ``apply_plan(allow_drift=True)`` must record actions whose raw values
  are gone as skipped, and applied actions with their observed cell count.
"""

from __future__ import annotations

import decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata.learning import learn, load_profile, save_profile
from freshdata.learning.extract import _pattern_value
from freshdata.learning.merge import _merge_memory
from freshdata.repairplan import _cell_counts, _count_matches, compute_decisions_hash
from freshdata.semantic.apply import _apply_column

# --------------------------------------------------------------------------- #
# #256 — JSON-native learned replay patterns
# --------------------------------------------------------------------------- #

LEVELS = ["low", "medium", "high"] * 10


def _ordinal_pair(mapping: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    messy = pd.DataFrame({"id": range(len(LEVELS)), "priority": LEVELS})
    clean = pd.DataFrame({"id": range(len(LEVELS)), "priority": [mapping[v] for v in LEVELS]})
    return messy, clean


def _assert_json_native(value_patterns: dict[str, object]) -> None:
    for column, patterns in value_patterns.items():
        if not isinstance(patterns, dict):
            continue
        for value in patterns.values():
            assert value is None or type(value) in (bool, int, float, str), (
                column,
                value,
                type(value),
            )


class TestLearnedPatternsAreJsonNative:
    def test_integer_clean_values_save_load_and_replay(self, tmp_path: Path) -> None:
        messy, clean = _ordinal_pair({"low": 1, "medium": 2, "high": 3})
        profile = learn(messy, clean, key="id", min_support=2)
        assert profile.memory is not None
        expected = {"low": 1, "medium": 2, "high": 3}
        assert profile.memory.value_patterns["priority"] == expected
        _assert_json_native(profile.memory.value_patterns)
        # to_json used default=str, which silently turned 1 into "1".
        assert json.loads(profile.memory.to_json())["value_patterns"]["priority"] == expected

        path = tmp_path / "ordinal.fdprofile"
        save_profile(profile, path)  # raised TypeError: int64 not JSON serializable
        loaded = load_profile(path)
        assert loaded.memory is not None
        assert loaded.memory.to_dict() == profile.memory.to_dict()

        before = fd.clean(messy, profile=profile, semantic_mode="auto", verbose=False)
        after = fd.clean(messy, profile=loaded, semantic_mode="auto", verbose=False)
        pd.testing.assert_frame_equal(before, after)

    @pytest.mark.parametrize(
        "mapping",
        [
            {"low": 0.5, "medium": 1.5, "high": 2.5},
            {"low": False, "medium": True, "high": True},
            {
                "low": pd.Timestamp("2024-01-01"),
                "medium": pd.Timestamp("2024-01-02"),
                "high": pd.Timestamp("2024-01-03"),
            },
            {
                "low": decimal.Decimal("1.1"),
                "medium": decimal.Decimal("2.2"),
                "high": decimal.Decimal("3.3"),
            },
        ],
        ids=["float", "bool", "timestamp", "decimal"],
    )
    def test_other_scalar_clean_values_save(
        self, tmp_path: Path, mapping: dict[str, object]
    ) -> None:
        messy, clean = _ordinal_pair(mapping)
        profile = learn(messy, clean, key="id", min_support=2)
        assert profile.memory is not None
        _assert_json_native(profile.memory.value_patterns)
        json.dumps(profile.memory.to_dict())  # no default= needed

        path = tmp_path / "scalars.fdprofile"
        save_profile(profile, path)
        loaded = load_profile(path)
        assert loaded.memory is not None
        assert loaded.memory.to_dict() == profile.memory.to_dict()

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (np.int64(3), 3),
            (np.float64(1.5), 1.5),
            (np.bool_(True), True),
            (np.str_("x"), "x"),
            (pd.Timestamp("2024-01-02"), "2024-01-02T00:00:00"),
            (decimal.Decimal("1.5"), "1.5"),
            (7, 7),
            ("text", "text"),
            (None, None),
        ],
    )
    def test_pattern_value_unwraps_scalars(self, value: object, expected: object) -> None:
        got = _pattern_value(value)
        assert got == expected
        assert type(got) is type(expected)


# --------------------------------------------------------------------------- #
# #257 — merge never mutates or shares a parent's CleaningMemory
# --------------------------------------------------------------------------- #

STRATEGIES = ["union_min_precision", "prefer_self", "prefer_other", "error_on_conflict"]


def _status_pair(raw: str, fixed: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    messy = pd.DataFrame({"id": range(20), "status": [raw] * 10 + ["done"] * 10})
    clean = pd.DataFrame({"id": range(20), "status": [fixed] * 10 + ["done"] * 10})
    return messy, clean


@pytest.fixture()
def parents():
    p1 = learn(*_status_pair("pend-ing", "pending"), key="id", min_support=2)
    p2 = learn(*_status_pair("in_prog", "in progress"), key="id", min_support=2)
    assert p1.memory is not None and p2.memory is not None
    return p1, p2


def _memory_snapshot(profile) -> str:
    return json.dumps(profile.memory.to_dict(), sort_keys=True)


class TestMergeLeavesParentsUntouched:
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_merge_neither_mutates_nor_shares_parent_memory(self, parents, strategy) -> None:
        p1, p2 = parents
        before = (_memory_snapshot(p1), _memory_snapshot(p2))
        merged = p1.merge(p2, strategy=strategy)
        assert (_memory_snapshot(p1), _memory_snapshot(p2)) == before
        assert merged.memory is not p1.memory
        assert merged.memory is not p2.memory
        # Later edits to the merged profile must not leak into either parent.
        merged.memory.value_patterns["status"]["zzz"] = "leaked"
        merged.memory.value_patterns["new_col"] = {"a": "b"}
        assert (_memory_snapshot(p1), _memory_snapshot(p2)) == before

    def test_default_union_still_folds_in_other_patterns(self, parents) -> None:
        p1, p2 = parents
        merged = p1.merge(p2)
        assert merged.memory.value_patterns["status"] == {
            "pend-ing": "pending",
            "in_prog": "in progress",
        }
        assert p1.memory.value_patterns == {"status": {"pend-ing": "pending"}}
        assert p2.memory.value_patterns == {"status": {"in_prog": "in progress"}}

    def test_saved_parent_bytes_unchanged_by_merge(self, parents, tmp_path: Path) -> None:
        p1, p2 = parents
        first, second = tmp_path / "before.fdprofile", tmp_path / "after.fdprofile"
        save_profile(p1, first)
        p1.merge(p2)
        save_profile(p1, second)
        assert first.read_bytes() == second.read_bytes()

    def test_merge_memory_helper_copies_every_branch(self, parents) -> None:
        a, b = parents[0].memory, parents[1].memory
        results = [
            _merge_memory(a, None, "union_min_precision"),
            _merge_memory(None, b, "union_min_precision"),
            _merge_memory(a, b, "prefer_self"),
            _merge_memory(a, b, "prefer_other"),
            _merge_memory(a, b, "union_min_precision"),
        ]
        for got in results:
            assert got is not a and got is not b
            for column, patterns in got.value_patterns.items():
                assert patterns is not a.value_patterns.get(column)
                assert patterns is not b.value_patterns.get(column)
        assert _merge_memory(None, None, "prefer_self") is None


# --------------------------------------------------------------------------- #
# #258 — apply_plan audit reflects what actually happened under drift
# --------------------------------------------------------------------------- #

PLAN_RULES = "Allowed status values are active, inactive, pending."
RAW_VALUES = (" Active ", "INACTIVE", "pend-ing")


def _messy_status() -> pd.DataFrame:
    return pd.DataFrame({"status": [" Active ", "INACTIVE", "pend-ing", "active"]})


def _approved_plan(df: pd.DataFrame):
    plan = fd.suggest_plan(df, context=PLAN_RULES, semantic_mode="auto", verbose=False)
    repair_plan = plan.repair_plan
    repair_plan.approve_all(max_risk="high")
    by_raw = {a.params.get("raw_value"): a for a in repair_plan.actions}
    assert set(RAW_VALUES) <= set(by_raw)
    return repair_plan, by_raw


def _plan_entries(report) -> list:
    return [a for a in report.actions if a.step == "apply_plan"]


class TestApplyPlanDriftAudit:
    def test_all_raw_values_gone_records_skips_not_applies(self) -> None:
        plan, by_raw = _approved_plan(_messy_status())
        fixed = pd.DataFrame({"status": ["active", "inactive", "pending", "active"]})
        hash_before = compute_decisions_hash(plan)

        out, report = fd.apply_plan(fixed, plan, allow_drift=True)

        pd.testing.assert_frame_equal(out, fixed)
        entries = _plan_entries(report)
        assert entries
        assert [e for e in entries if e.status == "approved"] == []
        assert all(e.status == "skipped" and e.count == 0 for e in entries)
        descriptions = [e.description for e in entries]
        for raw in RAW_VALUES:
            action_id = by_raw[raw].id
            assert f"skipped {action_id}: value {raw!r} not present in column 'status' " \
                "(frame drift)" in descriptions
            assert f"did not apply {action_id} (frame drift)" in descriptions
        assert report.decisions_hash == hash_before
        assert compute_decisions_hash(plan) == hash_before  # plan not mutated

    def test_partial_drift_records_observed_counts_and_undo(self) -> None:
        plan, by_raw = _approved_plan(_messy_status())
        active = by_raw[" Active "]
        assert active.n_affected == 1  # plan-time estimate
        drifted = pd.DataFrame({"status": [" Active ", "active", "pending", " Active "]})

        out, report = fd.apply_plan(drifted, plan, allow_drift=True, keep_undo=True)

        assert out["status"].tolist() == ["active", "active", "pending", "active"]
        entries = _plan_entries(report)
        applied = [(e.metadata["action_id"], e.count) for e in entries if e.status == "approved"]
        assert applied == [(active.id, 2)]  # observed cells, not n_affected
        skipped_ids = {
            e.metadata["action_id"] for e in entries if e.description.startswith("skipped ")
        }
        assert skipped_ids == {by_raw["INACTIVE"].id, by_raw["pend-ing"].id}
        assert [e["action_id"] for e in report.undo_log["entries"]] == [active.id]

    def test_without_drift_counts_match_plan_estimate(self) -> None:
        df = _messy_status()
        plan, by_raw = _approved_plan(df)

        out, report = fd.apply_plan(df, plan)

        assert out["status"].tolist() == ["active", "inactive", "pending", "active"]
        entries = _plan_entries(report)
        applied = {e.metadata["action_id"]: e.count for e in entries if e.status == "approved"}
        assert applied == {by_raw[raw].id: by_raw[raw].n_affected for raw in RAW_VALUES}
        assert not any("frame drift" in e.description for e in entries)

    def test_count_matches_mirrors_mapping_lookup(self) -> None:
        series = pd.Series(["a", "b", "a", None, 1, 1.0, True, "1"], dtype=object)
        counts = _cell_counts(series)
        assert counts is not None
        for raw in ("a", "b", "zzz", None, 1, "1"):
            replaced = _apply_column(series, {raw: "__hit__"})
            expected = int((replaced == "__hit__").sum())
            assert _count_matches(series, raw, counts) == expected, raw

    def test_count_matches_tolerates_unhashable_cells(self) -> None:
        series = pd.Series([["x"], "a", "a", {"k": 1}], dtype=object)
        assert _cell_counts(series) is None
        assert _count_matches(series, "a", None) == 2
        assert _count_matches(series, "zzz", None) == 0
