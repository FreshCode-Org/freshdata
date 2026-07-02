"""Tests for persistent cleaning memory (learn / save / load / replay / diff)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import freshdata as fd
from freshdata import CleaningMemory


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({
        "amount": [1.0, 2.0, None, 4.0, 5.0],
        "name": ["x", "x", "y", None, "z"],
        "id": [1, 2, 3, 4, 5],
    })


def test_learn_and_serialize_roundtrip(df: pd.DataFrame, tmp_path) -> None:
    _, report = fd.clean(df, return_report=True)
    mem = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm",
                                   exceptions=[{"column": "amount", "reason": "spikes"}])
    assert isinstance(mem, CleaningMemory)
    assert mem.signature["n_cols"] == 3
    assert mem.accepted  # decisions captured

    p = tmp_path / "mem.json"
    mem.to_json(str(p))
    loaded = fd.load_cleaning_memory(str(p))
    assert loaded.to_dict() == mem.to_dict()


def test_sqlite_roundtrip(df: pd.DataFrame, tmp_path) -> None:
    mem = fd.learn_cleaning_memory(df, decisions=[], dataset_id="crm")
    p = tmp_path / "mem.sqlite"
    mem.to_json(str(p))
    loaded = fd.load_cleaning_memory(str(p))
    assert loaded.dataset_id == "crm"


def test_replay_marks_actions_and_appears_in_report(df: pd.DataFrame) -> None:
    _, report = fd.clean(df, return_report=True)
    mem = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm")
    similar = pd.DataFrame({
        "amount": [10.0, None, 30.0, 40.0, 50.0],
        "name": ["a", "b", "b", "c", None],
        "id": [6, 7, 8, 9, 10],
    })
    cleaned, rep2 = fd.clean(similar, memory=mem, return_report=True)
    memory_actions = [a for a in rep2.actions if a.step == "memory"]
    assert memory_actions, "memory usage must appear in the report"
    assert any(a.memory_influenced for a in rep2.actions)


def test_unsafe_replay_is_blocked_and_explained(df: pd.DataFrame) -> None:
    mem = fd.learn_cleaning_memory(df, decisions=[], dataset_id="crm")
    unrelated = pd.DataFrame({"totally": [1], "different": [2], "schema": [3]})
    _, rep = fd.clean(unrelated, memory=mem, return_report=True)
    assert any("ignored" in w for w in rep.warnings)
    # No actions should be marked as memory-influenced when replay is blocked.
    assert not any(a.memory_influenced for a in rep.actions)


def test_memory_match_overlap(df: pd.DataFrame) -> None:
    mem = fd.learn_cleaning_memory(df, decisions=[], dataset_id="crm")
    assert mem.match(df).ok
    assert mem.match(df).overlap == 1.0
    dropped = df.drop(columns=["name", "id"])
    assert not mem.match(dropped).ok


def test_diff(df: pd.DataFrame) -> None:
    mem_a = fd.learn_cleaning_memory(df, decisions=[], dataset_id="a",
                                     thresholds={"duplicate_threshold": 0.1})
    mem_b = fd.learn_cleaning_memory(df.drop(columns=["id"]), decisions=[],
                                     dataset_id="b",
                                     thresholds={"duplicate_threshold": 0.2})
    d = mem_a.diff(mem_b)
    assert d["signature_changed"] is True
    assert "id" in d["columns_removed"]
    assert "duplicate_threshold" in d["threshold_changes"]


def test_memory_requires_pandas_and_type() -> None:
    with pytest.raises(TypeError):
        fd.clean(pd.DataFrame({"a": [1]}), memory="not a memory", return_report=True)


def test_summary_and_html(df: pd.DataFrame, tmp_path, monkeypatch) -> None:
    _, report = fd.clean(df, return_report=True)
    mem = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm")
    assert "cleaning memory 'crm'" in mem.summary()
    assert "<div class=\"fd-report\"" in mem.to_html()
    monkeypatch.chdir(tmp_path)
    assert mem.show().endswith(".html")


def test_config_overrides_preserve_and_roles(df: pd.DataFrame) -> None:
    mem = fd.learn_cleaning_memory(
        df,
        decisions=[{"column": "amount", "action": "preserve"}],
        dataset_id="crm",
        roles={"id": "id", "amount": "target"},
        thresholds={"duplicate_threshold": 0.25},
    )
    ov = mem.config_overrides()
    assert "amount" in ov["preserve_columns"]
    assert ov["target_column"] == "amount"
    assert ov["duplicate_threshold"] == 0.25


# -- Retrieval-backed semantic memory ---------------------------------------- #

DATE_COMMON = {"return_report": True, "verbose": False, "fix_dtypes": False}


def date_frame(values: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"id": range(len(values)), "signup_date": values})


ISO_VALUES = ["2026-07-01", "2026-06-15", "2025-12-31", "2026-01-01",
              "2026-03-10", "2026-07-01", "2026-06-15", "2025-12-31"]


def test_learn_stores_semantic_repairs_in_value_patterns() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")
    repairs = mem.value_patterns["semantic_repairs"]
    assert repairs
    assert all(r["issue_type"] == "date_phrase" for r in repairs)
    assert all(r["dataset_id"] == "crm" for r in repairs)
    assert {"2026-07-01", "2026-06-15", "2025-12-31", "2026-01-01", "2026-03-10"} == {
        r["raw_value"] for r in repairs
    }


def test_skipped_and_rejected_semantic_actions_are_not_learned() -> None:
    # Ambiguous dates without dayfirst context are always suggested/skipped,
    # never "approved"/"automatic" -> must not be learned.
    values = ["01/02/2026", "03/04/2026", "05/06/2026", "02/01/2026",
              "04/03/2026", "06/05/2026", "01/02/2026", "03/04/2026"]
    ctx = {"columns": {"signup_date": {"semantic_type": "date"}}}
    learn_df = date_frame(values)
    _, report = fd.clean(learn_df, semantic_mode="auto", semantic_context=ctx, **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")
    assert mem.value_patterns.get("semantic_repairs", []) == []


def test_replay_applies_compatible_semantic_date_repair() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")

    replay_df = date_frame(ISO_VALUES)
    out, rep2 = fd.clean(replay_df, semantic_mode="auto", memory=mem, **DATE_COMMON)
    assert pd.api.types.is_datetime64_any_dtype(out["signup_date"])
    semantic_actions = [a for a in rep2 if a.step == "semantic"]
    assert semantic_actions
    assert all(a.memory_influenced for a in semantic_actions)
    assert all(a.model_id == "semantic:date_phrase:memory" for a in semantic_actions)


def test_replay_sets_memory_influenced_true() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")
    _, rep2 = fd.clean(date_frame(ISO_VALUES), semantic_mode="auto", memory=mem, **DATE_COMMON)
    assert any(a.step == "semantic" and a.memory_influenced for a in rep2)


def test_replay_uses_memory_model_id() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")
    _, rep2 = fd.clean(date_frame(ISO_VALUES), semantic_mode="auto", memory=mem, **DATE_COMMON)
    model_ids = {a.model_id for a in rep2 if a.step == "semantic"}
    assert model_ids == {"semantic:date_phrase:memory"}


def test_replay_does_not_bypass_target_column_protection() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")

    replay_df = date_frame(ISO_VALUES)
    out, rep2 = fd.clean(
        replay_df, semantic_mode="auto", memory=mem, target_column="signup_date", **DATE_COMMON
    )
    assert out["signup_date"].tolist() == ISO_VALUES  # untouched
    semantic_actions = [a for a in rep2 if a.step == "semantic" and a.column == "signup_date"]
    assert semantic_actions and all(a.status == "skipped" for a in semantic_actions)


def test_replay_blocked_when_match_fails() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")

    unrelated = pd.DataFrame({"totally": [1, 2, 3], "different": [4, 5, 6], "schema": [7, 8, 9]})
    assert not mem.match(unrelated).ok
    _, rep2 = fd.clean(
        unrelated, semantic_mode="auto", memory=mem, return_report=True, verbose=False
    )
    assert not any(a.step == "semantic" for a in rep2)


def test_fuzzy_retrieval_below_threshold_is_not_applied() -> None:
    learn_df = date_frame(ISO_VALUES)
    _, report = fd.clean(learn_df, semantic_mode="auto", **DATE_COMMON)
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")

    below_values = ["xx-yy-zzzz", "2026-06-15", "2025-12-31", "2026-01-01",
                     "2026-03-10", "xx-yy-zzzz", "2026-06-15", "2025-12-31"]
    out, rep2 = fd.clean(date_frame(below_values), semantic_mode="auto", memory=mem, **DATE_COMMON)
    assert "xx-yy-zzzz" in out["signup_date"].tolist()
    garbage_actions = [
        a for a in rep2 if a.step == "semantic" and a.metadata.get("raw_value") == "xx-yy-zzzz"
    ]
    assert garbage_actions == []


def test_conflicting_memory_and_deterministic_proposals_are_not_auto_applied() -> None:
    # Learn a dayfirst-resolved repair (01/02/2026 -> 2026-02-01).
    learn_values = ["01/02/2026"] * 8
    learn_df = date_frame(learn_values)
    dayfirst_ctx = {"columns": {"signup_date": {"semantic_type": "date", "dayfirst": True}}}
    _, report = fd.clean(
        learn_df, semantic_mode="auto", semantic_context=dayfirst_ctx, **DATE_COMMON
    )
    mem = fd.learn_cleaning_memory(learn_df, decisions=report, dataset_id="crm")

    # Replay the same raw string *without* the dayfirst hint: the deterministic
    # expert now treats it as ambiguous (different guess) -> conflict with memory.
    replay_values = ["01/02/2026"] * 8
    out, rep2 = fd.clean(
        date_frame(replay_values), semantic_mode="auto", memory=mem, **DATE_COMMON
    )
    assert out["signup_date"].tolist() == replay_values  # never mutated
    conflicts = [
        a for a in rep2
        if a.step == "semantic" and a.metadata.get("issue_type") == "unsafe_ambiguous"
    ]
    assert conflicts
    assert all(a.status != "automatic" and a.human_review for a in conflicts)


def test_report_to_dict_serializes_semantic_metadata_safely() -> None:
    out, report = fd.clean(date_frame(ISO_VALUES), semantic_mode="auto", **DATE_COMMON)
    payload = report.to_dict()
    semantic_entries = [a for a in payload["actions"] if a["step"] == "semantic"]
    assert semantic_entries
    assert all("metadata" in a for a in semantic_entries)
    json.dumps(payload)  # must round-trip through plain JSON


def test_non_semantic_memory_replay_still_marks_actions(df: pd.DataFrame) -> None:
    # Regression guard for the annotate_report step=="semantic" exclusion: plain
    # (non-semantic) decisions must still replay exactly as before.
    _, report = fd.clean(df, return_report=True)
    mem = fd.learn_cleaning_memory(df, decisions=report, dataset_id="crm")
    similar = pd.DataFrame({
        "amount": [10.0, None, 30.0, 40.0, 50.0],
        "name": ["a", "b", "b", "c", None],
        "id": [6, 7, 8, 9, 10],
    })
    cleaned, rep2 = fd.clean(similar, memory=mem, return_report=True)
    memory_actions = [a for a in rep2.actions if a.step == "memory"]
    assert memory_actions
    assert any(a.memory_influenced for a in rep2.actions)
