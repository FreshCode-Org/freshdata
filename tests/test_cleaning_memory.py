"""Tests for persistent cleaning memory (learn / save / load / replay / diff)."""

from __future__ import annotations

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
