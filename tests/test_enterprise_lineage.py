"""Lineage capture and OpenLineage export tests."""

import json

import pandas as pd
import pytest

from freshdata.enterprise import LineageEvent, LineageTracker, schema_of
from freshdata.enterprise import lineage as lineage_mod
from freshdata.enterprise.config import LineageConfig


def test_schema_of_pandas():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    schema = schema_of(df)
    assert [f["name"] for f in schema] == ["a", "b"]
    assert all(set(f) == {"name", "type"} for f in schema)


def test_schema_of_polars():
    pl = pytest.importorskip("polars")
    schema = schema_of(pl.DataFrame({"a": [1], "b": ["x"]}))
    assert [f["name"] for f in schema] == ["a", "b"]


def test_schema_of_accepts_precomputed_forms():
    assert schema_of([{"name": "a", "type": "int"}]) == [{"name": "a", "type": "int"}]
    assert schema_of([("a", "int64")]) == [{"name": "a", "type": "int64"}]


def test_schema_of_rejects_unknown():
    with pytest.raises(TypeError):
        schema_of(42)


def test_default_actor_falls_back_when_getuser_fails(monkeypatch):
    def boom():
        raise OSError("no user in this environment")

    monkeypatch.setattr(lineage_mod.getpass, "getuser", boom)
    assert lineage_mod._default_actor() == "unknown"


def test_record_captures_who_when_and_schemas():
    before = pd.DataFrame({"a": [1], "b": [2]})
    after = pd.DataFrame({"a": [1]})
    tracker = LineageTracker(LineageConfig(actor="alice"))
    event = tracker.record("drop_col", before, after, count=1, description="dropped b")
    assert isinstance(event, LineageEvent)
    assert event.who == "alice"
    assert event.rule_applied == "drop_col"
    assert "T" in event.when
    assert [n for n, _ in event.input_schema] == ["a", "b"]
    assert [n for n, _ in event.output_schema] == ["a"]


def test_record_uses_login_actor_when_unset():
    tracker = LineageTracker()  # default LineageConfig, actor=None
    event = tracker.record("noop", pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}))
    assert isinstance(event.who, str) and event.who


def test_schema_properties_empty_then_populated():
    tracker = LineageTracker()
    assert tracker.input_schema == ()
    assert tracker.output_schema == ()
    tracker.record("s", pd.DataFrame({"a": [1], "b": [2]}), pd.DataFrame({"a": [1]}))
    assert [n for n, _ in tracker.input_schema] == ["a", "b"]
    assert [n for n, _ in tracker.output_schema] == ["a"]


def test_event_to_dict_shape():
    tracker = LineageTracker(LineageConfig(actor="bob"))
    event = tracker.record("r", pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}), count=3)
    payload = event.to_dict()
    assert payload["who"] == "bob"
    assert payload["count"] == 3
    assert payload["input_schema"] == [{"name": "a", "type": payload["input_schema"][0]["type"]}]


def test_to_dict_and_repr():
    tracker = LineageTracker()
    tracker.record("r", pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}))
    payload = tracker.to_dict()
    assert set(payload) >= {"run_id", "namespace", "job", "events"}
    assert len(payload["events"]) == 1
    assert repr(tracker).startswith("<LineageTracker")


def test_openlineage_start_complete_with_facets():
    tracker = LineageTracker(LineageConfig(namespace="ns", job_name="job"))
    tracker.record("clean", pd.DataFrame({"a": [1], "b": [2]}), pd.DataFrame({"a": [1], "b": [2]}))
    events = tracker.to_openlineage()
    assert [e["eventType"] for e in events] == ["START", "COMPLETE"]
    start, complete = events
    assert start["job"] == {"namespace": "ns", "name": "job"}
    # START carries the run transformations facet but no dataset schema facet.
    assert "freshdata_transformations" in start["run"]["facets"]
    assert start["inputs"][0]["facets"] == {}
    # COMPLETE carries schema + identity column lineage.
    out_facets = complete["outputs"][0]["facets"]
    assert [f["name"] for f in out_facets["schema"]["fields"]] == ["a", "b"]
    assert set(out_facets["columnLineage"]["fields"]) == {"a", "b"}


def test_to_json_is_valid():
    tracker = LineageTracker()
    tracker.record("r", pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}))
    assert len(json.loads(tracker.to_json())) == 2


def test_emit_writes_file_and_returns_payload(tmp_path):
    tracker = LineageTracker()
    tracker.record("r", pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}))
    path = tmp_path / "lineage.json"
    returned = tracker.emit(str(path))
    assert path.read_text() == returned
    assert json.loads(path.read_text())[0]["eventType"] == "START"


def test_emit_without_path_returns_payload_only():
    tracker = LineageTracker()
    tracker.record("r", pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1]}))
    assert json.loads(tracker.emit())  # no file, just the string
