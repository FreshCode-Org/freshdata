import json

import freshdata as fd


def test_shape_and_memory_bookkeeping(messy):
    _, report = fd.clean(messy, report=True)
    assert report.rows_before == 5
    assert report.cols_before == 6
    assert report.rows_after == 4
    assert report.cols_after == 5
    assert report.memory_before > 0
    assert report.memory_after > 0
    assert report.duration_seconds >= 0


def test_report_is_iterable_and_sized(messy):
    _, report = fd.clean(messy, report=True)
    assert len(report) == len(list(report))
    assert all(isinstance(a, fd.Action) for a in report)
    assert report.cells_changed > 0


def test_to_dict_is_json_serializable(messy):
    _, report = fd.clean(messy, report=True)
    payload = json.dumps(report.to_dict())
    assert "drop_duplicates" in payload


def test_to_frame(messy):
    _, report = fd.clean(messy, report=True)
    frame = report.to_frame()
    assert list(frame.columns) == ["step", "column", "description", "count",
                                   "rationale", "risk", "confidence", "model_id"]
    assert len(frame) == len(report)


def test_summary_mentions_key_facts(messy):
    _, report = fd.clean(messy, report=True)
    text = report.summary()
    assert "rows:" in text and "5 -> 4" in text
    assert "[fix_dtypes]" in text
    assert str(report) == text


def test_action_str_format():
    action = fd.Action(step="impute", column="age", description="filled 2", count=2)
    assert str(action) == "[impute] 'age': filled 2"
    table_level = fd.Action(step="drop_empty_rows", column=None, description="dropped 1",
                            count=1)
    assert str(table_level) == "[drop_empty_rows] dropped 1"


def test_bool_reflects_whether_anything_changed(messy, already_clean):
    _, dirty_report = fd.clean(messy, report=True)
    _, clean_report = fd.clean(already_clean, report=True)
    assert dirty_report
    assert not clean_report


def test_repr_is_compact(messy):
    _, report = fd.clean(messy, report=True)
    assert repr(report).startswith("<CleanReport:")
