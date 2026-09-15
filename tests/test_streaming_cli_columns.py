"""``freshdata stream`` / ``stream-kafka`` reject column-name flags that name no column."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from freshdata.enterprise.cli import main


def _events_csv(path: Path, rows: int = 30) -> Path:
    pd.DataFrame({
        "event_time": pd.date_range("2024-01-01", periods=rows, freq="min").astype(str),
        "sensor": ["a", "b"] * (rows // 2),
        "value": [float(i) for i in range(rows)],
    }).to_csv(path, index=False)
    return path


def _assert_no_output(out: Path) -> None:
    assert not out.exists()
    assert not Path(f"{out}.partial").exists()


def test_unknown_timestamp_exits_1_with_one_line_error(tmp_path, capsys):
    src = _events_csv(tmp_path / "events.csv")
    out = tmp_path / "out.csv"
    reports = tmp_path / "rep"

    rc = main(["stream", str(src), "-o", str(out), "--report", str(reports),
               "--timestamp", "event_tim", "--anomaly", "mad", "--quiet"])

    assert rc == 1
    err = capsys.readouterr().err
    lines = err.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("freshdata: error: ")
    assert "--timestamp column 'event_tim' not found" in lines[0]
    _assert_no_output(out)
    assert not (reports / "summary.json").exists()
    assert not list(reports.glob("batch_*.json"))


def test_unknown_timestamp_suggests_close_match(tmp_path, capsys):
    src = _events_csv(tmp_path / "events.csv")

    rc = main(["stream", str(src), "-o", str(tmp_path / "out.csv"),
               "--timestamp", "event_tim", "--quiet"])

    assert rc == 1
    assert "(did you mean 'event_time'?)" in capsys.readouterr().err


def test_unknown_timestamp_without_close_match_has_no_suggestion(tmp_path, capsys):
    src = _events_csv(tmp_path / "events.csv")

    rc = main(["stream", str(src), "-o", str(tmp_path / "out.csv"),
               "--timestamp", "zzzz", "--quiet"])

    assert rc == 1
    err = capsys.readouterr().err
    assert "did you mean" not in err
    assert "input columns: event_time, sensor, value" in err


def test_unknown_timestamp_leaves_no_parquet_output(tmp_path):
    pytest.importorskip("pyarrow")
    src = _events_csv(tmp_path / "events.csv")
    out = tmp_path / "out.parquet"

    rc = main(["stream", str(src), "-o", str(out), "--batch-size", "10",
               "--timestamp", "ts", "--quiet"])

    assert rc == 1
    _assert_no_output(out)


def test_valid_timestamp_still_streams(tmp_path, capsys):
    src = _events_csv(tmp_path / "events.csv")
    out = tmp_path / "out.csv"
    reports = tmp_path / "rep"

    rc = main(["stream", str(src), "-o", str(out), "--report", str(reports),
               "--batch-size", "10", "--timestamp", "event_time",
               "--entity-id", "sensor", "--watermark", "event_time", "--quiet"])

    assert rc == 0
    assert capsys.readouterr().err == ""
    assert len(pd.read_csv(out)) == 30
    assert not Path(f"{out}.partial").exists()
    summary = json.loads((reports / "summary.json").read_text())
    assert "time_series" in summary["streaming"]


@pytest.mark.parametrize("flag_args, flag", [
    (["--timestamp", "event_time", "--watermark", "evt"], "--watermark"),
    (["--timestamp", "event_time", "--entity-id", "sensor", "sensr"], "--entity-id"),
    (["--timestamp", "event_time", "--ordered-dedupe-keys", "id"], "--ordered-dedupe-keys"),
    (["--target-column", "valu"], "--target-column"),
    (["--id-columns", "sensor", "id"], "--id-columns"),
])
def test_other_unknown_column_flags_exit_1(tmp_path, capsys, flag_args, flag):
    src = _events_csv(tmp_path / "events.csv")
    out = tmp_path / "out.csv"

    rc = main(["stream", str(src), "-o", str(out), "--quiet", *flag_args])

    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("freshdata: error: ")
    assert f"{flag} column " in err
    _assert_no_output(out)


def test_all_unknown_columns_are_reported_together(tmp_path, capsys):
    src = _events_csv(tmp_path / "events.csv")

    rc = main(["stream", str(src), "-o", str(tmp_path / "out.csv"), "--quiet",
               "--timestamp", "event_tim", "--target-column", "valu"])

    assert rc == 1
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "--timestamp column 'event_tim'" in err
    assert "--target-column column 'valu'" in err


def test_time_series_flags_are_not_checked_without_timestamp(tmp_path):
    src = _events_csv(tmp_path / "events.csv")
    out = tmp_path / "out.csv"

    rc = main(["stream", str(src), "-o", str(out), "--quiet", "--watermark", "nope"])

    assert rc == 0
    assert out.exists()


def _fake_kafka(monkeypatch):
    records = [json.dumps({"customer_id": i, "amount": float(i % 5)}).encode()
               for i in range(200)]
    fake = types.ModuleType("kafka")

    class KafkaConsumer:
        def __init__(self, topic, **kwargs):
            self._messages = [type("M", (), {"value": r})() for r in records]

        def __iter__(self):
            return iter(self._messages)

    fake.KafkaConsumer = KafkaConsumer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kafka", fake)


def test_stream_kafka_unknown_id_column_exits_1(tmp_path, monkeypatch, capsys):
    _fake_kafka(monkeypatch)
    out = tmp_path / "out.csv"

    rc = main(["stream-kafka", "--topic", "events", "--bootstrap-servers", "localhost:9092",
               "--batch-size", "100", "--max-batches", "2",
               "--id-columns", "customer", "-o", str(out), "--quiet"])

    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("freshdata: error: ")
    assert "--id-columns column 'customer' not found" in err
    assert "(did you mean 'customer_id'?)" in err
    _assert_no_output(out)


def test_stream_kafka_valid_columns_still_work(tmp_path, monkeypatch):
    _fake_kafka(monkeypatch)
    out = tmp_path / "out.csv"

    rc = main(["stream-kafka", "--topic", "events", "--bootstrap-servers", "localhost:9092",
               "--batch-size", "100", "--max-batches", "2",
               "--id-columns", "customer_id", "--target-column", "amount",
               "-o", str(out), "--quiet"])

    assert rc == 0
    assert len(pd.read_csv(out)) == 200
