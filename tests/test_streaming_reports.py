"""Per-batch report content: streaming summary fields, trust scoring, schema-drift
detection, optional-connector ImportErrors, and CleanReport serialization compatibility.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import freshdata as fd


def make_batch(n: int = 200, *, seed: int = 0, extra_col: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "customer_id": np.arange(n, dtype="int64"),
        "churn": rng.integers(0, 2, n),
        "amount": rng.normal(50, 10, n),
        "region": rng.choice(["north", "south", "east"], n),
    }
    df = pd.DataFrame(data)
    df.loc[df.index[:10], "amount"] = np.nan
    if extra_col:
        df["surprise"] = rng.normal(0, 1, n)
    return df


def high_missing_batch(n: int = 200, *, seed: int = 0) -> pd.DataFrame:
    """A batch whose ``sparse`` column is ~70% missing, so it is *preserved* (not
    filled) and the cleaned batch keeps an imperfection that holds trust below 100."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": np.arange(n, dtype="int64"),
        "churn": rng.integers(0, 2, n),
        "amount": rng.normal(50, 10, n),
        "sparse": rng.normal(0, 1, n),
    })
    df.loc[df.index[: int(0.7 * n)], "sparse"] = np.nan
    return df


# -- streaming summary fields (#10) -------------------------------------------------


def test_report_carries_batch_id_and_rows_seen_total():
    cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                                  warmup_batches=1)
    reports = [report for _, report in
               cleaner.clean_batches(make_batch(100, seed=i) for i in range(4))]

    assert [r.streaming["batch_id"] for r in reports] == [1, 2, 3, 4]
    assert [r.streaming["rows_seen_total"] for r in reports] == [100, 200, 300, 400]
    for r in reports:
        assert r.streaming["rows_in_batch"] == 100
        assert {"warmup_phase", "schema_drift_detected", "trust_gate_passed"} <= r.streaming.keys()


def test_warmup_phase_flag_flips_after_warmup():
    cleaner = fd.StreamingCleaner(warmup_batches=2)
    flags = [report.streaming["warmup_phase"]
             for _, report in cleaner.clean_batches(make_batch(80, seed=i) for i in range(4))]
    assert flags == [True, True, False, False]


# -- trust scoring per batch and cumulatively (#13) ---------------------------------


def test_trust_score_per_batch_and_cumulative():
    cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                                  warmup_batches=1)
    seen_cumulatives = []
    for _, report in cleaner.clean_batches(make_batch(150, seed=i) for i in range(5)):
        s = report.streaming
        for key in ("batch_trust_score", "rolling_trust_score", "cumulative_trust_score"):
            assert 0.0 <= s[key] <= 100.0
        seen_cumulatives.append(s["cumulative_trust_score"])

    # The cleaner exposes the running scores, matching the last batch's report.
    assert cleaner.cumulative_trust_score == pytest.approx(seen_cumulatives[-1], abs=0.01)
    assert cleaner.rolling_trust_score is not None
    # finalize() reports the same cumulative score it has been tracking.
    final = cleaner.finalize()
    assert final.streaming["cumulative_trust_score"] == pytest.approx(
        seen_cumulatives[-1], abs=0.01)


def test_fail_under_trust_records_gate_failures():
    cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
                                  warmup_batches=0, fail_under_trust=99.0)
    for _, report in cleaner.clean_batches(high_missing_batch(120, seed=i) for i in range(3)):
        assert report.streaming["trust_gate_passed"] is False
    assert cleaner.finalize().streaming["trust_gate_failures"] == 3


# -- schema-drift detection (#11, #12) ----------------------------------------------


def test_schema_drift_new_column_detected():
    # Stay in warmup so imputation never runs: this isolates drift behaviour.
    cleaner = fd.StreamingCleaner(warmup_batches=10)
    cleaner.clean_batch(make_batch(120, seed=0))            # batch 1: locks the baseline
    _, report = cleaner.clean_batch(make_batch(120, seed=1, extra_col=True))

    assert report.streaming["schema_drift_detected"] is True
    drift_actions = [a for a in report if a.step == "drift"]
    assert any(a.column == "surprise" and "new column" in a.description for a in drift_actions)
    assert any("drift" in w for w in report.warnings)


def test_schema_drift_missing_column_detected():
    cleaner = fd.StreamingCleaner(warmup_batches=10)
    cleaner.clean_batch(make_batch(120, seed=0, extra_col=True))   # baseline has 'surprise'
    _, report = cleaner.clean_batch(make_batch(120, seed=1))       # 'surprise' gone

    assert report.streaming["schema_drift_detected"] is True
    assert any(a.step == "drift" and a.column == "surprise" for a in report)


def test_schema_drift_dtype_change_detected():
    cleaner = fd.StreamingCleaner(warmup_batches=10)
    baseline = pd.DataFrame({"score": pd.array([1, 2, 3, 4, 5], dtype="int64")})
    cleaner.clean_batch(baseline)
    drifted = pd.DataFrame({"score": ["high", "low", "mid", "high", "low"]})  # object dtype
    _, report = cleaner.clean_batch(drifted)

    assert report.streaming["schema_drift_detected"] is True
    assert any(a.step == "drift" and "dtype" in a.description for a in report)


def test_first_batch_never_reports_drift():
    cleaner = fd.StreamingCleaner(warmup_batches=10)
    _, report = cleaner.clean_batch(make_batch(100, seed=0))
    assert report.streaming["schema_drift_detected"] is False
    assert not [a for a in report if a.step == "drift"]


# -- drift actions carry the audit fields -------------------------------------------


def test_drift_actions_have_rationale_and_risk():
    cleaner = fd.StreamingCleaner(warmup_batches=10)
    cleaner.clean_batch(make_batch(100, seed=0))
    _, report = cleaner.clean_batch(make_batch(100, seed=1, extra_col=True))
    drift_actions = [a for a in report if a.step == "drift"]
    assert drift_actions
    for a in drift_actions:
        assert a.rationale
        assert a.risk in {"low", "medium", "high"}


# -- CleanReport serialization stays backward compatible ----------------------------


def test_streaming_report_serializes_with_streaming_block():
    cleaner = fd.StreamingCleaner(warmup_batches=0)
    _, report = cleaner.clean_batch(make_batch(100, seed=0))
    payload = report.to_dict()
    # Core schema is unchanged; the streaming block is purely additive.
    assert {"actions", "warnings", "streaming"} <= payload.keys()
    assert payload["streaming"]["batch_id"] == 1
    assert isinstance(payload["actions"], list)


def test_non_streaming_report_has_no_streaming_block():
    _, report = fd.clean(pd.DataFrame({"a": [1, 2, None, 4]}), return_report=True)
    assert "streaming" not in report.to_dict()


# -- optional connectors raise a clear ImportError when missing (#17, #18) ----------


def test_clean_kafka_raises_clear_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "kafka", None)  # force `import kafka` to fail
    cleaner = fd.StreamingCleaner()
    gen = cleaner.clean_kafka(topic="events", bootstrap_servers="localhost:9092")
    with pytest.raises(ImportError, match=r"kafka"):
        next(gen)


def test_clean_arrow_flight_raises_clear_importerror(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow.flight", None)  # force flight import to fail
    cleaner = fd.StreamingCleaner()
    gen = cleaner.clean_arrow_flight("grpc://localhost:8815")
    with pytest.raises(ImportError, match=r"[Ff]light"):
        next(gen)


# -- mock-based connector happy paths (no live Kafka / Flight) -----------------------


def _install_fake_kafka(monkeypatch, records):
    """Inject a minimal in-memory ``kafka`` module whose consumer replays *records*."""
    fake = types.ModuleType("kafka")

    class _Message:
        def __init__(self, value):
            self.value = value

    class KafkaConsumer:
        def __init__(self, topic, **kwargs):
            self.topic = topic
            self._messages = [_Message(r) for r in records]

        def __iter__(self):
            return iter(self._messages)

    fake.KafkaConsumer = KafkaConsumer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kafka", fake)


def test_clean_kafka_with_mock_consumer(monkeypatch):
    records = [json.dumps({"customer_id": i, "amount": float(i % 7)}).encode()
               for i in range(250)]
    _install_fake_kafka(monkeypatch, records)

    cleaner = fd.StreamingCleaner(id_columns=("customer_id",), warmup_batches=1)
    out = list(cleaner.clean_kafka(topic="events", bootstrap_servers="localhost:9092",
                                   batch_size=100))
    # 250 records / 100 -> two full batches plus a 50-row remainder.
    assert [len(df) for df, _ in out] == [100, 100, 50]
    assert out[-1][1].streaming["rows_seen_total"] == 250


@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="pyarrow not installed")
def test_clean_arrow_flight_with_mock(monkeypatch):
    df = pd.DataFrame({"customer_id": range(120), "amount": [1.0, None, 3.0] * 40})

    fake_flight = types.ModuleType("pyarrow.flight")

    class _Chunk:
        def __init__(self, frame):
            self._frame = frame

        def to_pandas(self):
            return self._frame

    class _Reader:
        def __iter__(self):
            return iter([_Chunk(df)])

    class _Endpoint:
        ticket = "ticket"

    class _Info:
        endpoints = [_Endpoint()]

    class _Client:
        def get_flight_info(self, descriptor):
            return _Info()

        def do_get(self, ticket):
            return _Reader()

    class _FlightDescriptor:
        @staticmethod
        def for_path(*path):
            return ("path", path)

    fake_flight.connect = lambda location: _Client()           # type: ignore[attr-defined]
    fake_flight.FlightDescriptor = _FlightDescriptor           # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyarrow.flight", fake_flight)

    cleaner = fd.StreamingCleaner(id_columns=("customer_id",), warmup_batches=1)
    out = list(cleaner.clean_arrow_flight("grpc://localhost:8815", batch_size=50))
    # 120 rows sliced into 50/50/20 micro-batches by the Flight connector.
    assert [len(df) for df, _ in out] == [50, 50, 20]
