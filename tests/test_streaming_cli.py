"""CLI streaming mode: chunked CSV reading (never a full load), per-batch + summary
report files, the trust-gate exit code, and the benchmark's lazy data generator.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

import freshdata.streaming._cli as cli
from freshdata.enterprise.cli import main

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks"


def _write_csv(path: Path, rows: int = 250) -> Path:
    df = pd.DataFrame({
        "customer_id": range(rows),
        "churn": [0, 1] * (rows // 2),
        "amount": [1.0, 2.0, None, 4.0, 5.0] * (rows // 5),
        "region": ["north", "south"] * (rows // 2),
    })
    df.to_csv(path, index=False)
    return path


# -- chunked CSV reading (#19) ------------------------------------------------------


def test_cli_stream_reads_csv_in_chunks(tmp_path, monkeypatch):
    csv = _write_csv(tmp_path / "in.csv", rows=250)
    out = tmp_path / "out.csv"

    real_read_csv = pd.read_csv
    seen: dict[str, object] = {}

    def spy(path, **kwargs):
        # Record how the CLI asked pandas to read — it must be chunked.
        seen["chunksize"] = kwargs.get("chunksize")
        result = real_read_csv(path, **kwargs)
        seen["returns_iterator"] = not isinstance(result, pd.DataFrame)
        return result

    monkeypatch.setattr(cli.pd, "read_csv", spy)

    rc = main(["stream", str(csv), "-o", str(out),
               "--batch-size", "100", "--id-columns", "customer_id",
               "--target-column", "churn", "--warmup-batches", "1", "--quiet"])

    assert rc == 0
    # Proof the full CSV was never materialized: read_csv was given a chunksize and
    # handed back a streaming reader, not a single DataFrame of every row.
    assert seen["chunksize"] == 100
    assert seen["returns_iterator"] is True
    assert out.exists()
    # Output round-trips and preserves the row count across all chunks.
    assert len(pd.read_csv(out)) == 250


def test_cli_stream_writes_per_batch_and_summary_reports(tmp_path):
    csv = _write_csv(tmp_path / "in.csv", rows=300)
    reports = tmp_path / "reports"

    rc = main(["stream", str(csv), "-o", str(tmp_path / "out.csv"),
               "--batch-size", "100", "--report", str(reports), "--quiet"])

    assert rc == 0
    batch_files = sorted(reports.glob("batch_*.json"))
    assert len(batch_files) == 3  # 300 rows / 100 per batch
    assert (reports / "summary.json").exists()

    first = json.loads(batch_files[0].read_text())
    assert first["streaming"]["batch_id"] == 1
    summary = json.loads((reports / "summary.json").read_text())
    assert summary["streaming"]["rows_seen_total"] == 300


def test_cli_stream_trust_gate_sets_nonzero_exit(tmp_path):
    n = 200
    # A ~70%-missing column is preserved, holding trust well below the 99 threshold.
    df = pd.DataFrame({
        "customer_id": range(n),
        "amount": [1.0] * n,
        "sparse": ([None] * 7 + [1.0, 2.0, 3.0]) * (n // 10),
    })
    csv = tmp_path / "in.csv"
    df.to_csv(csv, index=False)

    rc = main(["stream", str(csv), "--batch-size", "100", "--id-columns", "customer_id",
               "--fail-under-trust", "99", "--warmup-batches", "0", "--quiet"])
    assert rc == 1


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None, reason="pyarrow not installed"
)
def test_cli_stream_parquet_roundtrip(tmp_path):
    import pyarrow.parquet as pq

    src = pd.DataFrame({"customer_id": range(150), "amount": [1.0, None, 3.0] * 50})
    in_pq = tmp_path / "in.parquet"
    src.to_parquet(in_pq)
    out_pq = tmp_path / "out.parquet"

    rc = main(["stream", str(in_pq), "-o", str(out_pq),
               "--batch-size", "50", "--warmup-batches", "1", "--quiet"])
    assert rc == 0
    assert pq.ParquetFile(out_pq).metadata.num_rows == 150


# -- benchmark generator is lazy (#20) ----------------------------------------------


@pytest.fixture()
def bench_module():
    sys.path.insert(0, str(BENCH_DIR))
    try:
        import bench_streaming  # type: ignore[import-not-found]

        yield bench_streaming
    finally:
        sys.path.remove(str(BENCH_DIR))
        sys.modules.pop("bench_streaming", None)


def test_benchmark_generator_does_not_materialize_all_rows(bench_module):
    # Ask for a billion rows: a non-lazy generator would hang/OOM building them all.
    gen = bench_module.synth_batches(rows=1_000_000_000, batch_size=1000, cols=12)
    assert isinstance(gen, types.GeneratorType)

    # Pulling three batches yields only batch_size rows each — the rest is never built.
    batches = [next(gen) for _ in range(3)]
    assert all(len(b) == 1000 for b in batches)
    assert all(b.shape[1] == 12 for b in batches)
    # Early stop is possible precisely because nothing downstream was materialized.
    gen.close()


def test_benchmark_generator_emits_exact_row_count(bench_module):
    total = sum(len(b) for b in bench_module.synth_batches(rows=2500, batch_size=1000, cols=8))
    assert total == 2500


# -- benchmark-stream CLI command ---------------------------------------------------


def test_cli_benchmark_stream_writes_json(tmp_path):
    report = tmp_path / "bench.json"
    rc = main(["benchmark-stream", "--rows", "2000", "--batch-size", "1000",
               "--cols", "6", "--report", str(report)])
    assert rc == 0
    result = json.loads(report.read_text())
    assert result["rows_processed"] == 2000
    assert result["stable_memory"] is True
    assert 0.0 <= result["final_cumulative_trust_score"] <= 100.0


# -- stream-kafka CLI command (mock consumer) ---------------------------------------


def test_cli_stream_kafka_with_mock(tmp_path, monkeypatch):
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

    reports = tmp_path / "reports"
    rc = main(["stream-kafka", "--topic", "events", "--bootstrap-servers", "localhost:9092",
               "--batch-size", "100", "--max-batches", "2", "--id-columns", "customer_id",
               "-o", str(tmp_path / "out.csv"), "--report", str(reports), "--quiet"])

    assert rc == 0
    assert (reports / "summary.json").exists()
    assert len(list(reports.glob("batch_*.json"))) == 2  # capped by --max-batches
