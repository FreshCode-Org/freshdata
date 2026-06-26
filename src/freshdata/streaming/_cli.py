"""CLI handlers for streaming mode (wired into the ``freshdata`` entry point).

Everything here streams: input is read batch-by-batch (``pd.read_csv(chunksize=)`` /
``ParquetFile.iter_batches``) and output is written batch-by-batch
(``ParquetWriter`` row groups / appended CSV), so a 100M-row file is never held in
memory at once. Per-batch JSON reports plus a final summary are written when ``--report``
is given, and the trust gate sets the process exit code.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterator
from typing import Any

import pandas as pd

from ._cleaner import StreamingCleaner


def _read_chunks(path: str, batch_size: int) -> Iterator[pd.DataFrame]:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, chunksize=batch_size)


class _BatchWriter:
    """Append cleaned batches to one CSV or Parquet file without buffering them all."""

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.fmt = None if path is None else ("parquet"
                    if path.lower().endswith((".parquet", ".pq")) else "csv")
        self._pq_writer: Any = None
        self._csv_header = True

    def write(self, df: pd.DataFrame) -> None:
        if self.path is None:
            return
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._pq_writer is None:
                self._pq_writer = pq.ParquetWriter(self.path, table.schema)
            self._pq_writer.write_table(table)
        else:
            df.to_csv(self.path, mode="w" if self._csv_header else "a",
                      header=self._csv_header, index=False)
            self._csv_header = False

    def close(self) -> None:
        if self._pq_writer is not None:
            self._pq_writer.close()


def _stream_options(args: argparse.Namespace) -> dict[str, Any]:
    opts: dict[str, Any] = {"warmup_batches": args.warmup_batches, "strategy": args.strategy}
    if args.target_column:
        opts["target_column"] = args.target_column
    if args.id_columns:
        opts["id_columns"] = tuple(args.id_columns)
    if args.fail_under_trust is not None:
        opts["fail_under_trust"] = args.fail_under_trust
    return opts


def _run_stream(cleaner: StreamingCleaner, batches: Iterator[pd.DataFrame],
                writer: _BatchWriter, report_dir: str | None, quiet: bool) -> int:
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    for cleaned, report in cleaner.clean_batches(batches):
        writer.write(cleaned)
        if report_dir:
            bid = (report.streaming or {})["batch_id"]
            with open(os.path.join(report_dir, f"batch_{bid:06d}.json"), "w") as fh:
                json.dump(report.to_dict(), fh, default=str)
        if not quiet:
            print(json.dumps(report.streaming))
    writer.close()
    final = cleaner.finalize()
    if report_dir:
        with open(os.path.join(report_dir, "summary.json"), "w") as fh:
            json.dump(final.to_dict(), fh, default=str)
    if not quiet:
        print(json.dumps({"summary": final.streaming}))
    return 1 if cleaner._gate_failures else 0


def cmd_stream(args: argparse.Namespace) -> int:
    cleaner = StreamingCleaner(**_stream_options(args))
    return _run_stream(cleaner, _read_chunks(args.input, args.batch_size),
                       _BatchWriter(args.output), args.report, args.quiet)


def cmd_stream_kafka(args: argparse.Namespace) -> int:
    cleaner = StreamingCleaner(**_stream_options(args))
    batches = cleaner.clean_kafka(  # validates the kafka dependency
        topic=args.topic, bootstrap_servers=args.bootstrap_servers,
        batch_size=args.batch_size, max_batches=args.max_batches)
    writer = _BatchWriter(args.output)
    if args.report:
        os.makedirs(args.report, exist_ok=True)
    for cleaned, report in batches:
        writer.write(cleaned)
        if args.report:
            bid = (report.streaming or {})["batch_id"]
            with open(os.path.join(args.report, f"batch_{bid:06d}.json"), "w") as fh:
                json.dump(report.to_dict(), fh, default=str)
        if not args.quiet:
            print(json.dumps(report.streaming))
    writer.close()
    if args.report:
        with open(os.path.join(args.report, "summary.json"), "w") as fh:
            json.dump(cleaner.finalize().to_dict(), fh, default=str)
    return 1 if cleaner._gate_failures else 0


def cmd_benchmark_stream(args: argparse.Namespace) -> int:
    try:
        import sys

        sys.path.insert(0, os.path.join(os.getcwd(), "benchmarks"))
        from bench_streaming import run_benchmark  # type: ignore[import-not-found]
    except ImportError as exc:  # installed wheel has no benchmarks/ dir
        raise SystemExit(
            "benchmark-stream needs the benchmarks/ directory; run it from a source "
            "checkout or invoke 'python benchmarks/bench_streaming.py' directly"
        ) from exc

    result = run_benchmark(rows=args.rows, batch_size=args.batch_size, cols=args.cols)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    return 0


def add_stream_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the streaming subcommands on the shared ``freshdata`` parser."""
    s = subparsers.add_parser("stream", help="clean a CSV/Parquet file in micro-batches")
    s.add_argument("input")
    s.add_argument("-o", "--output")
    s.add_argument("--batch-size", "--chunksize", type=int, default=100_000, dest="batch_size")
    s.add_argument("--report", metavar="DIR", help="directory for per-batch + summary JSON")
    s.add_argument("--target-column")
    s.add_argument("--id-columns", nargs="*", default=())
    s.add_argument("--strategy", default="balanced",
                   choices=("conservative", "balanced", "aggressive"))
    s.add_argument("--warmup-batches", type=int, default=3)
    s.add_argument("--fail-under-trust", type=float, metavar="SCORE")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(func=cmd_stream)

    k = subparsers.add_parser("stream-kafka", help="clean a Kafka topic in micro-batches")
    k.add_argument("--topic", required=True)
    k.add_argument("--bootstrap-servers", required=True)
    k.add_argument("--batch-size", type=int, default=10_000)
    k.add_argument("--max-batches", type=int)
    k.add_argument("-o", "--output")
    k.add_argument("--report", metavar="DIR")
    k.add_argument("--target-column")
    k.add_argument("--id-columns", nargs="*", default=())
    k.add_argument("--strategy", default="balanced",
                   choices=("conservative", "balanced", "aggressive"))
    k.add_argument("--warmup-batches", type=int, default=3)
    k.add_argument("--fail-under-trust", type=float, metavar="SCORE")
    k.add_argument("--quiet", action="store_true")
    k.set_defaults(func=cmd_stream_kafka)

    b = subparsers.add_parser("benchmark-stream", help="run the stable-memory streaming benchmark")
    b.add_argument("--rows", type=int, default=1_000_000)
    b.add_argument("--batch-size", type=int, default=100_000)
    b.add_argument("--cols", type=int, default=20)
    b.add_argument("--report", metavar="FILE", help="write the benchmark JSON here")
    b.set_defaults(func=cmd_benchmark_stream)
