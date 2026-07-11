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

from .._util import sanitize_csv_formulas
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

    def __init__(self, path: str | None, sanitize_formulas: bool = False) -> None:
        self.path = path
        self.fmt = None if path is None else ("parquet"
                    if path.lower().endswith((".parquet", ".pq")) else "csv")
        self.sanitize_formulas = sanitize_formulas
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
            if self.sanitize_formulas:
                df = sanitize_csv_formulas(df)
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
    tcfg = _timeseries_config(args)
    if tcfg is not None:
        opts["time_series_config"] = tcfg
    return opts


def _timeseries_config(args: argparse.Namespace) -> Any:
    """Build a ``TimeSeriesCleanConfig`` from CLI flags, or ``None`` if ``--timestamp``
    was not given (so plain streaming is unaffected)."""
    timestamp = getattr(args, "timestamp", None)
    if not timestamp:
        return None
    from ._timeseries import TimeSeriesCleanConfig

    kwargs: dict[str, Any] = {"timestamp_column": timestamp}
    if getattr(args, "entity_id", None):
        kwargs["entity_id_columns"] = tuple(args.entity_id)
    if getattr(args, "watermark", None):
        kwargs["event_time_column"] = args.watermark
    if getattr(args, "allowed_lateness", None):
        kwargs["allowed_lateness"] = args.allowed_lateness
    kwargs["late_data_action"] = getattr(args, "late_data_action", "quarantine")
    if getattr(args, "max_interpolation_gap", None) is not None:
        kwargs["max_interpolation_gap"] = args.max_interpolation_gap
    if getattr(args, "interpolation_method", None):
        kwargs["interpolation_method"] = args.interpolation_method
    if getattr(args, "seasonal_period", None):
        kwargs["seasonal_period"] = args.seasonal_period
        kwargs["seasonal_imputation_enabled"] = True
    if getattr(args, "ordered_dedupe_keys", None):
        kwargs["ordered_dedupe_keys"] = tuple(args.ordered_dedupe_keys)
        kwargs["ordered_dedupe_keep"] = getattr(args, "dedupe_keep", "latest_event_time")
    if getattr(args, "anomaly", None):
        kwargs["anomaly_method"] = args.anomaly
        kwargs["anomaly_window_size"] = getattr(args, "anomaly_window", 50)
        kwargs["anomaly_action"] = getattr(args, "anomaly_action", "flag")
    return TimeSeriesCleanConfig(**kwargs)


def _write_exceptions(cleaner: StreamingCleaner, path: str | None,
                      sanitize_formulas: bool = False) -> None:
    """Persist any quarantined (late/anomalous) rows to *path* (CSV or Parquet)."""
    if not path:
        return
    exc = cleaner.exceptions_
    if exc is None or not len(exc):
        return
    if path.lower().endswith((".parquet", ".pq")):
        exc.to_parquet(path, index=False)
    else:
        if sanitize_formulas:
            exc = sanitize_csv_formulas(exc)
        exc.to_csv(path, index=False)


def _run_stream(cleaner: StreamingCleaner, batches: Iterator[pd.DataFrame],
                writer: _BatchWriter, report_dir: str | None, quiet: bool,
                quarantine_path: str | None = None,
                sanitize_formulas: bool = False) -> int:
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
    _write_exceptions(cleaner, quarantine_path, sanitize_formulas=sanitize_formulas)
    if report_dir:
        with open(os.path.join(report_dir, "summary.json"), "w") as fh:
            json.dump(final.to_dict(), fh, default=str)
    if not quiet:
        print(json.dumps({"summary": final.streaming}))
    return 1 if cleaner._gate_failures else 0


def cmd_stream(args: argparse.Namespace) -> int:
    cleaner = StreamingCleaner(**_stream_options(args))
    sanitize = getattr(args, "sanitize_formulas", False)
    return _run_stream(cleaner, _read_chunks(args.input, args.batch_size),
                       _BatchWriter(args.output, sanitize_formulas=sanitize),
                       args.report, args.quiet,
                       getattr(args, "quarantine", None),
                       sanitize_formulas=sanitize)


def cmd_stream_kafka(args: argparse.Namespace) -> int:
    cleaner = StreamingCleaner(**_stream_options(args))
    batches = cleaner.clean_kafka(  # validates the kafka dependency
        topic=args.topic, bootstrap_servers=args.bootstrap_servers,
        batch_size=args.batch_size, max_batches=args.max_batches)
    writer = _BatchWriter(args.output,
                          sanitize_formulas=getattr(args, "sanitize_formulas", False))
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


def _add_timeseries_args(p: argparse._ActionsContainer) -> None:
    """Add the time-series / streaming-aware flags to a stream subparser."""
    g = p.add_argument_group("time-series mode (enabled by --timestamp)")
    g.add_argument("--timestamp", metavar="COL",
                   help="timestamp column; enables time-series cleaning")
    g.add_argument("--entity-id", nargs="*", default=(), metavar="COL",
                   help="entity id column(s) defining independent series")
    g.add_argument("--watermark", metavar="COL",
                   help="event-time column used for the late-data watermark")
    g.add_argument("--allowed-lateness", metavar="DUR",
                   help="how far behind the watermark an event may still arrive (e.g. 10m)")
    g.add_argument("--late-data-action", default="quarantine",
                   choices=("quarantine", "keep_with_warning", "drop"))
    g.add_argument("--max-interpolation-gap", type=int, metavar="N",
                   help="fill missing runs no longer than N steps")
    g.add_argument("--interpolation-method", default="time",
                   choices=("linear", "time", "ffill", "bfill"))
    g.add_argument("--seasonal-period", metavar="PERIOD",
                   choices=("hour", "day", "dayofweek", "week", "month"),
                   help="enable seasonal imputation with this bucket")
    g.add_argument("--ordered-dedupe-keys", nargs="*", default=(), metavar="COL")
    g.add_argument("--dedupe-keep", default="latest_event_time",
                   choices=("first", "last", "latest_event_time", "highest_quality"))
    g.add_argument("--anomaly", metavar="METHOD",
                   choices=("rolling_zscore", "mad", "iqr", "ewma"),
                   help="enable windowed anomaly detection with this method")
    g.add_argument("--anomaly-window", type=int, default=50, metavar="N")
    g.add_argument("--anomaly-action", default="flag",
                   choices=("flag", "cap", "quarantine"))
    g.add_argument("--quarantine", metavar="FILE",
                   help="write late/anomalous rows pulled out of the stream here")


def add_stream_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the streaming subcommands on the shared ``freshdata`` parser."""
    s = subparsers.add_parser("stream", help="clean a CSV/Parquet file in micro-batches")
    s.add_argument("input")
    s.add_argument("-o", "--output")
    s.add_argument("--sanitize-formulas", action="store_true",
                   help="prefix ' to string cells starting with = + - @ tab/CR in "
                        "CSV outputs so spreadsheets render them as text (OWASP)")
    s.add_argument("--batch-size", "--chunksize", type=int, default=100_000, dest="batch_size")
    s.add_argument("--report", metavar="DIR", help="directory for per-batch + summary JSON")
    s.add_argument("--target-column")
    s.add_argument("--id-columns", nargs="*", default=())
    s.add_argument("--strategy", default="balanced",
                   choices=("conservative", "balanced", "aggressive"))
    s.add_argument("--warmup-batches", type=int, default=3)
    s.add_argument("--fail-under-trust", type=float, metavar="SCORE")
    s.add_argument("--quiet", action="store_true")
    _add_timeseries_args(s)
    s.set_defaults(func=cmd_stream)

    k = subparsers.add_parser("stream-kafka", help="clean a Kafka topic in micro-batches")
    k.add_argument("--topic", required=True)
    k.add_argument("--bootstrap-servers", required=True)
    k.add_argument("--batch-size", type=int, default=10_000)
    k.add_argument("--max-batches", type=int)
    k.add_argument("-o", "--output")
    k.add_argument("--sanitize-formulas", action="store_true",
                   help="prefix ' to string cells starting with = + - @ tab/CR in "
                        "CSV outputs so spreadsheets render them as text (OWASP)")
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
