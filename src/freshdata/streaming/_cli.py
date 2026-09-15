"""CLI handlers for streaming mode (wired into the ``freshdata`` entry point).

Everything here streams: input is read batch-by-batch (``pd.read_csv(chunksize=)`` /
``ParquetFile.iter_batches``) and output is written batch-by-batch
(``ParquetWriter`` row groups / appended CSV), so a 100M-row file is never held in
memory at once. Per-batch JSON reports plus a final summary are written when ``--report``
is given, and the trust gate sets the process exit code.
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
from collections.abc import Iterator
from typing import Any

import pandas as pd

from .._csv_io import leading_zero_dtypes
from .._util import sanitize_csv_formulas
from ._cleaner import StreamingCleaner


def _read_chunks(path: str, batch_size: int,
                 preserve_leading_zeros: bool = True) -> Iterator[pd.DataFrame]:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
            yield batch.to_pandas()
    else:
        # Probe the first chunk once for zero-padded numeric columns (ZIP codes, IDs)
        # and read them as text in *every* chunk, so types never flip between batches.
        dtype = (leading_zero_dtypes(path, nrows=batch_size)
                 if preserve_leading_zeros else {})
        if dtype:
            yield from pd.read_csv(path, chunksize=batch_size, dtype=dtype)
        else:
            yield from pd.read_csv(path, chunksize=batch_size)


class _BatchWriter:
    """Append cleaned batches to one CSV or Parquet file without buffering them all.

    The first batch fixes the output layout: later CSV batches are reindexed to its
    columns (a column missing from a batch is written empty) and later Parquet
    tables are cast to its schema. A batch with a column the first batch did not
    have, or with values that cannot be cast, raises :class:`ValueError`.

    Batches go to a sibling ``<path>.partial`` file that :meth:`commit` moves onto
    *path*; :meth:`abort` deletes it, so a failed run never leaves a truncated file
    at *path*.
    """

    def __init__(self, path: str | None, sanitize_formulas: bool = True) -> None:
        self.path = path
        self.fmt = None if path is None else ("parquet"
                    if path.lower().endswith((".parquet", ".pq")) else "csv")
        self.partial_path = None if path is None else f"{path}.partial"
        self.sanitize_formulas = sanitize_formulas
        self._pq_writer: Any = None
        self._schema: Any = None
        self._columns: list[Any] | None = None
        self._started = False

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._columns is None:
            self._columns = list(df.columns)
            return df
        if list(df.columns) == self._columns:
            return df
        known = set(self._columns)
        extra = [c for c in df.columns if c not in known]
        if extra:
            raise ValueError(
                f"stream batch has column(s) {extra!r} that the first batch did not; "
                f"output columns are fixed by the first batch: {self._columns!r}"
            )
        return df.reindex(columns=self._columns)

    def write(self, df: pd.DataFrame) -> None:
        if self.path is None or self.partial_path is None:
            return
        df = self._align(df)
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df, preserve_index=False)
            if self._pq_writer is None:
                self._schema = table.schema
                self._started = True
                self._pq_writer = pq.ParquetWriter(self.partial_path, self._schema)
            elif not table.schema.equals(self._schema, check_metadata=False):
                try:
                    table = table.cast(self._schema, safe=True)
                except (pa.ArrowException, ValueError, TypeError) as exc:
                    raise ValueError(
                        f"stream batch does not fit the Parquet schema set by the first "
                        f"batch ({exc}); expected schema:\n{self._schema.remove_metadata()}"
                    ) from exc
            self._pq_writer.write_table(table)
        else:
            if self.sanitize_formulas:
                df = sanitize_csv_formulas(df)
            first = not self._started
            self._started = True
            df.to_csv(self.partial_path, mode="w" if first else "a",
                      header=first, index=False)

    def close(self) -> None:
        """Release the Parquet writer (idempotent); does not move the output."""
        if self._pq_writer is not None:
            writer, self._pq_writer = self._pq_writer, None
            writer.close()

    def commit(self) -> None:
        """Finish the output: close it and move ``<path>.partial`` onto *path*."""
        self.close()
        if self._started and self.path is not None and self.partial_path is not None:
            os.replace(self.partial_path, self.path)

    def abort(self) -> None:
        """Discard the output: close it and delete ``<path>.partial`` if present."""
        try:
            self.close()
        finally:
            if self.partial_path is not None:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(self.partial_path)


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


def _column_options(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """The column-name flags given on the command line, as ``(flag, [names])``.

    The time-series flags only count when ``--timestamp`` is given, because
    without it they are not used at all.
    """
    opts: list[tuple[str, list[str]]] = []
    if getattr(args, "target_column", None):
        opts.append(("--target-column", [args.target_column]))
    if getattr(args, "id_columns", None):
        opts.append(("--id-columns", list(args.id_columns)))
    if getattr(args, "timestamp", None):
        opts.append(("--timestamp", [args.timestamp]))
        if getattr(args, "watermark", None):
            opts.append(("--watermark", [args.watermark]))
        if getattr(args, "entity_id", None):
            opts.append(("--entity-id", list(args.entity_id)))
        if getattr(args, "ordered_dedupe_keys", None):
            opts.append(("--ordered-dedupe-keys", list(args.ordered_dedupe_keys)))
    return opts


def _check_column_options(columns: Any, args: argparse.Namespace) -> None:
    """Raise :class:`ValueError` if a column-name flag names no column in *columns*."""
    present = list(columns)
    names = [str(c) for c in present]
    problems: list[str] = []
    for flag, wanted in _column_options(args):
        for name in wanted:
            if name in present:
                continue
            match = difflib.get_close_matches(name, names, n=1)
            hint = f" (did you mean {match[0]!r}?)" if match else ""
            problems.append(f"{flag} column {name!r} not found in input{hint}")
    if problems:
        shown = ", ".join(names[:20]) + (", ..." if len(names) > 20 else "")
        raise ValueError(f"{'; '.join(problems)}; input columns: {shown}")


def _checked_batches(batches: Iterator[pd.DataFrame],
                     args: argparse.Namespace) -> Iterator[pd.DataFrame]:
    """Pass *batches* through, checking the column-name flags against the first one.

    The check runs before the first batch is cleaned or written, so a bad column
    name stops the run before any output (or ``.partial`` file) exists.
    """
    first = True
    for batch in batches:
        if first:
            first = False
            _check_column_options(batch.columns, args)
        yield batch


def _write_exceptions(cleaner: StreamingCleaner, path: str | None,
                      sanitize_formulas: bool = True) -> None:
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
                sanitize_formulas: bool = True) -> int:
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    committed = False
    try:
        for cleaned, report in cleaner.clean_batches(batches):
            writer.write(cleaned)
            if report_dir:
                bid = (report.streaming or {})["batch_id"]
                with open(os.path.join(report_dir, f"batch_{bid:06d}.json"), "w") as fh:
                    json.dump(report.to_dict(), fh, default=str)
            if not quiet:
                print(json.dumps(report.streaming))
        writer.commit()
        committed = True
    finally:
        if not committed:
            writer.abort()
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
    sanitize = getattr(args, "sanitize_formulas", True)
    batches = _checked_batches(
        _read_chunks(args.input, args.batch_size,
                     preserve_leading_zeros=cleaner.config.preserve_leading_zeros),
        args)
    return _run_stream(cleaner, batches,
                       _BatchWriter(args.output, sanitize_formulas=sanitize),
                       args.report, args.quiet,
                       getattr(args, "quarantine", None),
                       sanitize_formulas=sanitize)


def cmd_stream_kafka(args: argparse.Namespace) -> int:
    from ._connectors import kafka_batches

    cleaner = StreamingCleaner(**_stream_options(args))
    # Same source as ``cleaner.clean_kafka`` (which validates the kafka dependency),
    # with the column-name flags checked against the first raw batch.
    batches = cleaner.clean_batches(_checked_batches(kafka_batches(
        topic=args.topic, bootstrap_servers=args.bootstrap_servers,
        batch_size=args.batch_size, max_batches=args.max_batches), args))
    writer = _BatchWriter(args.output,
                          sanitize_formulas=getattr(args, "sanitize_formulas", True))
    if args.report:
        os.makedirs(args.report, exist_ok=True)
    committed = False
    try:
        for cleaned, report in batches:
            writer.write(cleaned)
            if args.report:
                bid = (report.streaming or {})["batch_id"]
                with open(os.path.join(args.report, f"batch_{bid:06d}.json"), "w") as fh:
                    json.dump(report.to_dict(), fh, default=str)
            if not args.quiet:
                print(json.dumps(report.streaming))
        writer.commit()
        committed = True
    finally:
        if not committed:
            writer.abort()
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
    s.add_argument("--sanitize-formulas", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="prefix ' to string cells starting with = + - @ tab/CR in "
                        "CSV outputs so spreadsheets render them as text (OWASP); "
                        "on by default, --no-sanitize-formulas writes byte-exact CSV")
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
    k.add_argument("--sanitize-formulas", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="prefix ' to string cells starting with = + - @ tab/CR in "
                        "CSV outputs so spreadsheets render them as text (OWASP); "
                        "on by default, --no-sanitize-formulas writes byte-exact CSV")
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
