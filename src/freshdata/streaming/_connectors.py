"""Input coercion and optional streaming source connectors.

``coerce_to_pandas`` accepts pandas / polars / pyarrow batches without importing the
optional libraries unless such a batch is actually seen. Kafka and Arrow Flight are
optional integrations: importing them lazily keeps them out of the base install, and a
missing dependency raises a clear, actionable :class:`ImportError`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd


def coerce_to_pandas(batch: Any) -> pd.DataFrame:
    """Return *batch* as a pandas DataFrame (pandas/polars/pyarrow accepted)."""
    if isinstance(batch, pd.DataFrame):
        return batch
    module = type(batch).__module__.split(".")[0]
    if module == "polars":
        if hasattr(batch, "collect"):  # LazyFrame
            batch = batch.collect()
        return batch.to_pandas()
    if module == "pyarrow":  # Table or RecordBatch both expose to_pandas()
        return batch.to_pandas()
    raise TypeError(
        f"cannot coerce a {type(batch).__name__} batch to a DataFrame; pass a pandas, "
        "polars, or pyarrow object"
    )


def kafka_batches(*, topic: str, bootstrap_servers: str, value_format: str = "json",
                  batch_size: int = 10_000, max_batches: int | None = None,
                  consumer_kwargs: dict[str, Any] | None = None) -> Iterator[pd.DataFrame]:
    """Yield DataFrame micro-batches from a Kafka *topic* (requires ``kafka-python``)."""
    try:
        from kafka import KafkaConsumer
    except ImportError as exc:
        raise ImportError(
            "Kafka streaming requires kafka-python. Install it with: "
            "pip install 'freshdata-cleaner[kafka]'"
        ) from exc
    if value_format != "json":
        raise ValueError(f"unsupported value_format {value_format!r}; only 'json' is supported")

    consumer = KafkaConsumer(topic, bootstrap_servers=bootstrap_servers,
                             **(consumer_kwargs or {}))
    rows: list[dict[str, Any]] = []
    emitted = 0
    for message in consumer:
        value = message.value
        rows.append(json.loads(value.decode("utf-8") if isinstance(value, bytes) else value))
        if len(rows) >= batch_size:
            yield pd.DataFrame(rows)
            rows = []
            emitted += 1
            if max_batches is not None and emitted >= max_batches:
                return
    if rows:
        yield pd.DataFrame(rows)


def flight_batches(location: str, *, descriptor: Any = None,
                   batch_size: int = 10_000) -> Iterator[pd.DataFrame]:
    """Yield DataFrame batches from an Arrow Flight endpoint (requires ``pyarrow.flight``)."""
    try:
        from pyarrow import flight
    except ImportError as exc:
        raise ImportError(
            "Arrow Flight streaming requires pyarrow with the flight module. Install it "
            "with: pip install 'freshdata-cleaner[flight]'"
        ) from exc

    client = flight.connect(location)
    if descriptor is None:
        descriptor = flight.FlightDescriptor.for_path()
    info = client.get_flight_info(descriptor)
    for endpoint in info.endpoints:
        reader = client.do_get(endpoint.ticket)
        for chunk in reader:
            table = chunk.data if hasattr(chunk, "data") else chunk
            df = table.to_pandas()
            for start in range(0, len(df), batch_size):
                yield df.iloc[start : start + batch_size]
