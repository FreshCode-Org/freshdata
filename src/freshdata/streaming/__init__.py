"""Streaming / micro-batch cleaning: clean unbounded data in constant memory.

>>> import freshdata as fd
>>> cleaner = fd.StreamingCleaner(target_column="churn", id_columns=("customer_id",),
...                               window_size=100_000, warmup_batches=3)
>>> for cleaned_batch, report in cleaner.clean_batches(batch_iterable):  # doctest: +SKIP
...     write(cleaned_batch)
>>> final_report = cleaner.finalize()  # doctest: +SKIP

The cleaner keeps **bounded** running statistics across batches (Welford mean/variance,
reservoir-sampled medians, Space-Saving top-k categories), so memory stays flat whether
you feed it 100k rows or 100M. See :class:`StreamingCleaner`.
"""

from __future__ import annotations

from ._cleaner import StreamingCleaner
from ._config import StreamingCleanConfig
from ._state import ColumnState, StreamingState

__all__ = ["ColumnState", "StreamingCleanConfig", "StreamingCleaner", "StreamingState"]
