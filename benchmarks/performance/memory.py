from __future__ import annotations

import gc
import threading
from typing import Optional


class PeakRSS:
    def __init__(self, interval_seconds: float = 0.01) -> None:
        import psutil  # noqa: PLC0415

        self._process = psutil.Process()
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None  # noqa: UP045
        self._baseline = 0
        self._peak = 0

    def __enter__(self) -> PeakRSS:
        gc.collect()
        self._baseline = self._process.memory_info().rss
        self._peak = self._baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self) -> None:
        while not self._stop.wait(self._interval):
            self._peak = max(self._peak, self._process.memory_info().rss)

    def __exit__(self, *_args: object) -> None:
        self._peak = max(self._peak, self._process.memory_info().rss)
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def increase_bytes(self) -> int:
        return max(0, self._peak - self._baseline)
