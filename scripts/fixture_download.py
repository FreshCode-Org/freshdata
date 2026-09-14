"""Network download with retries for the online fixture tooling.

Shared by ``scripts/fetch_online_fixtures.py`` and the live-fetch test helper so
both survive transient upstream failures (connection resets, dropped responses,
throttling, 5xx) the same way.
"""

from __future__ import annotations

import http.client
import random
import time
import urllib.error
import urllib.request
from typing import Callable

USER_AGENT = "freshdata-fixture-fetch/1.0"


class TransientFetchError(RuntimeError):
    """A download kept failing with transient network errors after every retry."""


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    # URLError (DNS, refused, reset), RemoteDisconnected / IncompleteRead
    # (HTTPException) and socket timeouts are all worth another attempt.
    return isinstance(exc, (OSError, http.client.HTTPException))


def download(
    url: str,
    *,
    attempts: int = 4,
    timeout: float = 120.0,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """Return the body of *url*, retrying transient failures with backoff.

    Permanent HTTP errors (4xx other than 429) are raised immediately; after
    *attempts* transient failures a :class:`TransientFetchError` is raised.
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return resp.read()
        except (OSError, http.client.HTTPException) as exc:
            if not _is_transient(exc):
                raise
            if attempt == attempts:
                raise TransientFetchError(
                    f"{url}: {type(exc).__name__}: {exc} (after {attempts} attempts)"
                ) from exc
            sleep(base_delay * 2 ** (attempt - 1) + random.uniform(0, 1))  # noqa: S311
    raise AssertionError("unreachable: the loop always returns or raises")
