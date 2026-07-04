"""Teacher clients: a stub for offline development, a gated HTTP client.

Nothing in the runtime package touches this module. There is no vendored
provider SDK; the HTTP client is configured entirely by environment
variables and refuses to run without them **and** without an approved
compliance record (checked by the caller in ``tasks.py``).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Protocol


class TeacherUnavailable(RuntimeError):
    """The teacher client cannot run (unconfigured, offline, or refused)."""


class TeacherClient(Protocol):
    provider: str
    model: str

    def complete(self, prompt: str, *, schema: str) -> str:
        """Return the raw response text for one batched prompt."""
        ...


class StubTeacherClient:
    """Deterministic offline stand-in used by tests and dev pipelines.

    Responses come from a canned mapping keyed by schema; by default it
    returns an empty batch, which downstream tasks treat as "degrade to
    corruptor/hook data".
    """

    provider = "stub"
    model = "stub-teacher-0"

    def __init__(self, canned: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        self._canned = canned or {}
        self._fail = fail
        self.calls: list[dict[str, str]] = []

    def complete(self, prompt: str, *, schema: str) -> str:
        self.calls.append({"schema": schema, "prompt": prompt})
        if self._fail:
            raise TeacherUnavailable("stub teacher configured to fail")
        return json.dumps(self._canned.get(schema, []))


class HTTPTeacherClient:
    """Minimal JSON-over-HTTP teacher client (development-time only).

    Configuration (all required):

    - ``FRESHDATA_TEACHER_URL``      — full endpoint URL
    - ``FRESHDATA_TEACHER_PROVIDER`` — provider name for compliance/cache keys
    - ``FRESHDATA_TEACHER_MODEL``    — model name
    - ``FRESHDATA_TEACHER_API_KEY``  — bearer token (never committed)

    The request body is ``{"model": ..., "prompt": ..., "schema": ...}`` and
    the response body must be the raw JSON batch. Anything fancier belongs in
    a local adapter script, not in the repository.
    """

    def __init__(self) -> None:
        self.url = os.environ.get("FRESHDATA_TEACHER_URL", "")
        self.provider = os.environ.get("FRESHDATA_TEACHER_PROVIDER", "")
        self.model = os.environ.get("FRESHDATA_TEACHER_MODEL", "")
        self._api_key = os.environ.get("FRESHDATA_TEACHER_API_KEY", "")
        if not (self.url and self.provider and self.model and self._api_key):
            raise TeacherUnavailable(
                "teacher client not configured: set FRESHDATA_TEACHER_URL, "
                "FRESHDATA_TEACHER_PROVIDER, FRESHDATA_TEACHER_MODEL, "
                "FRESHDATA_TEACHER_API_KEY"
            )

    def complete(self, prompt: str, *, schema: str) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "schema": schema})
        request = urllib.request.Request(
            self.url,
            data=body.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read().decode("utf-8")
        except OSError as exc:
            raise TeacherUnavailable(f"teacher call failed: {exc}") from exc


def default_client() -> TeacherClient:
    """HTTP client when configured, else the empty stub (degraded mode)."""
    try:
        return HTTPTeacherClient()
    except TeacherUnavailable:
        return StubTeacherClient()
