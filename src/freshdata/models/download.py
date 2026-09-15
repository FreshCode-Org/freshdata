"""Explicit model downloads (``fd.models.pull``).

This module is the only place in the package that may touch the network, and
nothing in the cleaning path imports it: models are downloaded when — and only
when — the user calls :func:`pull` (or the ``freshdata models pull`` CLI).
``fd.clean`` can therefore never download anything.
"""

from __future__ import annotations

import math
import os
import tempfile
import urllib.request
from pathlib import Path

from .registry import (
    _required_checksums,
    _sha256_file,
    check_lib_version,
    get_config,
    model_dir,
    verify,
)
from .types import ModelChecksumError, ModelNotPublishedError

_URL_BASE_ENV = "FRESHDATA_MODEL_URL_BASE"
#: Empty until official artifacts are hosted; see docs/semantic-models.md.
_DEFAULT_URL_BASE = ""
_TIMEOUT_ENV = "FRESHDATA_MODEL_TIMEOUT"
_DEFAULT_TIMEOUT = "60"
_CHUNK_SIZE = 1024 * 1024


def _url_base() -> str:
    return os.environ.get(_URL_BASE_ENV, _DEFAULT_URL_BASE)


def _timeout() -> float:
    """Network timeout in seconds (``FRESHDATA_MODEL_TIMEOUT``, default 60)."""
    raw = os.environ.get(_TIMEOUT_ENV, _DEFAULT_TIMEOUT)
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{_TIMEOUT_ENV} must be a positive number of seconds, got {raw!r}")
    return value


def _content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    raw = headers.get("Content-Length") if headers is not None else None
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``. The single network seam (mocked in tests).

    Raises :class:`OSError` when the transfer ends before the advertised
    ``Content-Length`` (the caller discards the partial file).
    """
    timeout = _timeout()
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        expected = _content_length(response)
        written = 0
        with dest.open("wb") as fh:
            while True:
                chunk = response.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
    if expected is not None and written != expected:
        raise OSError(f"incomplete download of {url}: got {written} of {expected} bytes")


def pull(model_id: str, *, force: bool = False) -> Path:
    """Explicitly download ``model_id`` into the local model directory.

    When the model has pinned checksums, every downloaded file is checked
    against its own pin before it replaces the installed file, and a
    mismatching download is discarded. When every file is already present
    and ``force`` is False, the installed files are verified with
    :func:`~freshdata.models.registry.verify` before returning; a mismatch
    raises :class:`ModelChecksumError` and leaves the files in place (re-run
    with ``force=True`` to download them again). Returns the primary
    artifact path. Raises
    :class:`ModelNotPublishedError` while no download location exists — the
    air-gapped path (drop files into ``FRESHDATA_MODEL_DIR``) always works.
    """
    cfg = get_config(model_id)
    check_lib_version(model_id)
    base = _url_base()
    if not base:
        raise ModelNotPublishedError(
            f"Model {model_id!r} has no published download location yet. "
            f"Set {_URL_BASE_ENV} to a mirror hosting the artifacts, or place "
            f"{', '.join(cfg.files)} under {model_dir() / model_id} manually "
            "(see docs/semantic-models.md)."
        )

    pins = _required_checksums(cfg)
    target_dir = model_dir() / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    base_url = base.rstrip("/") + "/" + cfg.url.rsplit("/", 1)[0]

    primary = target_dir / cfg.files[0]
    if primary.is_file() and not force and all((target_dir / f).is_file() for f in cfg.files):
        verify(model_id)
        return primary

    for name in cfg.files:
        dest = target_dir / name
        handle, tmp_name = tempfile.mkstemp(dir=str(target_dir), suffix=".part")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            _fetch(f"{base_url}/{name}", tmp)
            expected = pins.get(name)
            if expected is not None:
                actual = _sha256_file(tmp)
                if actual != expected:
                    raise ModelChecksumError(
                        f"Downloaded {model_id!r} file {name!r} failed checksum "
                        f"verification: expected {expected}, got {actual}. The partial "
                        "download was discarded."
                    )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    return primary
