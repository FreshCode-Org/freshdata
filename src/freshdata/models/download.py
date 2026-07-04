"""Explicit model downloads (``fd.models.pull``).

This module is the only place in the package that may touch the network, and
nothing in the cleaning path imports it: models are downloaded when — and only
when — the user calls :func:`pull` (or the ``freshdata models pull`` CLI).
``fd.clean`` can therefore never download anything.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .registry import _sha256_file, check_lib_version, get_config, model_dir
from .types import ModelChecksumError, ModelNotPublishedError

_URL_BASE_ENV = "FRESHDATA_MODEL_URL_BASE"
#: Empty until official artifacts are hosted; see docs/semantic-models.md.
_DEFAULT_URL_BASE = ""


def _url_base() -> str:
    return os.environ.get(_URL_BASE_ENV, _DEFAULT_URL_BASE)


def _fetch(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest``. The single network seam (mocked in tests)."""
    with urllib.request.urlopen(url) as response, dest.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(response, fh)


def pull(model_id: str, *, force: bool = False) -> Path:
    """Explicitly download ``model_id`` into the local model directory.

    Verifies the registry checksum when one is pinned and refuses to keep a
    mismatching artifact. Returns the primary artifact path. Raises
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

    target_dir = model_dir() / model_id
    target_dir.mkdir(parents=True, exist_ok=True)
    base_url = base.rstrip("/") + "/" + cfg.url.rsplit("/", 1)[0]

    primary = target_dir / cfg.files[0]
    if primary.is_file() and not force and all((target_dir / f).is_file() for f in cfg.files):
        return primary

    for name in cfg.files:
        dest = target_dir / name
        handle, tmp_name = tempfile.mkstemp(dir=str(target_dir), suffix=".part")
        os.close(handle)
        tmp = Path(tmp_name)
        try:
            _fetch(f"{base_url}/{name}", tmp)
            if name == cfg.files[0] and cfg.sha256 is not None:
                actual = _sha256_file(tmp)
                if actual != cfg.sha256:
                    raise ModelChecksumError(
                        f"Downloaded {model_id!r} failed checksum verification: "
                        f"expected {cfg.sha256}, got {actual}. The partial download "
                        "was discarded."
                    )
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
    return primary
