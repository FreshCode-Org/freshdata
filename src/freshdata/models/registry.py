"""Local model registry: what artifacts exist, where they live, their status.

Models are stored under ``~/.freshdata/models/<model_id>/`` (override with
``FRESHDATA_MODEL_DIR``). Nothing here touches the network — downloads live in
:mod:`freshdata.models.download` and only run when the user explicitly calls
:func:`freshdata.models.pull`. Air-gapped installs simply drop the files into
the model directory; :func:`status` detects them.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from .types import (
    ModelChecksumError,
    ModelConfig,
    ModelNotInstalledError,
    ModelVersionError,
    UnknownModelError,
)

_MODEL_DIR_ENV = "FRESHDATA_MODEL_DIR"
_DEFAULT_MODEL_DIR = Path.home() / ".freshdata" / "models"

#: Runtime-critical column/value encoder (embedding backend + resolver rung).
COL_ENCODER_ID = "fd-col-encoder-v1"
#: Context-sentence intent classifier. Registered for forward compatibility;
#: not consulted by any Phase-3 code path.
INTENT_ID = "fd-intent-v1"
#: Isotonic confidence-calibration tables (plain JSON, not a neural model).
CALIBRATION_ID = "calib-v1"

REGISTRY: dict[str, ModelConfig] = {
    COL_ENCODER_ID: ModelConfig(
        model_id=COL_ENCODER_ID,
        sha256=None,  # unpublished: pinned when the artifact is hosted
        size_bytes=26_000_000,
        license="Apache-2.0",
        url="fd-col-encoder-v1/model.onnx",
        min_lib_version="1.0.0",
        quantization="int8",
        files=("model.onnx", "tokenizer.json"),
    ),
    INTENT_ID: ModelConfig(
        model_id=INTENT_ID,
        sha256=None,  # unpublished
        size_bytes=2_000_000,
        license="Apache-2.0",
        url="fd-intent-v1/model.onnx",
        min_lib_version="1.0.0",
        quantization="int8",
        files=("model.onnx",),
    ),
    CALIBRATION_ID: ModelConfig(
        model_id=CALIBRATION_ID,
        sha256=None,  # packaged default ships in the wheel; a pull can override
        size_bytes=4_096,
        license="Apache-2.0",
        url="calib-v1/calibration.json",
        min_lib_version="1.0.0",
        quantization="json",
        files=("calibration.json",),
        packaged_default=True,
    ),
}


def model_dir() -> Path:
    """Return the local model directory without creating it."""
    override = os.environ.get(_MODEL_DIR_ENV)
    return Path(override).expanduser() if override else _DEFAULT_MODEL_DIR


def get_config(model_id: str) -> ModelConfig:
    """Return the registry entry for ``model_id`` or raise :class:`UnknownModelError`."""
    try:
        return REGISTRY[model_id]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise UnknownModelError(f"Unknown model id {model_id!r}. Known models: {known}") from None


def list_available() -> list[ModelConfig]:
    """Return every registered model's metadata, sorted by model id."""
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def _model_files(cfg: ModelConfig) -> list[Path]:
    base = model_dir() / cfg.model_id
    return [base / name for name in cfg.files]


def is_installed(model_id: str) -> bool:
    """Return True when every file of ``model_id`` exists locally."""
    cfg = get_config(model_id)
    files = _model_files(cfg)
    return bool(files) and all(f.is_file() for f in files)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(model_id: str) -> bool:
    """Verify the primary artifact's checksum against the registry pin.

    Returns True when pinned and matching, False when no hash is pinned
    (manually placed / unpublished artifacts load as "unverified"). Raises
    :class:`ModelChecksumError` on a pinned mismatch and
    :class:`ModelNotInstalledError` when files are missing.
    """
    cfg = get_config(model_id)
    if not is_installed(model_id):
        raise ModelNotInstalledError(
            f"Model {model_id!r} is not installed. Run fd.models.pull({model_id!r}) "
            f"or place its files under {model_dir() / model_id}."
        )
    if cfg.sha256 is None:
        return False
    actual = _sha256_file(_model_files(cfg)[0])
    if actual != cfg.sha256:
        raise ModelChecksumError(
            f"Checksum mismatch for {model_id!r}: expected {cfg.sha256}, got {actual}. "
            "Refusing to load; re-run fd.models.pull(..., force=True) or replace the file."
        )
    return True


def check_lib_version(model_id: str) -> None:
    """Raise :class:`ModelVersionError` when the library is too old for the model."""
    from freshdata import __version__  # noqa: PLC0415 - avoid import cycle at module load

    cfg = get_config(model_id)
    if _version_tuple(__version__) < _version_tuple(cfg.min_lib_version):
        raise ModelVersionError(
            f"Model {model_id!r} requires freshdata-cleaner>={cfg.min_lib_version}; "
            f"installed version is {__version__}. Upgrade the library to use this model."
        )


def _version_tuple(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in text.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def path(model_id: str) -> Path:
    """Return the path of the model's primary artifact.

    For installed models this is the file under :func:`model_dir`. The
    packaged calibration default is resolved via package data when no local
    override exists. Raises :class:`ModelNotInstalledError` otherwise.
    """
    cfg = get_config(model_id)
    if is_installed(model_id):
        return _model_files(cfg)[0]
    if cfg.packaged_default:
        packaged = _packaged_default_path(cfg)
        if packaged is not None:
            return packaged
    raise ModelNotInstalledError(
        f"Model {model_id!r} is not installed. Run fd.models.pull({model_id!r}) "
        f"or place {', '.join(cfg.files)} under {model_dir() / model_id}."
    )


def _packaged_default_path(cfg: ModelConfig) -> Path | None:
    if cfg.model_id != CALIBRATION_ID:  # pragma: no cover - only calib ships packaged
        return None
    candidate = Path(__file__).resolve().parent.parent / "semantic" / "data" / "calib_default.json"
    return candidate if candidate.is_file() else None


def status() -> dict[str, dict[str, Any]]:
    """Return install/verification status for every registered model.

    Purely local and side-effect free: no directories are created and no
    network is touched, so it is safe on air-gapped or read-only machines.
    """
    out: dict[str, dict[str, Any]] = {}
    for cfg in list_available():
        installed = is_installed(cfg.model_id)
        verified: bool | None = None
        note = ""
        if installed:
            try:
                verified = verify(cfg.model_id)
            except ModelChecksumError as exc:
                verified = False
                note = str(exc)
            else:
                if not verified:
                    note = "installed manually; no checksum pinned (unverified)"
        elif cfg.packaged_default and _packaged_default_path(cfg) is not None:
            installed = True
            verified = True
            note = "using packaged default"
        else:
            note = "not installed" + ("" if cfg.sha256 else " (not yet published)")
        out[cfg.model_id] = {
            "installed": installed,
            "verified": verified,
            "quantization": cfg.quantization,
            "size_bytes": cfg.size_bytes,
            "license": cfg.license,
            "path": str(model_dir() / cfg.model_id),
            "note": note,
        }
    return out
