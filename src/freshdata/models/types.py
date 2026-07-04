"""Typed metadata and errors for the local model registry.

Note: the architecture spec asks for ``slots=True`` dataclasses, but the
package still supports Python 3.9 where ``dataclass(slots=...)`` does not
exist, so the repo convention (see :mod:`freshdata.repairplan`) of frozen
dataclasses without slots is followed instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ModelError(Exception):
    """Base class for local model registry/runtime errors."""


class ModelNotInstalledError(ModelError):
    """A model referenced by id has no files in the local model directory."""


class ModelChecksumError(ModelError):
    """A model file's sha256 does not match the registry pin."""


class ModelNotPublishedError(ModelError):
    """The model has no published download location yet."""


class ModelVersionError(ModelError):
    """The installed library is older than the model's ``min_lib_version``."""


class UnknownModelError(ModelError):
    """The model id is not present in the registry."""


@dataclass(frozen=True)
class ModelConfig:
    """Registry metadata for one downloadable (or packaged) model artifact.

    ``sha256`` is ``None`` while the artifact is unpublished and no hash has
    been pinned yet; installed-but-unpinned models load with a "unverified"
    note instead of a checksum guarantee. ``url`` is relative to the download
    base (``FRESHDATA_MODEL_URL_BASE``); an empty base means the model is not
    yet published and :func:`freshdata.models.pull` explains the manual
    placement path instead of downloading.
    """

    model_id: str
    sha256: str | None
    size_bytes: int
    license: str
    url: str
    min_lib_version: str
    quantization: str  # "int8" | "fp32" | "json"
    files: tuple[str, ...] = field(default_factory=tuple)
    packaged_default: bool = False
