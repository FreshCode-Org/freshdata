"""Optional local model registry and runtime (``fd.models``).

Everything importable here is stdlib+numpy only; the ONNX runtime dependencies
live behind the ``[semantic]`` extra and are imported lazily at encode time.
Models are never downloaded implicitly — see :func:`pull`.

    import freshdata as fd

    fd.models.status()                     # what is installed / verified
    fd.models.pull("fd-col-encoder-v1")    # explicit download (network)
    fd.models.path("fd-col-encoder-v1")    # local artifact path
    fd.models.list_available()             # registry metadata
"""

from .download import pull
from .registry import list_available, model_dir, path, status
from .types import (
    ModelChecksumError,
    ModelConfig,
    ModelError,
    ModelNotInstalledError,
    ModelNotPublishedError,
    ModelVersionError,
    UnknownModelError,
)

__all__ = [
    "ModelChecksumError",
    "ModelConfig",
    "ModelError",
    "ModelNotInstalledError",
    "ModelNotPublishedError",
    "ModelVersionError",
    "UnknownModelError",
    "list_available",
    "model_dir",
    "path",
    "pull",
    "status",
]
