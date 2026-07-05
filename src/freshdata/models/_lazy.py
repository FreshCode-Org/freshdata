"""Lazy optional-dependency import guards for the local model runtime.

The embedding backend depends on optional packages (``onnxruntime``,
``tokenizers``) shipped behind the ``[semantic]`` extra. Importing
:mod:`freshdata` must never require them, so the runtime resolves its
dependencies through these helpers at call time and raises a clear,
install-pointing error if one is missing.
"""

from __future__ import annotations

from typing import Any

_INSTALL_HINT = "Install it with: pip install 'freshdata-cleaner[semantic]'"


def require_onnxruntime() -> Any:
    """Return the imported :mod:`onnxruntime` module or raise a helpful error."""
    try:
        import onnxruntime
    except ImportError as exc:  # pragma: no cover - exercised via message
        raise ImportError(
            f"The local semantic model runtime requires onnxruntime. {_INSTALL_HINT}"
        ) from exc
    return onnxruntime


def has_onnxruntime() -> bool:
    """Return True when :mod:`onnxruntime` is importable."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def require_tokenizers() -> Any:
    """Return the imported :mod:`tokenizers` module or raise a helpful error."""
    try:
        import tokenizers
    except ImportError as exc:  # pragma: no cover - exercised via message
        raise ImportError(
            f"The local semantic model runtime requires tokenizers. {_INSTALL_HINT}"
        ) from exc
    return tokenizers


def has_tokenizers() -> bool:
    """Return True when :mod:`tokenizers` is importable."""
    try:
        import tokenizers  # noqa: F401
    except ImportError:
        return False
    return True


def has_semantic_extra() -> bool:
    """Return True when every ``[semantic]`` extra dependency is importable."""
    return has_onnxruntime() and has_tokenizers()
