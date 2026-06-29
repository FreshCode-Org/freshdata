"""Optional-dependency probing for the rendering layer.

Keeps imports of viz/notebook extras lazy and turns a missing package into a
clear, actionable install hint instead of a bare ``ModuleNotFoundError``.
"""

from __future__ import annotations

import importlib
from types import ModuleType

#: Which extra ships each optional renderer dependency.
_EXTRA_FOR = {
    "itables": "viz",
    "plotly": "viz",
    "great_tables": "viz",
    "anywidget": "notebook",
    "IPython": "notebook",
}


def has(package: str) -> bool:
    """Return ``True`` if *package* can be imported, without importing it eagerly."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


def require(package: str, *, feature: str | None = None) -> ModuleType:
    """Import *package* or raise a helpful ``ImportError`` naming the extra.

    Parameters
    ----------
    package:
        Import name, e.g. ``"plotly"``.
    feature:
        Optional human name of the feature needing it, for the message.
    """
    try:
        return importlib.import_module(package)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        extra = _EXTRA_FOR.get(package, "viz")
        what = f"{feature} needs " if feature else ""
        raise ImportError(
            f"{what}the optional package {package!r}. "
            f"Install it with: pip install 'freshdata[{extra}]'"
        ) from exc
