"""Display configuration for Peel renderers (spec §11).

Display options never affect cleaning behavior — they are consumed only by
renderers. Precedence: explicit arguments > process-wide :func:`set_display`
> environment (``FRESHDATA_DISPLAY``, ``NO_COLOR``, ``FRESHDATA_NO_PREVIEWS``)
> defaults.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

#: Valid display modes (spec §11.1).
MODES = ("auto", "compact", "standard", "verbose", "debug", "json", "plain", "silent")


@dataclass(frozen=True)
class RenderOptions:
    """Resolved display options handed to a renderer."""

    mode: str = "auto"
    color: str = "auto"  # "auto" | "always" | "never"
    width: int = 74
    previews: bool = True  # False → schema-only display, values never shown
    ascii_icons: bool = False

    def resolved_mode(self, *, isatty: bool | None = None) -> str:
        """Collapse ``auto`` to a concrete mode for the current environment."""
        if self.mode != "auto":
            return self.mode
        if isatty is None:
            isatty = sys.stdout.isatty()
        return "standard" if isatty else "compact"


_state = {"options": RenderOptions()}


def set_display(**changes: object) -> RenderOptions:
    """Set process-wide display preferences, e.g. ``fd.set_display(mode="compact")``.

    Returns the new options. Unknown fields raise ``TypeError``; an unknown
    mode raises ``ValueError`` so typos fail at configuration time, not render
    time.
    """
    new = replace(_state["options"], **changes)  # type: ignore[arg-type]
    if new.mode not in MODES:
        raise ValueError(f"unknown display mode {new.mode!r}; expected one of {MODES}")
    _state["options"] = new
    return new


def get_display(**overrides: object) -> RenderOptions:
    """The effective options: defaults ← env ← :func:`set_display` ← *overrides*."""
    opts = _state["options"]
    env_mode = os.environ.get("FRESHDATA_DISPLAY")
    if env_mode and opts.mode == "auto" and env_mode in MODES:
        opts = replace(opts, mode=env_mode)
    if os.environ.get("NO_COLOR"):
        opts = replace(opts, color="never")
    if os.environ.get("FRESHDATA_NO_PREVIEWS"):
        opts = replace(opts, previews=False)
    if overrides:
        opts = replace(opts, **overrides)  # type: ignore[arg-type]
        if opts.mode not in MODES:
            raise ValueError(f"unknown display mode {opts.mode!r}; expected one of {MODES}")
    if opts.mode == "plain":
        opts = replace(opts, color="never", ascii_icons=True)
    return opts


def reset_display() -> None:
    """Restore defaults (used by tests and ``FRESHDATA_LEGACY_DISPLAY`` flows)."""
    _state["options"] = RenderOptions()
