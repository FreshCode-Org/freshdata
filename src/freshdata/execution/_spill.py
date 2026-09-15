"""Private, per-run spill directories for out-of-core engines.

DuckDB writes intermediate relation data (rows of the dataset being cleaned) to
its ``temp_directory`` once a query exceeds ``memory_limit``. Those files use
fixed names and the process umask, so a directory shared between users or runs
exposes them to other local accounts and makes concurrent runs collide.

Each run therefore spills into its own ``tempfile.mkdtemp`` directory (mode
0700), created inside a base directory that is checked for ownership and
permissions, and removed when the run's connection closes. The base is:

1. ``EngineConfig.temp_directory`` when set explicitly;
2. otherwise ``$FRESHDATA_SPILL_DIR`` when set;
3. otherwise the per-user cache directory (``$XDG_CACHE_HOME/freshdata/spill``
   or ``~/.cache/freshdata/spill`` on Linux, ``~/Library/Caches/freshdata/spill``
   on macOS, ``%LOCALAPPDATA%\\freshdata\\spill`` on Windows);
4. and, only when that cache directory cannot be created or written,
   ``tempfile.gettempdir()``.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ._config import EngineConfig

log = logging.getLogger("freshdata.execution.spill")

#: Environment variable that overrides the default spill base directory.
SPILL_DIR_ENV = "FRESHDATA_SPILL_DIR"
_RUN_PREFIX = "freshdata_spill_"


class UnsafeSpillDirectoryError(PermissionError):
    """A spill base directory other local users could read from or tamper with."""


def _default_cache_base() -> str | None:
    """The per-user cache spill directory, or ``None`` if it cannot be resolved."""
    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local"
        )
    elif sys.platform == "darwin":
        root = os.path.join(os.path.expanduser("~"), "Library", "Caches")
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        # The XDG spec says a relative value is invalid and must be ignored.
        root = xdg if xdg and os.path.isabs(xdg) else os.path.join(
            os.path.expanduser("~"), ".cache"
        )
    if not os.path.isabs(root):  # no resolvable home directory
        return None
    return os.path.join(root, "freshdata", "spill")


def _ensure_safe_base(path: str) -> str:
    """Create *path* (mode 0700) if needed and check it is safe to spill into.

    On POSIX the directory is accepted when it is owned by the effective user
    and not writable by group or other, or when it has the sticky bit (like
    ``/tmp``) and is owned by the effective user or root. Anything else raises
    :class:`UnsafeSpillDirectoryError` naming the owner and mode.
    """
    os.makedirs(path, mode=0o700, exist_ok=True)
    geteuid = getattr(os, "geteuid", None)
    if os.name != "posix" or geteuid is None:
        return path
    info = os.stat(path)  # follows symlinks: the real target is what gets checked
    if not stat.S_ISDIR(info.st_mode):
        raise NotADirectoryError(f"spill directory {path!r} is not a directory")
    euid = geteuid()
    mode = stat.S_IMODE(info.st_mode)
    private = info.st_uid == euid and not mode & (stat.S_IWGRP | stat.S_IWOTH)
    sticky = bool(mode & stat.S_ISVTX) and info.st_uid in (euid, 0)
    if not (private or sticky):
        raise UnsafeSpillDirectoryError(
            f"refusing to spill into {path!r}: owned by uid {info.st_uid} with mode "
            f"{oct(mode)}; a spill directory must be owned by the current user "
            f"(uid {euid}) and not group/other-writable, or be a sticky directory "
            "such as /tmp. Fix its permissions (chmod 700) or choose another "
            "EngineConfig.temp_directory."
        )
    return path


def _user_spill_base() -> str:
    """Resolve and prepare the default spill base directory (see module docstring)."""
    override = os.environ.get(SPILL_DIR_ENV)
    if override:
        return _ensure_safe_base(os.path.expanduser(override))
    base = _default_cache_base()
    if base is not None:
        try:
            # Keep the freshdata cache parent private too, not just the leaf.
            os.makedirs(os.path.dirname(base), mode=0o700, exist_ok=True)
            base = _ensure_safe_base(base)
            if os.access(base, os.W_OK | os.X_OK):
                return base
        except UnsafeSpillDirectoryError:
            raise
        except OSError as exc:
            log.debug("freshdata spill: cache directory %r unusable (%s)", base, exc)
    # The per-run mkdtemp child below is still 0700 inside the temp directory.
    return _ensure_safe_base(tempfile.gettempdir())


def create_run_spill_dir(engine_config: EngineConfig) -> str:
    """Create and return a private (0700) spill directory for one run.

    The caller removes it with :func:`remove_run_spill_dir` once the engine
    connection using it is closed.
    """
    explicit = engine_config.temp_directory
    if explicit is not None:
        base = _ensure_safe_base(os.path.expanduser(os.fspath(explicit)))
    else:
        base = _user_spill_base()
    return tempfile.mkdtemp(prefix=_RUN_PREFIX, dir=base)


def remove_run_spill_dir(path: str) -> None:
    """Remove a run directory made by :func:`create_run_spill_dir` (best effort)."""
    shutil.rmtree(path, ignore_errors=True)
