from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess

import numpy as np
import pandas as pd

import freshdata

from .models import EnvironmentInfo


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment() -> EnvironmentInfo:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
        ).stdout
    )
    try:
        import psutil  # noqa: PLC0415

        physical = psutil.cpu_count(logical=False)
        total_ram = int(psutil.virtual_memory().total)
    except ImportError:
        physical = None
        total_ram = None
    return EnvironmentInfo(
        git_commit=commit,
        git_dirty=dirty,
        python_version=platform.python_version(),
        pandas_version=pd.__version__,
        numpy_version=np.__version__,
        freshdata_version=freshdata.__version__,
        optional_versions={
            name: _version(name) for name in ("polars", "duckdb", "pyspark", "pyarrow")
        },
        platform=platform.platform(),
        processor=platform.processor(),
        cpu_count_logical=os.cpu_count() or 1,
        cpu_count_physical=physical,
        total_ram_bytes=total_ram,
    )
