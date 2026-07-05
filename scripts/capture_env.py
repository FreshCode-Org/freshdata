"""Capture environment details for reproducible benchmarking.

Outputs a JSON file with Python version, OS, CPU, RAM, and installed
package versions (freshdata, pandas, numpy, pyarrow).

Usage::

    python scripts/capture_env.py [output_path]
    # Defaults to bench/artifacts/env_info.json

"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_package_version(name: str) -> str:
    """Return installed version of a package, or 'not installed'."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not installed"


def get_cpu_info() -> dict:
    """Collect CPU information."""
    info = {
        "model": platform.processor() or "unknown",
        "count_logical": os.cpu_count() or 0,
    }
    try:
        import psutil

        info["count_physical"] = psutil.cpu_count(logical=False) or 0
        freq = psutil.cpu_freq()
        if freq:
            info["freq_mhz"] = freq.current
    except ImportError:
        pass
    return info


def get_ram_gb() -> float:
    """Return total system RAM in GB."""
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 2)
    except ImportError:
        return 0.0


def capture_environment() -> dict:
    """Build a complete environment snapshot."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": get_cpu_info(),
        "ram_gb": get_ram_gb(),
        "packages": {
            "freshdata-cleaner": get_package_version("freshdata-cleaner"),
            "pandas": get_package_version("pandas"),
            "numpy": get_package_version("numpy"),
            "pyarrow": get_package_version("pyarrow"),
            "asv": get_package_version("asv"),
            "psutil": get_package_version("psutil"),
        },
    }


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "bench/artifacts/env_info.json"
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    env = capture_environment()
    output.write_text(json.dumps(env, indent=2) + "\n")
    print(f"Environment captured → {output}")
    print(json.dumps(env, indent=2))


if __name__ == "__main__":
    main()
