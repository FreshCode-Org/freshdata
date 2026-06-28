"""Make the ``benchmarks/`` modules importable from the benchmark test suite.

The harness lives outside the installed ``freshdata`` package (it is a tool, not
library code), so its directory is put on ``sys.path`` here rather than being
imported as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
