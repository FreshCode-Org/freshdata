"""Shared fixtures for training-pipeline tests.

These tests exercise ``training/`` (a dev-only package outside the runtime
import path) and are excluded from the freshdata coverage gate the same way
``tests/benchmark`` is — run with ``pytest tests/training -q --no-cov``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "benchmarks"
if str(BENCH_DIR) not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(BENCH_DIR))

pytest.importorskip("training")
