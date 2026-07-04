"""Phase-5 backward compatibility: runtime stays model-free, training stays isolated.

Complements ``test_semantic_backcompat.py`` (defaults/config/report/wheel)
with the guarantees specific to the Phase-5 training pipeline: the runtime
must not import ``training``, must not call any teacher/LLM client, and the
core public-API defaults introduced across Phases 1-4 remain unchanged now
that Phase 5 exists alongside them.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_clean_default_unchanged(messy):
    out = fd.clean(messy)
    assert isinstance(out, pd.DataFrame)


def test_clean_context_none_unchanged(messy):
    a = fd.clean(messy, verbose=False)
    b = fd.clean(messy, context=None, verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_clean_profile_none_unchanged(messy):
    a = fd.clean(messy, verbose=False)
    b = fd.clean(messy, profile=None, verbose=False)
    pd.testing.assert_frame_equal(a, b)


def test_default_semantic_backends_unchanged():
    from freshdata.config import CleanConfig

    assert CleanConfig().semantic_backends == ("deterministic",)


def test_runtime_package_does_not_import_training():
    """No module under src/freshdata may import the training/ package."""
    src_root = REPO_ROOT / "src" / "freshdata"
    offenders = []
    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            imports_training = any(
                name and (name == "training" or name.startswith("training.")) for name in names)
            if imports_training:
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"runtime files import training/: {offenders}"


def test_training_package_is_not_importable_via_freshdata_namespace():
    assert not hasattr(fd, "training")


def test_no_teacher_client_reachable_from_clean(messy):
    """Cleaning never touches network I/O: no urllib call originates from fd.clean."""
    from unittest import mock

    with mock.patch("urllib.request.urlopen") as mocked:
        fd.clean(messy, verbose=False)
        mocked.assert_not_called()


def test_fresh_freshdata_import_never_pulls_in_training():
    """A clean subprocess ``import freshdata`` must never import training/."""
    import subprocess

    probe = (
        "import freshdata, sys; "
        "print('training' in sys.modules "
        "or any(m.startswith('training.') for m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=True,
    )
    assert result.stdout.strip() == "False", result.stdout


@pytest.mark.parametrize("path", [
    ".github/workflows/ci.yml",
    ".github/workflows/nightly-real-model.yml",
    ".github/workflows/cleanbench-full.yml",
    ".github/workflows/perf-regression.yml",
    ".github/workflows/wheel-size.yml",
])
def test_ci_workflow_files_exist(path):
    assert (REPO_ROOT / path).is_file()


def test_pr_ci_test_job_excludes_online_and_large_markers():
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'pytest -m "not online and not large"' in text


def test_wheel_size_workflow_checks_onnx_and_training_paths():
    text = (REPO_ROOT / ".github" / "workflows" / "wheel-size.yml").read_text(encoding="utf-8")
    assert ".onnx" in text
    assert "training/" in text
