"""The [semantic] extra is strictly optional: core must never require it."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from freshdata.models import _lazy


def test_core_import_with_semantic_deps_blocked():
    """`import freshdata` and fd.models work with onnxruntime/tokenizers uninstallable."""
    code = (
        "import sys\n"
        "class Blocker:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name in ('onnxruntime', 'tokenizers') else None\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name in ('onnxruntime', 'tokenizers'):\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import freshdata as fd\n"
        "import pandas as pd\n"
        "df = pd.DataFrame({'a': [1, None, 3]})\n"
        "out = fd.clean(df)\n"
        "assert out is not None\n"
        "status = fd.models.status()\n"
        "assert 'fd-col-encoder-v1' in status\n"
        "from freshdata.models.runtime import availability\n"
        "ok, reason = availability()\n"
        "assert ok is False\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_require_helpers_error_messages():
    if _lazy.has_onnxruntime():
        pytest.skip("onnxruntime installed; error-path exercised in subprocess test")
    with pytest.raises(ImportError, match=r"freshdata\[semantic\]"):
        _lazy.require_onnxruntime()


def test_has_semantic_extra_is_bool():
    assert isinstance(_lazy.has_semantic_extra(), bool)


def test_no_module_level_optional_imports():
    """No freshdata module may import onnxruntime/tokenizers at module level."""
    for name in ("freshdata.models.runtime", "freshdata.semantic.cache"):
        module = importlib.import_module(name)
        assert "onnxruntime" not in getattr(module, "__dict__", {})
        assert "tokenizers" not in getattr(module, "__dict__", {})
