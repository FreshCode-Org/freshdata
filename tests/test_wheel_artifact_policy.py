"""No model weights ship inside the wheel (README trust claim + release gate).

A full ``python -m build`` + wheel inspection is a CI-level check (see
``.github/workflows/wheel-size.yml``); this is the fast, always-on unit-test
proxy: the packaging config never lists a weight-file pattern, and no weight
file exists anywhere under the shipped source tree.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
#: Extensions a trained model artifact would use; none may ship in the wheel.
_WEIGHT_EXTENSIONS = (".onnx", ".safetensors", ".bin", ".pt", ".pth", ".h5")


def _wheel_artifacts_patterns() -> list[str]:
    """Glob patterns from ``[tool.hatch.build.targets.wheel] artifacts``.

    Parsed with a small regex rather than a TOML library so this test runs on
    the project's full supported Python range (``>=3.9``; ``tomllib`` needs
    3.11+ and ``tomli`` is not a declared dependency).
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    section = re.search(
        r"\[tool\.hatch\.build\.targets\.wheel\](.*?)(?:\n\[|\Z)", text, re.S
    )
    assert section, "pyproject.toml has no [tool.hatch.build.targets.wheel] section"
    match = re.search(r"artifacts\s*=\s*\[(.*?)\]", section.group(1), re.S)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def test_wheel_artifacts_list_has_no_model_weights():
    for pattern in _wheel_artifacts_patterns():
        assert not pattern.endswith(_WEIGHT_EXTENSIONS), (
            f"wheel artifacts list includes a model-weight pattern: {pattern!r}"
        )


def test_no_model_weight_files_in_shipped_source_tree():
    src = REPO_ROOT / "src" / "freshdata"
    offenders = [
        p for p in src.rglob("*")
        if p.is_file() and p.suffix in _WEIGHT_EXTENSIONS
    ]
    assert offenders == [], f"model weight file(s) found under src/freshdata: {offenders}"
