"""Packaging and typing marker smoke tests."""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.9/3.10 (tomli ships with pytest there)
    import tomli as tomllib

REPO = Path(__file__).resolve().parents[1]

# The import package is `freshdata`, but the PyPI distribution must stay
# `freshdata-cleaner`: PyPI rejects `freshdata` as too similar to the existing
# `fresh-data` project (the v0.x upload failure that forced the rename).
DIST_NAME = "freshdata-cleaner"


def _project_metadata() -> dict:
    return tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_py_typed_marker_exists():
    marker = REPO / "src" / "freshdata" / "py.typed"
    assert marker.is_file()
    assert marker.read_text() == "" or marker.stat().st_size == 0


def test_distribution_name_is_the_publishable_pypi_name():
    assert _project_metadata()["name"] == DIST_NAME


def test_readme_and_docs_install_commands_match_distribution_name():
    for doc in ("README.md", "docs/installation.md", "docs/index.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        bad = [
            line.strip()
            for line in text.splitlines()
            if re.search(r"pip install ['\"]?freshdata(?!-cleaner)\b", line)
        ]
        assert not bad, f"{doc} installs the unpublishable name: {bad}"


def test_source_install_hints_match_distribution_name():
    offenders = []
    for path in (REPO / "src" / "freshdata").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if re.search(r"freshdata(?!-cleaner)\[", text):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"install hints use the unpublishable name: {offenders}"
