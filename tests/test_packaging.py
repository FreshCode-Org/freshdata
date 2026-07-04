"""Packaging and typing marker smoke tests."""

from pathlib import Path

import tomllib


def test_py_typed_marker_exists():
    marker = Path(__file__).resolve().parents[1] / "src" / "freshdata" / "py.typed"
    assert marker.is_file()
    assert marker.read_text() == "" or marker.stat().st_size == 0


def test_distribution_name_matches_install_command():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    metadata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "freshdata"
