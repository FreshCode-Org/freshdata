"""Packaging and typing marker smoke tests."""

from pathlib import Path


def test_py_typed_marker_exists():
    marker = Path(__file__).resolve().parents[1] / "src" / "freshdata" / "py.typed"
    assert marker.is_file()
    assert marker.read_text() == "" or marker.stat().st_size == 0
