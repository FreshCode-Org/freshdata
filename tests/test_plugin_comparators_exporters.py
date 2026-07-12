"""Comparator and exporter plugin interfaces (freshdata.comparators /
freshdata.exporters)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import freshdata as fd
from freshdata import testing as fdt
from freshdata.enterprise import BlockingRule, ComparisonLevel, EntityResolutionConfig
from freshdata.enterprise.config import _COMPARISON_KINDS
from freshdata.plugins import (
    _RESERVED_COMPARATOR_NAMES,
    clear_plugins,
    get_active_comparator,
    get_active_exporter,
    register_comparator,
    register_exporter,
    registered_plugins,
)

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "plugins"


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_plugins()
    yield
    clear_plugins()


class SuffixComparator:
    """1.0 when the last 3 characters agree (toy example)."""

    name = "suffix3"

    def __call__(self, a: str, b: str) -> float:
        return 1.0 if a[-3:].lower() == b[-3:].lower() else 0.0


class DictExporter:
    name = "summary_dict"

    def export(self, report):
        d = report.to_dict()
        return {"rows_before": d["rows_before"], "rows_after": d["rows_after"]}


# -- comparators --------------------------------------------------------------


def test_register_and_lookup_comparator():
    register_comparator(SuffixComparator())
    comp = get_active_comparator("suffix3")
    assert comp is not None
    assert comp("Johnson", "Jameson") == 1.0
    assert comp("Johnson", "Smith") == 0.0


def test_reserved_names_rejected():
    class Impostor:
        name = "exact"

        def __call__(self, a, b):
            return 1.0

    with pytest.raises(ValueError, match="built-in"):
        register_comparator(Impostor())


def test_reserved_names_match_er_config():
    assert frozenset(_COMPARISON_KINDS) == _RESERVED_COMPARATOR_NAMES


def test_non_callable_comparator_rejected():
    class NotCallable:
        name = "broken"

    with pytest.raises(TypeError, match="callable"):
        register_comparator(NotCallable())


def test_comparator_output_clamped_and_exceptions_skip():
    class Wild:
        name = "wild"

        def __call__(self, a, b):
            if a == "boom":
                raise RuntimeError("bug")
            return 7.5  # out of range

    register_comparator(Wild())
    comp = get_active_comparator("wild")
    assert comp("x", "y") == 1.0  # clamped
    assert comp("boom", "y") is None  # exception -> field skipped


def test_comparison_level_accepts_registered_plugin_kind():
    register_comparator(SuffixComparator())
    level = ComparisonLevel("name", kind="suffix3")
    assert level.kind == "suffix3"


def test_comparison_level_rejects_unknown_kind():
    with pytest.raises(ValueError, match="registered comparator plugin"):
        ComparisonLevel("name", kind="nope_never_registered")


def test_plugin_comparator_end_to_end_in_resolve():
    register_comparator(SuffixComparator())
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["anderson", "henderson"],
            "dob": ["1980-01-01", "1980-01-01"],
        }
    )
    cfg = EntityResolutionConfig(
        enabled=True,
        backend="pandas",
        unique_id_column="id",
        blocking_rules=(BlockingRule("l.dob = r.dob", "same dob"),),
        comparisons=(
            ComparisonLevel("name", kind="suffix3", weight=1.0),
            ComparisonLevel("dob", kind="exact", weight=1.0),
        ),
    )
    _, rep = fd.resolve_entities(df, config=cfg)
    [pair] = rep.pairs
    assert pair.comparison_vector["name"] == 1.0  # "son" == "son"
    assert pair.decision == "match"


def test_example_comparator_passes_contract():
    sys.path.insert(0, str(_EXAMPLES / "custom_comparator"))
    try:
        from initials_comparator import InitialsComparator  # noqa: PLC0415

        fdt.comparator_contract(InitialsComparator())
        comp = InitialsComparator()
        assert comp("J. Smith", "John Smith") == 1.0
    finally:
        sys.path.pop(0)


# -- exporters -----------------------------------------------------------------


def _tiny_report():
    _, report = fd.clean(
        pd.DataFrame({"a": [" x", "y ", " x"]}), return_report=True, verbose=False
    )
    return report


def test_register_and_run_exporter(tmp_path):
    register_exporter(DictExporter())
    report = _tiny_report()
    out = fd.export(report, format="summary_dict")
    assert out == {"rows_before": 3, "rows_after": report.rows_after}
    dest = tmp_path / "out.json"
    fd.export(report, format="summary_dict", path=dest)
    assert json.loads(dest.read_text(encoding="utf-8"))["rows_before"] == 3


def test_export_unknown_format_lists_available():
    register_exporter(DictExporter())
    with pytest.raises(ValueError, match="summary_dict"):
        fd.export(_tiny_report(), format="nope")


def test_exporter_without_export_method_rejected():
    class Broken:
        name = "broken"

    with pytest.raises(TypeError, match="export"):
        register_exporter(Broken())


def test_exporter_bad_return_type_raises():
    class BadReturn:
        name = "bad"

        def export(self, report):
            return 42

    register_exporter(BadReturn())
    with pytest.raises(TypeError, match="expected str or dict"):
        fd.export(_tiny_report(), format="bad")


def test_example_exporter_passes_contract(tmp_path):
    sys.path.insert(0, str(_EXAMPLES / "custom_exporter"))
    try:
        from markdown_exporter import MarkdownExporter  # noqa: PLC0415

        fdt.exporter_contract(MarkdownExporter())
        register_exporter(MarkdownExporter())
        text = fd.export(_tiny_report(), format="markdown", path=tmp_path / "r.md")
        assert text.startswith("# freshdata clean report")
        assert (tmp_path / "r.md").read_text(encoding="utf-8") == text
    finally:
        sys.path.pop(0)


# -- registry introspection ------------------------------------------------------


def test_registered_plugins_lists_new_kinds():
    register_comparator(SuffixComparator())
    register_exporter(DictExporter())
    kinds = {p["kind"] for p in registered_plugins()}
    assert {"comparator", "exporter"} <= kinds
    assert registered_plugins("comparator")[0]["name"] == "suffix3"


def test_clear_plugins_clears_new_buckets():
    register_comparator(SuffixComparator())
    register_exporter(DictExporter())
    clear_plugins()
    assert get_active_comparator("suffix3") is None
    assert get_active_exporter("summary_dict") is None
