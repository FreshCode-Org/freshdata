"""Layer 8: ParseResult display via the Peel 'parse' normalizer (spec §9)."""

from __future__ import annotations

import pandas as pd
import pytest

from freshdata.parsers.base import ParseResult
from freshdata.render.normalize import normalize
from freshdata.render.options import get_display, reset_display
from freshdata.render.plain import render_plain


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for var in ("FRESHDATA_DISPLAY", "FRESHDATA_LEGACY_DISPLAY", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)
    reset_display()
    yield
    reset_display()


def make_parse(**kw) -> ParseResult:
    defaults = {
        "format": "hl7v2",
        "frames": {
            "patient": pd.DataFrame({"id": range(250)}),
            "observation": pd.DataFrame({"v": range(8421)}),
            "encounter": pd.DataFrame({"e": range(611)}),
            "medication_request": pd.DataFrame(),
        },
        "suggested_domain": "healthcare",
        "metadata": {"message_type": "ORU^R01"},
        "warnings": [
            "medication_request: resource type not supported",
            "observation: 12 values failed the range check",
            "segment ZXT-4 not understood (12 occurrences)",
        ],
    }
    defaults.update(kw)
    return ParseResult(**defaults)


class TestParseStatus:
    def test_warnings_or_empty_frame_make_partial(self):
        assert normalize(make_parse()).status == ("PARTIAL",)

    def test_clean_parse_is_clean(self):
        res = make_parse(
            frames={"patient": pd.DataFrame({"id": [1, 2]})},
            warnings=[],
        )
        assert normalize(res).status == ("CLEAN",)

    def test_empty_frame_alone_is_partial(self):
        res = make_parse(frames={"a": pd.DataFrame()}, warnings=[])
        assert normalize(res).status == ("PARTIAL",)


class TestParseContent:
    def test_headline_counts_frames_and_rows(self):
        view = normalize(make_parse())
        assert "4 frame(s)" in view.headline
        assert "9,282 rows total" in view.headline
        assert "3 warning(s)" in view.headline

    def test_stage_ladder_only_reaches_parsed(self):
        view = normalize(make_parse())
        stage = next(m for m in view.metrics if m.label == "stage")
        assert "parsed ✓" in stage.value
        assert "validated —" in stage.value
        assert "cleaned —" in stage.value

    def test_suggested_domain_marked_advisory(self):
        view = normalize(make_parse())
        assert any("advisory" in m.value for m in view.metrics)

    def test_empty_frame_becomes_unsupported_attention(self):
        view = normalize(make_parse())
        med = next(a for a in view.attention if a.subject == "medication_request")
        assert "unsupported items" in med.text
        assert med.severity == "warning"

    def test_general_warning_becomes_attention(self):
        view = normalize(make_parse())
        assert any("ZXT-4" in a.text for a in view.attention)

    def test_frame_warning_not_double_counted(self):
        # "medication_request: ..." is represented by the frame item, not again
        view = normalize(make_parse())
        med_texts = [a.text for a in view.attention if "resource type not supported" in a.text]
        assert med_texts == []  # folded into the unsupported-frame item

    def test_partial_banner_distinguishes_read_from_validated(self):
        view = normalize(make_parse())
        assert "read, not checked" in view.banner

    def test_next_step_points_to_clean_with_domain(self):
        view = normalize(make_parse())
        assert 'domain="healthcare"' in view.next_step
        assert "result.frames[" in view.next_step

    def test_frames_section_lists_rows_and_status(self):
        view = normalize(make_parse())
        frames = next(s for s in view.sections if s.key == "frames")
        rows = {r["frame"]: r for r in frames.rows()}
        assert rows["observation"]["rows"] == 8421
        assert "warning" in rows["observation"]["status"]
        assert rows["patient"]["status"] == "ready"
        assert "unsupported" in rows["medication_request"]["status"]


class TestParseRendering:
    def test_summary_is_plain_text(self):
        text = make_parse().summary()
        assert "freshdata parse" in text
        assert "PARTIAL" in text
        assert "unsupported items" in text  # the empty-frame finding

    def test_str_delegates_to_summary(self):
        assert str(make_parse()) == make_parse().summary()

    def test_to_html_always_uses_peel_no_legacy(self):
        # parse has no legacy renderer, so Peel renders regardless of style
        html = make_parse().to_html()
        assert "freshdata parse" in html
        assert "Needs attention" in html

    def test_to_html_ignores_legacy_env_when_no_legacy_renderer(self, monkeypatch):
        monkeypatch.setenv("FRESHDATA_LEGACY_DISPLAY", "1")
        html = make_parse().to_html()  # must not raise; still Peel
        assert "freshdata parse" in html

    def test_verbose_shows_metadata_section(self):
        out = render_plain(normalize(make_parse()), get_display(mode="verbose"))
        assert "Format metadata" in out
        assert "ORU^R01" in out

    def test_does_not_read_frame_contents(self):
        class Poison(pd.DataFrame):
            @property
            def values(self):  # any content access raises
                raise AssertionError("frame contents must not be read")

        res = make_parse(frames={"patient": Poison({"id": [1, 2, 3]})})
        view = normalize(res)  # only len() is used
        frames = next(s for s in view.sections if s.key == "frames")
        assert frames.rows()[0]["rows"] == 3
