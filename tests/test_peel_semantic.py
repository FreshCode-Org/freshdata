"""Layer 7: semantic evidence surfaced in the Peel view (spec §7)."""

from __future__ import annotations

import pytest

from freshdata.render.normalize import normalize_clean_report
from freshdata.render.notebook import render_notebook
from freshdata.render.options import get_display, reset_display
from freshdata.render.plain import render_plain
from freshdata.report import CleanReport


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for var in ("FRESHDATA_DISPLAY", "NO_COLOR", "FRESHDATA_NO_PREVIEWS"):
        monkeypatch.delenv(var, raising=False)
    reset_display()
    yield
    reset_display()


def base_report() -> CleanReport:
    return CleanReport(
        rows_before=1_000, rows_after=1_000, cols_before=4, cols_after=4, duration_seconds=0.1
    )


def add_semantic(rep, column, raw, proposed, *, status, confidence, evidence,
                 backend="pattern", memory=False, calibrated=None, rationale=""):
    meta = {
        "raw_value": raw,
        "proposed_value": proposed,
        "issue_type": "canonicalize",
        "backend": backend,
        "evidence": [{"kind": k, "detail": d, "weight": w} for k, d, w in evidence],
    }
    if memory:
        meta["memory_key"] = f"{column}:{raw}"
    if calibrated is not None:
        meta["calibrated_confidence"] = calibrated
        meta["calibration_version"] = "v1"
    rep.add(
        "semantic",
        f"proposed {proposed!r} for {raw!r}",
        column=column,
        count=0 if status == "skipped" else 1,
        status=status,
        confidence=confidence,
        model_id=f"semantic:canonicalize:{column}",
        memory_influenced=memory,
        rationale=rationale,
        metadata=meta,
    )


class TestConfidenceAndDecision:
    def test_review_item_shows_change_and_confidence_phrase(self):
        rep = base_report()
        add_semantic(
            rep, "country", "Germny", "Germany",
            status="suggested", confidence=0.84,
            evidence=[("value_share", "matches known country list", 0.62)],
        )
        view = normalize_clean_report(rep)
        (item,) = view.attention
        assert item.id == "S1"
        assert "'Germny' → 'Germany'" in item.text
        assert "moderate evidence (~0.84)" in item.text  # ~ = uncalibrated

    def test_ambiguous_item_says_no_change_made(self):
        rep = base_report()
        add_semantic(
            rep, "status", "actv", "active",
            status="skipped", confidence=0.51,
            evidence=[("value_share", "'active' vs 'archived' too close", 0.51)],
        )
        (item,) = normalize_clean_report(rep).attention
        assert "no change made" in item.text
        assert "ambiguous — no change made" in item.text

    def test_uncalibrated_confidence_is_marked(self):
        rep = base_report()
        add_semantic(
            rep, "city", "Sao Palo", "São Paulo",
            status="suggested", confidence=0.81,
            evidence=[("pattern", "diacritic normalization", 0.5)],
        )
        (item,) = normalize_clean_report(rep).attention
        assert "(~0.81)" in item.text  # ~ means confidence not independently calibrated

    def test_calibrated_confidence_has_no_tilde(self):
        rep = base_report()
        add_semantic(
            rep, "city", "Sao Palo", "São Paulo",
            status="suggested", confidence=0.81, calibrated=0.88,
            evidence=[("pattern", "diacritic", 0.5)],
        )
        (item,) = normalize_clean_report(rep).attention
        assert "(0.88)" in item.text
        assert "~" not in item.text


class TestSemanticSection:
    def test_section_appears_with_counts(self):
        rep = base_report()
        add_semantic(rep, "a", "x", "y", status="automatic", confidence=0.97,
                     evidence=[("pattern", "rule", 0.9)])
        add_semantic(rep, "b", "p", "q", status="suggested", confidence=0.8,
                     evidence=[("pattern", "rule", 0.5)])
        add_semantic(rep, "c", "m", "n", status="skipped", confidence=0.5,
                     evidence=[("pattern", "tie", 0.4)])
        view = normalize_clean_report(rep)
        section = next(s for s in view.sections if s.key == "semantic")
        assert section.count == 3
        assert "1 applied" in section.title
        assert "1 review" in section.title
        assert "1 no change" in section.title

    def test_no_section_when_no_semantic_actions(self):
        rep = base_report()
        rep.add("missing", "filled", column="a", count=3)
        view = normalize_clean_report(rep)
        assert all(s.key != "semantic" for s in view.sections)

    def test_rows_ordered_ambiguous_then_review_then_applied(self):
        rep = base_report()
        add_semantic(rep, "applied_col", "x", "y", status="automatic", confidence=0.97,
                     evidence=[("pattern", "rule", 0.9)])
        add_semantic(rep, "ambig_col", "m", "n", status="skipped", confidence=0.5,
                     evidence=[("pattern", "tie", 0.4)])
        add_semantic(rep, "review_col", "p", "q", status="suggested", confidence=0.8,
                     evidence=[("pattern", "rule", 0.5)])
        section = next(s for s in normalize_clean_report(rep).sections if s.key == "semantic")
        decisions = [r["decision"] for r in section.rows()]
        assert decisions == ["ambiguous", "review", "applied"]

    def test_evidence_summary_is_signed_and_sorted(self):
        rep = base_report()
        add_semantic(
            rep, "country", "Germny", "Germany", status="suggested", confidence=0.84,
            evidence=[
                ("value_share", "96% of values are countries", 0.20),
                ("edit_distance", "matches known country list", 0.62),
            ],
        )
        section = next(s for s in normalize_clean_report(rep).sections if s.key == "semantic")
        (row,) = section.rows()
        # strongest signal first, signed
        assert row["evidence"].startswith("+0.62 matches known country list")
        assert "+0.20 96% of values are countries" in row["evidence"]

    def test_provenance_source_distinguishes_memory(self):
        rep = base_report()
        add_semantic(rep, "a", "x", "y", status="suggested", confidence=0.8, memory=True,
                     evidence=[("memory", "seen before", 0.4)])
        section = next(s for s in normalize_clean_report(rep).sections if s.key == "semantic")
        assert section.rows()[0]["source"] == "cleaning memory"


class TestCoverageNote:
    def test_note_names_sources_that_ran(self):
        rep = base_report()
        add_semantic(rep, "a", "x", "y", status="automatic", confidence=0.97,
                     evidence=[("pattern", "rule", 0.9)])
        section = next(s for s in normalize_clean_report(rep).sections if s.key == "semantic")
        assert "checked with rules" in section.title

    def test_note_flags_skipped_model(self):
        rep = base_report()
        add_semantic(rep, "a", "x", "y", status="suggested", confidence=0.8,
                     evidence=[("pattern", "rule", 0.5)])
        rep.fallback_events = [
            {"backend": "embedding-model", "fallback_step": "semantic_embed",
             "fallback_reason": "model 'fd-embed-v1' not installed"}
        ]
        section = next(s for s in normalize_clean_report(rep).sections if s.key == "semantic")
        assert "continued without the optional model" in section.title


class TestRendering:
    def test_verbose_plain_shows_semantic_section(self):
        rep = base_report()
        add_semantic(rep, "country", "Germny", "Germany", status="suggested", confidence=0.84,
                     evidence=[("edit", "matches known country list", 0.62)])
        out = render_plain(normalize_clean_report(rep), get_display(mode="verbose"))
        assert "Semantic proposals" in out
        assert "Germany" in out
        assert "+0.62" in out

    def test_notebook_shows_semantic_section(self):
        rep = base_report()
        add_semantic(rep, "country", "Germny", "Germany", status="suggested", confidence=0.84,
                     evidence=[("edit", "matches known country list", 0.62)])
        html = render_notebook(normalize_clean_report(rep))
        assert "Semantic proposals" in html
        assert "moderate evidence" in html
