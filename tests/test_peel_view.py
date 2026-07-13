"""Layer 1: the Peel view model (freshdata.render.view)."""

from __future__ import annotations

import pytest

from freshdata.render.view import (
    DOMAINS,
    SEVERITIES,
    STATUSES,
    AttentionItem,
    Metric,
    PeelView,
    Section,
    confidence_phrase,
    rank_attention,
)


class TestVocabulary:
    def test_status_labels(self):
        assert STATUSES == (
            "CLEAN",
            "CHANGED",
            "REVIEW",
            "BLOCKED",
            "PARTIAL",
            "SKIPPED",
            "FAILED",
        )

    def test_severities_most_urgent_first(self):
        assert SEVERITIES == ("error", "warning", "review", "info")

    def test_domains_privacy_first_cosmetic_last(self):
        assert DOMAINS[0] == "privacy"
        assert DOMAINS[-1] == "cosmetic"


class TestConfidencePhrase:
    @pytest.mark.parametrize(
        ("value", "phrase"),
        [
            (1.0, "strong evidence"),
            (0.95, "strong evidence"),
            (0.94, "moderate evidence"),
            (0.80, "moderate evidence"),
            (0.79, "uncertain"),
            (0.60, "uncertain"),
            (0.59, "uncertain"),
            (0.10, "uncertain"),
        ],
    )
    def test_ladder(self, value, phrase):
        assert confidence_phrase(value) == phrase

    def test_ambiguous_overrides_score(self):
        assert confidence_phrase(0.91, ambiguous=True) == "ambiguous — no change made"


class TestAttentionRanking:
    def test_domain_outranks_severity(self):
        privacy_info = AttentionItem("A1", "info", "email", "pii", domain="privacy")
        cosmetic_error = AttentionItem("A2", "error", "name", "case", domain="cosmetic")
        assert rank_attention([cosmetic_error, privacy_info])[0] is privacy_info

    def test_severity_then_count_then_name(self):
        a = AttentionItem("1", "warning", "b", "x", count=5)
        b = AttentionItem("2", "warning", "a", "x", count=5)
        c = AttentionItem("3", "warning", "z", "x", count=99)
        d = AttentionItem("4", "error", "z", "x", count=0)
        assert [i.id for i in rank_attention([a, b, c, d])] == ["4", "3", "2", "1"]

    def test_deterministic(self):
        items = [
            AttentionItem(str(i), "info", f"col{i % 3}", "t", count=i % 2) for i in range(10)
        ]
        assert rank_attention(list(items)) == rank_attention(list(reversed(items)))

    def test_unknown_domain_and_severity_sort_last(self):
        odd = AttentionItem("1", "bizarre", "a", "x", domain="martian")
        known = AttentionItem("2", "info", "a", "x", domain="cosmetic")
        assert rank_attention([odd, known])[-1] is odd


class TestPeelView:
    def _view(self, **kw):
        defaults = {
            "kind": "clean_report",
            "status": ("CHANGED", "REVIEW"),
            "headline": "9,988 of 10,000 rows kept",
            "metrics": (Metric("missing", "0", before="1,204", after="0"),),
            "attention": (),
            "next_step": None,
            "sections": (),
        }
        defaults.update(kw)
        return PeelView(**defaults)

    def test_status_label_joined(self):
        assert self._view().status_label == "CHANGED · REVIEW"

    def test_rejects_unknown_status(self):
        with pytest.raises(ValueError, match="unknown status"):
            self._view(status=("CHANGED", "AMAZING"))

    def test_frozen(self):
        with pytest.raises(AttributeError):
            self._view().kind = "other"

    def test_section_body_is_lazy(self):
        calls = []

        def body():
            calls.append(1)
            return [{"a": 1}]

        s = Section("actions", "All actions", body, count=1)
        assert calls == []
        assert s.rows() == [{"a": 1}]
        assert calls == [1]

    def test_audit_ref_is_identity(self):
        report = object()
        assert self._view(audit_ref=report).audit_ref is report
