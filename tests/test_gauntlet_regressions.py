"""Regression tests for defects surfaced by the Validation Gauntlet.

Each test class documents one defect found by ``benchmarks/gauntlet`` and
pins the corrected behaviour. See docs/validation-gauntlet.md.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import freshdata as fd
from freshdata import FieldSpec
from freshdata.textclean import config_for_field


def _numeric_with_stragglers(n: int = 200) -> pd.DataFrame:
    """A mostly-numeric text price column plus a control column."""
    rng = np.random.default_rng(0)
    price = [f"{v:.2f}" for v in rng.uniform(10, 500, n)]
    price[7] = "apple"          # unparseable text — information, not noise
    price[90] = "$1,200.50"     # unparseable by pd.to_numeric
    control = rng.uniform(0, 1, n).astype(object)
    control[5] = None           # one genuinely missing value
    return pd.DataFrame({"price": price, "control": control})


class TestCoercedCellsAreNotImputed:
    """Defect: fix_dtypes coerced unparseable text to NaN and the auto engine
    then imputed the median — 'apple' in a price column silently became a
    fabricated number with no per-cell trace."""

    def test_unparseable_values_stay_missing_for_review(self):
        df = _numeric_with_stragglers()
        out, report = fd.clean(df, return_report=True)
        assert str(out["price"].dtype) in ("float64", "Float64")
        assert pd.isna(out.loc[7, "price"]), "coerced junk must not be imputed"

    def test_true_missing_values_are_still_imputed(self):
        df = _numeric_with_stragglers()
        out, _ = fd.clean(df, return_report=True)
        assert not pd.isna(out.loc[5, "control"]), (
            "genuine missing values keep the documented auto-impute behaviour")

    def test_report_preserves_original_values_per_cell(self):
        df = _numeric_with_stragglers()
        _, report = fd.clean(df, return_report=True)
        cells = report.coerced_cells.get("price")
        assert cells, "report.coerced_cells must record the quarantined cells"
        assert cells[7] == "apple"
        assert list(cells) == [7], "the parseable '$1,200.50' is repaired, not quarantined"

    def test_review_action_is_recorded(self):
        df = _numeric_with_stragglers()
        _, report = fd.clean(df, return_report=True)
        review = [a for a in report.actions
                  if a.column == "price" and "unparseable" in a.description
                  and a.human_review]
        assert review, "quarantine decision must appear in the audit trail"
        assert review[0].rationale
        assert review[0].count == 1

    def test_sentinels_are_not_quarantined(self):
        # 'N/A' is a documented null marker: it is *missing*, not junk, and the
        # imputation contract for it is unchanged.
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"v": [f"{x:.1f}" for x in rng.uniform(1, 9, 100)],
                           "pad": range(100)})
        df.loc[3, "v"] = "N/A"
        out, report = fd.clean(df, return_report=True)
        assert not pd.isna(out.loc[3, "v"])
        assert "v" not in report.coerced_cells

    def test_invalid_dates_stay_nat(self):
        rng = np.random.default_rng(2)
        dates = [f"2025-01-{d:02d}" for d in rng.integers(1, 28, 100)]
        dates[11] = "2023-02-30"
        df = pd.DataFrame({"d": dates, "pad": range(100)})
        out, report = fd.clean(df, return_report=True)
        assert str(out["d"].dtype).startswith("datetime64")
        assert pd.isna(out.loc[11, "d"])
        assert report.coerced_cells["d"][11] == "2023-02-30"

    def test_coerced_cells_serialize(self):
        df = _numeric_with_stragglers()
        _, report = fd.clean(df, return_report=True)
        as_dict = report.to_dict()
        assert as_dict["coerced_cells"]["price"]["7"] == "apple"
        assert json.dumps(as_dict)  # row labels / values must be JSON-safe

    def test_explicit_impute_still_fills_everything(self):
        df = _numeric_with_stragglers()
        out = fd.clean(df, impute="median")
        assert not out["price"].isna().any(), (
            "an explicit impute= request overrides the quarantine default")


class TestFormattedNumberRescue:
    """Defect: the '$1,234.56' rescue in _try_numeric only ran when the plain
    parse failed the threshold — formatted stragglers in a mostly-plain
    numeric column were coerced to missing instead of parsed."""

    def test_formatted_stragglers_are_parsed_not_quarantined(self):
        df = _numeric_with_stragglers()
        out, report = fd.clean(df, return_report=True)
        assert out.loc[90, "price"] == pytest.approx(1200.50)
        assert 90 not in report.coerced_cells.get("price", {})

    def test_unparseable_text_is_still_quarantined(self):
        df = _numeric_with_stragglers()
        out, report = fd.clean(df, return_report=True)
        assert pd.isna(out.loc[7, "price"])
        assert report.coerced_cells["price"][7] == "apple"


class TestValidateFieldsHandoff:
    """Defect: fd.clean's contamination warning points users at
    fd.validate_fields, but the consensus gate needed an 80% share while the
    warning fires from 60% — the documented handoff found nothing."""

    def test_readme_handoff_frame_reports_the_bad_cell(self):
        df = pd.DataFrame({
            "company": ["Apple", "Microsoft", "apple", "Tesla"],
            "ticker": ["AAPL", "MSFT", "AAPL", "TSLA"],
            "price": ["189.5", "402.1", "apple", "212.0"],
        })
        report = fd.validate_fields(df)
        bad = [i for i in report.issues if i.column == "price"]
        assert len(bad) == 1
        assert bad[0].row == 2
        assert bad[0].classification == "semantic_mismatch"

    def test_large_minority_still_blocks_consensus(self):
        # 60/40 mixed content is a legitimately mixed column, not contamination.
        df = pd.DataFrame({"x": ["1", "2", "3", "a", "b", "4", "c", "5", "d", "6"]})
        report = fd.validate_fields(df)
        assert not [i for i in report.issues if i.column == "x"]


class TestAllowedValuesBeatNullMarkers:
    """Defect: a value explicitly present in allowed_values ('NA' = Namibia)
    was swallowed by the generic null-marker heuristic before the vocabulary
    was ever consulted."""

    def test_na_in_vocabulary_is_a_value_not_a_null(self):
        spec = FieldSpec(allowed_values=frozenset({"US", "DE", "NA"}),
                         required=True, nullable=False)
        df = pd.DataFrame({"country": ["US", "NA", "DE", "US"]})
        report = fd.validate_fields(df, schema={"country": spec})
        assert not report.issues, (
            "'NA' is explicitly allowed here and must not be treated as missing")

    def test_na_outside_vocabulary_is_still_a_null_marker(self):
        spec = FieldSpec(allowed_values=frozenset({"United States", "Germany"}),
                         required=True, nullable=False)
        df = pd.DataFrame({"country": ["United States", "NA", "Germany", "Germany"]})
        report = fd.validate_fields(df, schema={"country": spec})
        assert [i for i in report.issues
                if i.row == 1 and i.classification == "schema_violation"]


class TestCaseVariantSuggestions:
    """Gap: 'ACTIVE' against allowed {'active'} was silently accepted by the
    casefold comparison; the canonical form should be suggested."""

    def test_case_variant_gets_warning_and_suggestion(self):
        spec = FieldSpec(allowed_values=frozenset({"active", "churned"}))
        df = pd.DataFrame({"status": ["active", "ACTIVE", "churned"]})
        report = fd.validate_fields(df, schema={"status": spec})
        issues = [i for i in report.issues if i.row == 1]
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].suggestion == "active"
        assert issues[0].action == "accept_with_warning"

    def test_exact_matches_stay_silent(self):
        spec = FieldSpec(allowed_values=frozenset({"active", "churned"}))
        df = pd.DataFrame({"status": ["active", "churned", "active"]})
        report = fd.validate_fields(df, schema={"status": spec})
        assert not report.issues


class TestDateBounds:
    """Gap: date fields had no range checks, so a future date of birth or an
    1875 admission date validated cleanly."""

    def test_future_date_flagged(self):
        spec = FieldSpec(semantic_type="date", max_value="2026-07-12")
        df = pd.DataFrame({"dob": ["1980-02-01", "2031-05-01"]})
        report = fd.validate_fields(df, schema={"dob": spec})
        issues = [i for i in report.issues if i.row == 1]
        assert len(issues) == 1
        assert issues[0].classification == "domain_mismatch"

    def test_min_bound(self):
        spec = FieldSpec(semantic_type="date", min_value="1900-01-01")
        df = pd.DataFrame({"admitted": ["1875-01-01", "2020-06-01"]})
        report = fd.validate_fields(df, schema={"admitted": spec})
        assert [i for i in report.issues if i.row == 0]

    def test_in_range_dates_pass(self):
        spec = FieldSpec(semantic_type="date",
                         min_value="1900-01-01", max_value="2026-12-31")
        df = pd.DataFrame({"d": ["1980-02-01", "2020-06-01"]})
        report = fd.validate_fields(df, schema={"d": spec})
        assert not report.issues


class TestContentTypesKeepTypography:
    """Defect: normalize_punctuation rewrote em-dashes, curly quotes and prime
    marks in free-text and entity-name fields — content, not noise."""

    @pytest.mark.parametrize("ftype", ["free_text", "entity_name", "company_name"])
    def test_field_config_withholds_punctuation_mapping(self, ftype):
        assert config_for_field(ftype).normalize_punctuation is False

    def test_free_text_column_keeps_typography(self):
        df = pd.DataFrame({"notes": ["très bien — merci", "curly “quotes”",
                                     'Café Press — 12″ (limited)']})
        out, _ = fd.clean_text(df, field_types={"notes": "free_text"})
        assert out["notes"].tolist() == df["notes"].tolist()

    def test_untyped_columns_keep_normalizing(self):
        df = pd.DataFrame({"c": ["a — b"]})
        out, _ = fd.clean_text(df)
        assert out["c"].tolist() == ["a - b"]
