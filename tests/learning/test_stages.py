"""Stage tests for the learn pipeline: align, diff, classify.

Each stage is exercised directly (not through ``fd.learn``) so a failure
points at the stage that broke, with fixtures small enough to eyeball.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from freshdata.learning.align import align_pair
from freshdata.learning.classify import classify_diffs
from freshdata.learning.diff import compute_diff
from freshdata.learning.types import TRANSFORM_FAMILIES


def _pair(messy_values: list, clean_values: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(messy_values)
    ids = [f"r{i}" for i in range(n)]
    messy = pd.DataFrame({"id": ids, "col": messy_values})
    clean = pd.DataFrame({"id": ids, "col": clean_values})
    return messy, clean


def _families(messy_values: list, clean_values: list, *, use_frame: bool = False) -> dict:
    """Map raw_value -> classified family for a tiny single-column pair."""
    messy, clean = _pair(messy_values, clean_values)
    aligned = align_pair(messy, clean, key="id")
    kwargs = {"clean_frame": aligned.clean_aligned} if use_frame else {}
    classified = classify_diffs(compute_diff(aligned), **kwargs)
    return {c.diff.raw_value: c.family for lst in classified.values() for c in lst}


# ---------------------------------------------------------------------------
# align
# ---------------------------------------------------------------------------


class TestAlign:
    def test_key_alignment_reorders_rows(self) -> None:
        messy, clean = _pair(["a", "b"], ["a", "B"])
        shuffled = clean.iloc[::-1].reset_index(drop=True)
        pair = align_pair(messy, shuffled, key="id")
        assert pair.alignment_report.mode == "key"
        assert pair.alignment_report.matched_rows == 2
        assert pair.alignment_report.row_level
        assert pair.clean_aligned["col"].tolist() == ["a", "B"]

    def test_multi_column_key(self) -> None:
        messy = pd.DataFrame({"k1": ["a", "a"], "k2": [1, 2], "col": ["x", "y"]})
        clean = pd.DataFrame({"k1": ["a", "a"], "k2": [2, 1], "col": ["y2", "x2"]})
        pair = align_pair(messy, clean, key=["k1", "k2"])
        assert pair.alignment_report.matched_rows == 2
        assert tuple(pair.alignment_report.key) == ("k1", "k2")
        assert pair.clean_aligned["col"].tolist() == ["x2", "y2"]

    def test_missing_key_column_raises(self) -> None:
        messy, clean = _pair(["a"], ["a"])
        with pytest.raises(KeyError, match="key column"):
            align_pair(messy, clean.drop(columns=["id"]), key="id")

    def test_duplicate_keys_warn_and_keep_first(self) -> None:
        messy = pd.DataFrame({"id": ["r0", "r0", "r1"], "col": ["x", "y", "z"]})
        clean = pd.DataFrame({"id": ["r0", "r1"], "col": ["x", "z"]})
        report = align_pair(messy, clean, key="id").alignment_report
        assert report.duplicate_messy_keys == 1
        assert any("duplicate" in w for w in report.warnings)
        assert report.matched_rows == 2

    def test_unmatched_rows_counted_not_stored(self) -> None:
        messy = pd.DataFrame({"id": ["r0", "r1", "r2"], "col": ["a", "b", "c"]})
        clean = pd.DataFrame({"id": ["r0"], "col": ["a"]})
        pair = align_pair(messy, clean, key="id")
        report = pair.alignment_report
        assert report.unmatched_messy == 2
        assert report.matched_rows == 1
        assert len(pair.messy_aligned) == 1  # unmatched rows never ride along

    def test_positional_fallback_equal_length(self) -> None:
        messy = pd.DataFrame({"col": ["a", "b"], "n": [1, 2]})
        clean = pd.DataFrame({"col": ["a", "B"], "n": [1, 2]})
        report = align_pair(messy, clean, key=None).alignment_report
        assert report.mode == "positional"
        assert report.row_level
        assert report.matched_rows == 2

    def test_unequal_length_without_key_degrades_to_column_level(self) -> None:
        messy = pd.DataFrame({"col": ["a", "b"]})
        clean = pd.DataFrame({"col": ["a", "B", "c"]})
        report = align_pair(messy, clean, key=None).alignment_report
        assert report.mode == "column_only"
        assert not report.row_level
        assert any("degrading to column-level" in w for w in report.warnings)

    def test_key_beats_positional_for_unequal_lengths(self) -> None:
        messy = pd.DataFrame({"id": ["r0", "r1"], "col": ["a", "b"]})
        clean = pd.DataFrame({"id": ["r1", "r0", "r9"], "col": ["b2", "a2", "zz"]})
        pair = align_pair(messy, clean, key="id")
        assert pair.alignment_report.row_level
        assert pair.clean_aligned["col"].tolist() == ["a2", "b2"]


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_distinct_pairs_with_support(self) -> None:
        messy, clean = _pair(["N/A", "N/A", "x", "y"], [None, None, "x", "y"])
        summary = compute_diff(align_pair(messy, clean, key="id"))
        diffs = summary.column_diffs["col"]
        assert len(diffs) == 1
        assert diffs[0].raw_value == "N/A"
        assert diffs[0].support == 2

    def test_identical_frames_produce_no_diffs(self) -> None:
        messy, clean = _pair(["x", "y"], ["x", "y"])
        summary = compute_diff(align_pair(messy, clean, key="id"))
        assert summary.column_diffs == {}

    def test_shared_nans_are_not_diffs(self) -> None:
        messy, clean = _pair([np.nan, "x"], [np.nan, "x"])
        summary = compute_diff(align_pair(messy, clean, key="id"))
        assert summary.column_diffs == {}

    def test_schema_adds_and_removals(self) -> None:
        messy = pd.DataFrame({"id": ["r0"], "old": ["x"], "shared": [1]})
        clean = pd.DataFrame({"id": ["r0"], "new": ["y"], "shared": [1]})
        summary = compute_diff(align_pair(messy, clean, key="id"))
        assert "old" in summary.schema_diffs.removed_columns
        assert "new" in summary.schema_diffs.added_columns
        assert "shared" in summary.schema_diffs.shared_columns

    def test_dtype_change_recorded(self) -> None:
        messy, clean = _pair(["42", "17"], [42, 17])
        summary = compute_diff(align_pair(messy, clean, key="id"))
        assert "col" in summary.schema_diffs.dtype_changes

    def test_no_row_payloads_stored(self) -> None:
        # A ValueDiff carries only the cell pair + counts — never row data.
        messy, clean = _pair(["N/A", "x"], [None, "x"])
        summary = compute_diff(align_pair(messy, clean, key="id"))
        diff = summary.column_diffs["col"][0]
        payload = diff.to_dict()
        assert set(payload) <= {"column", "raw_value", "clean_value", "support", "kind"}


# ---------------------------------------------------------------------------
# classify — one case per observable transform family
# ---------------------------------------------------------------------------


class TestClassifyFamilies:
    def test_family_names_are_the_documented_vocabulary(self) -> None:
        assert "allowed_value_map" in TRANSFORM_FAMILIES
        assert "unexplained" in TRANSFORM_FAMILIES
        assert len(TRANSFORM_FAMILIES) == 19

    def test_case_fold(self) -> None:
        fams = _families(["YES-VAL", "ok"], ["yes-val", "ok"])
        assert fams["YES-VAL"] == "case_fold"

    def test_whitespace_and_case(self) -> None:
        fams = _families([" Padded ", "ok"], ["padded", "ok"])
        assert fams[" Padded "] in ("whitespace", "case_fold")

    def test_email_normalize(self) -> None:
        fams = _families(["a@@gmail.com", "b@x.com"], ["a@gmail.com", "b@x.com"])
        assert fams["a@@gmail.com"] == "email_normalize"

    def test_phone_normalize(self) -> None:
        fams = _families(["98765 43210", "+919000000001"], ["+919876543210", "+919000000001"])
        assert fams["98765 43210"] == "phone_normalize"

    def test_reference_typo_with_clean_domain(self) -> None:
        fams = _families(
            ["activve", "pending", "active"],
            ["active", "pending", "active"],
            use_frame=True,
        )
        assert fams["activve"] in ("reference_normalize", "allowed_value_map")

    def test_dayfirst_inference(self) -> None:
        fams = _families(
            ["05/02/2024", "07/03/2024"],
            [pd.Timestamp(2024, 2, 5), pd.Timestamp(2024, 3, 7)],
        )
        assert set(fams.values()) == {"date_dayfirst_inference"}

    def test_unambiguous_dates_are_not_diffs(self) -> None:
        # "02 Jan 2024" parses identically with or without dayfirst, so the
        # diff stage already treats it as equal — only a dtype change remains.
        messy, clean = _pair(
            ["02 Jan 2024", "04 Mar 2024"],
            [pd.Timestamp(2024, 1, 2), pd.Timestamp(2024, 3, 4)],
        )
        summary = compute_diff(align_pair(messy, clean, key="id"))
        assert summary.column_diffs == {}
        assert "col" in summary.schema_diffs.dtype_changes

    def test_currency_parse(self) -> None:
        fams = _families(["₹1,000", "₹2,500"], [1000.0, 2500.0])
        assert set(fams.values()) == {"currency_parse"}

    def test_unit_strip(self) -> None:
        fams = _families(["10 kg", "12 kg"], [10.0, 12.0])
        assert set(fams.values()) == {"unit_strip"}

    def test_spelled_number(self) -> None:
        fams = _families(["five", "seven"], [5, 7])
        assert set(fams.values()) == {"spelled_number"}

    def test_boolean_synonym(self) -> None:
        fams = _families(
            ["yes", "no", "yes", "no"],
            pd.array([True, False, True, False], dtype="object"),
        )
        assert set(fams.values()) == {"boolean_synonym"}

    def test_sentinel_to_missing(self) -> None:
        fams = _families(["MISSING_VAL", "3.5", "2.0"], [np.nan, 3.5, 2.0])
        assert fams["MISSING_VAL"] == "sentinel_to_missing"

    def test_allowed_value_map_for_domain_vocabulary(self) -> None:
        fams = _families(
            ["H.R.", "Fin", "Eng"],
            ["human_resources", "finance", "engineering"],
            use_frame=True,
        )
        assert set(fams.values()) == {"allowed_value_map"}

    def test_numeric_rounding(self) -> None:
        fams = _families([3.14159, 2.71828], [3.14, 2.72])
        assert set(fams.values()) == {"numeric_rounding"}

    def test_dtype_coercion(self) -> None:
        fams = _families(["42", "17"], [42, 17])
        assert set(fams.values()) == {"dtype_coercion"}

    def test_missing_imputation_is_never_a_literal_map(self) -> None:
        fams = _families([np.nan, 5.0, 3.0], [4.0, 5.0, 3.0])
        assert set(fams.values()) == {"missing_imputation"}

    def test_unexplained_fallback(self) -> None:
        fams = _families(["zzqqk", "ok"], ["completely-other", "ok"])
        assert fams["zzqqk"] == "unexplained"
