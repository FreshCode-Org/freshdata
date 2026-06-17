import numpy as np
import pandas as pd

import freshdata as fd


def test_whitespace_stripped_object_and_string_dtype():
    df = pd.DataFrame(
        {
            "obj": [" x ", "y\t", "z"],
            "str": pd.array([" a", "b ", "c"], dtype="string"),
        }
    )
    out = fd.clean(df)
    assert out["obj"].tolist() == ["x", "y", "z"]
    assert out["str"].tolist() == ["a", "b", "c"]


def test_internal_whitespace_preserved():
    df = pd.DataFrame({"city": ["New  York ", " San Francisco"]})
    out = fd.clean(df)
    assert out["city"].tolist() == ["New  York", "San Francisco"]


def test_mixed_type_column_numbers_survive():
    df = pd.DataFrame({"mix": [1, " keep ", 2.5, None]})
    out = fd.clean(df, fix_dtypes=False, drop_empty_rows=False)
    assert out["mix"].tolist()[:3] == [1, "keep", 2.5]


def test_sentinels_are_case_insensitive():
    df = pd.DataFrame({"v": ["NULL", "n/a", "None", "ok", "-", "#REF!"]})
    out = fd.clean(df, drop_empty_rows=False, drop_duplicates=False)
    assert out["v"].isna().sum() == 5
    assert out["v"].dropna().tolist() == ["ok"]


def test_extra_sentinels():
    df = pd.DataFrame({"v": ["unknown", "ok", "UNKNOWN "]})
    out = fd.clean(df, extra_sentinels=("unknown",), drop_empty_rows=False,
                   drop_duplicates=False)
    assert out["v"].isna().sum() == 2


def test_sentinel_only_when_entire_cell_matches():
    df = pd.DataFrame({"v": ["banana", "nathan", "na"]})
    out = fd.clean(df, drop_empty_rows=False)
    assert out["v"].isna().sum() == 1  # only the bare "na"


def test_steps_can_be_disabled():
    df = pd.DataFrame({"v": [" x ", "N/A"]})
    out = fd.clean(df, strip_whitespace=False, normalize_sentinels=False)
    assert out["v"].tolist() == [" x ", "N/A"]


def test_empty_string_becomes_missing():
    df = pd.DataFrame({"v": ["", "  ", "x"]})
    out = fd.clean(df, drop_empty_rows=False, drop_duplicates=False)
    assert out["v"].isna().sum() == 2


def test_unhashable_values_pass_through():
    df = pd.DataFrame({"v": [[1, 2], [3], None], "w": ["a", "b", "c"]})
    out = fd.clean(df)
    assert out["v"].iloc[0] == [1, 2]
    assert not np.any(out["w"].isna())


def test_blank_string_cleanup_visible_in_report():
    """Blank-string normalisation is reflected in the CleanReport.

    Regression test for Issue #17: FreshData promises that every cleaning
    action is explainable.  This test verifies that:

    1. Empty strings (``""``) and whitespace-only strings (``" "``) as well as
       common sentinel values like ``"N/A"`` are converted to proper missing
       values (``pd.NA`` / ``float('nan')``).
    2. A ``normalize_sentinels`` action is recorded in the :class:`CleanReport`
       with a count that reflects all three affected cells.

    ``drop_empty_rows=False`` and ``drop_duplicates=False`` are passed so that
    the resulting DataFrame keeps all rows and the count comparison is
    straightforward.
    """
    df = pd.DataFrame({"v": ["", " ", "N/A", "real_value"]})

    out, report = fd.clean(
        df,
        return_report=True,
        drop_empty_rows=False,
        drop_duplicates=False,
    )

    # --- data assertions ---
    # The three sentinel / blank cells must have become NaN
    assert out["v"].isna().sum() == 3, (
        f"Expected 3 missing values after cleaning blanks and N/A, "
        f"got {out['v'].isna().sum()!r}; cleaned column: {out['v'].tolist()!r}"
    )
    # The real value must be untouched
    assert out["v"].dropna().tolist() == ["real_value"], (
        f"Expected ['real_value'] to survive, got {out['v'].dropna().tolist()!r}"
    )

    # --- report assertions ---
    sentinel_actions = [a for a in report if a.step == "normalize_sentinels"]
    assert sentinel_actions, (
        "CleanReport contains no 'normalize_sentinels' action; "
        "blank-string cleanup is not being recorded in the audit trail."
    )
    total_normalised = sum(a.count for a in sentinel_actions)
    assert total_normalised >= 3, (
        f"Expected at least 3 cells recorded under 'normalize_sentinels', "
        f"but the combined count is {total_normalised}."
    )
