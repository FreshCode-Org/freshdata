"""A declared vocabulary outranks a generic null marker in fd.clean (FD2-001).

``fieldcheck`` has applied this rule since the ``TestAllowedValuesBeatNullMarkers``
regression: "'NA' may be Namibia: when the schema literally allows a value, it
is a value, not a missing marker" (``fieldcheck.py:466``). ``fd.clean`` did not.
``normalize_sentinels`` applied ``DEFAULT_SENTINELS`` unconditionally, so a
caller who had explicitly declared ``NA`` as permitted still lost it, and the
only escapes were protecting the column outright (which disables every other
repair) or turning sentinel handling off globally for every column.

The gauntlet fixture states the intended behaviour directly: "without a
vocabulary containing 'NA', the null-marker reading wins; with allowed_values
that includes 'NA' the value survives".
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd

ISO = ["US", "GB", "FR", "DE", "JP", "CA", "AU", "BR", "NA"]


def _frame():
    return pd.DataFrame({"cust": [f"c{i}" for i in range(9)], "country": list(ISO)})


def test_declared_allowed_values_keep_a_sentinel_looking_value():
    """'NA' is Namibia when the caller says the column allows it."""
    out = fd.clean(
        _frame(),
        verbose=False,
        semantic_context={"columns": {"country": {"allowed_values": ISO}}},
    )
    assert out["country"].iloc[8] == "NA"


def test_a_compiled_context_policy_reaches_the_sentinel_step():
    """The documented natural-language route must work as well as the dict."""
    out = fd.clean(
        _frame(),
        verbose=False,
        context="Allowed country values are US, GB, FR, DE, JP, CA, AU, BR, NA.",
    )
    assert out["country"].iloc[8] == "NA"


def test_without_a_declaration_the_null_marker_reading_still_wins():
    """Unchanged default. This is the gauntlet 'sentinel_collision' gold label."""
    out = fd.clean(_frame(), verbose=False)
    assert pd.isna(out["country"].iloc[8])


def test_a_vocabulary_that_excludes_na_still_nulls_it():
    """Mirrors fieldcheck's test_na_outside_vocabulary_is_still_a_null_marker."""
    out = fd.clean(
        _frame(),
        verbose=False,
        semantic_context={"columns": {"country": {"allowed_values": ["US", "GB", "FR"]}}},
    )
    assert pd.isna(out["country"].iloc[8])


def test_the_exemption_is_scoped_to_the_declaring_column():
    """Declaring NA for one column must not rescue it everywhere."""
    df = pd.DataFrame(
        {
            "country": ["US", "NA", "DE", "FR", "GB", "JP", "CA", "AU", "BR"],
            "note": ["a", "NA", "c", "d", "e", "f", "g", "h", "i"],
        }
    )
    out = fd.clean(
        df,
        verbose=False,
        semantic_context={"columns": {"country": {"allowed_values": ISO}}},
    )
    assert out["country"].iloc[1] == "NA"
    assert pd.isna(out["note"].iloc[1])


@pytest.mark.parametrize("declared", ["na", "Na", " NA "])
def test_vocabulary_matching_is_casefolded_and_trimmed(declared):
    """Consistent with fieldcheck and with extra_sentinels normalisation."""
    out = fd.clean(
        _frame(),
        verbose=False,
        semantic_context={"columns": {"country": {"allowed_values": ["US", declared]}}},
    )
    assert out["country"].iloc[8] == "NA"


@pytest.mark.parametrize("allowed", [None, [], "NA", 42, {"nested": "dict"}])
def test_a_malformed_vocabulary_is_ignored_not_fatal(allowed):
    """A bad hint must not crash cleaning, and must not grant an exemption.

    A bare string is rejected deliberately: iterating it would treat 'N' and
    'A' as separate permitted values.
    """
    out = fd.clean(
        _frame(),
        verbose=False,
        semantic_context={"columns": {"country": {"allowed_values": allowed}}},
    )
    assert pd.isna(out["country"].iloc[8])


def test_the_none_brand_case_behaves_the_same_way():
    """'None' is a real brand; the same declaration rescues it."""
    df = pd.DataFrame(
        {
            "k": [f"r{i}" for i in range(9)],
            "brand": [
                "Acme",
                "Globex",
                "Initech",
                "Stark",
                "Wayne",
                "Hooli",
                "Umbrella",
                "Vandelay",
                "None",
            ],
        }
    )
    plain = fd.clean(df, verbose=False)
    assert pd.isna(plain["brand"].iloc[8])

    declared = fd.clean(
        df,
        verbose=False,
        semantic_context={"columns": {"brand": {"allowed_values": [*df["brand"].tolist()]}}},
    )
    assert declared["brand"].iloc[8] == "None"
