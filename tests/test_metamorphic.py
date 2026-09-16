"""Metamorphic properties of ``fd.clean``: what must not change, and what must.

The suite had no invariance tests on the public API. Worse,
``tests/test_execution/test_action_parity.py::_normalize`` sorts rows *and*
columns before comparing engines, so an order-sensitivity bug in ``fd.clean``
would pass that gate silently. These tests assert the relations directly, on
unsorted output.

Two kinds of property:

**Invariances** -- reordering rows or columns, adding an unrelated column, or
renaming a column whose meaning is pinned by ``semantic_context`` must not
change any surviving cell. A cleaner whose decisions depend on positional
accident fails here.

**Required changes** -- replacing a valid value with an invalid one *must*
change the result. Without these, a cleaner that does nothing at all would
satisfy every invariance above and look perfect.
"""

from __future__ import annotations

import pandas as pd
import pytest

import freshdata as fd

SEEDS = (0, 1, 7, 11)


def _frame() -> pd.DataFrame:
    """A frame that exercises several steps at once.

    ``row_key`` anchors identity so a permuted frame can be realigned, and
    keeps every row non-empty so ``drop_empty_rows`` never removes one and
    confuses "the cell was nulled" with "the row went away".
    """
    return pd.DataFrame(
        {
            "row_key": [f"r{i}" for i in range(10)],
            "customer_id": ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010"],
            "amount": [10.5, 20.0, "  30.5 ", 40.0, 50.0, "N/A", 70.0, 80.0, 90.0, 100.0],
            "country": ["US", "GB", "FR", "DE", "JP", "CA", "AU", "BR", "IN", "US"],
            "note": list("abcdefghij"),
        }
    )


def _clean(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Clean and return a plain DataFrame.

    ``fd.clean`` returns a ``CleanResult`` subclass, but its ``_constructor``
    is ``pd.DataFrame`` by design, so any derived frame (``.reindex``, a slice)
    is a plain DataFrame and does not carry a now-stale ``.report()``. Comparing
    a realigned frame against a raw result would therefore fail on frame *type*
    while every value matched, so both sides are normalised here.
    """
    return pd.DataFrame(fd.clean(df, verbose=False, **kw))


# -- invariances ------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_row_order_does_not_change_any_cell(seed):
    """Shuffling the input must not change what happens to a row."""
    df = _frame()
    reference = _clean(df)
    shuffled = _clean(df.sample(frac=1, random_state=seed)).reindex(reference.index)
    pd.testing.assert_frame_equal(shuffled, reference)


@pytest.mark.parametrize("seed", SEEDS)
def test_row_order_does_not_change_imputed_values(seed):
    """Imputation reads the whole column, so its result must not track order."""
    df = pd.DataFrame(
        {
            "row_key": [f"r{i}" for i in range(10)],
            "a": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
        }
    )
    reference = _clean(df, impute="mean")
    shuffled = _clean(df.sample(frac=1, random_state=seed), impute="mean")
    pd.testing.assert_frame_equal(shuffled.reindex(reference.index), reference)


def test_column_order_does_not_change_any_cell():
    df = _frame()
    reference = _clean(df)
    reversed_out = _clean(df[list(reversed(df.columns))])
    for column in reference.columns:
        pd.testing.assert_series_equal(reversed_out[column], reference[column])


def test_an_unrelated_column_does_not_change_the_others():
    """Adding a column must not perturb decisions about existing ones."""
    df = _frame()
    reference = _clean(df)
    widened = df.copy()
    widened["unrelated"] = list(range(10))
    out = _clean(widened)
    for column in reference.columns:
        pd.testing.assert_series_equal(out[column], reference[column])


@pytest.mark.parametrize("name", ["customer_id", "code", "value", "x"])
def test_a_declared_identifier_survives_whatever_the_column_is_called(name):
    """Renaming must not change values once the meaning is declared.

    Column names *are* documented semantic evidence, so this property is only
    claimed when ``semantic_context`` pins the type explicitly.
    """
    df = pd.DataFrame({name: ["001", "002", "003", "004", "005"], "amt": [1.0] * 5})
    out = _clean(df, semantic_context={"columns": {name: {"semantic_type": "identifier"}}})
    assert out[name].tolist() == ["001", "002", "003", "004", "005"]


def test_cleaning_twice_is_a_no_op_on_the_default_path():
    """Idempotency stated as a metamorphic relation over the public API."""
    once = _clean(_frame())
    twice = _clean(once)
    pd.testing.assert_frame_equal(twice, once)


# -- required changes -------------------------------------------------------
#
# Without these, a cleaner that returned its input unchanged would satisfy
# every invariance above.


def test_replacing_a_valid_amount_with_text_changes_the_result():
    good = _frame()
    bad = good.copy()
    bad.loc[8, "amount"] = "apple"
    assert not _clean(good)["amount"].equals(_clean(bad)["amount"])


def test_replacing_a_value_with_a_sentinel_changes_the_result():
    good = _frame()
    bad = good.copy()
    bad.loc[3, "country"] = "N/A"
    assert not _clean(good)["country"].equals(_clean(bad)["country"])


def test_an_outlier_is_reported_even_though_the_value_is_untouched():
    """``outliers="flag"`` must flag and must not mutate.

    Both halves matter: silence would be a missed detection, and a changed
    value would be an unrequested repair.
    """
    df = pd.DataFrame(
        {
            "row_key": [f"r{i}" for i in range(10)],
            "v": [10.0, 11.0, 12.0, 10.5, 11.5, 10.2, 11.1, 10.8, 11.9, 9999.0],
        }
    )
    out, report = fd.clean(df, verbose=False, outliers="flag", return_report=True)
    assert out["v"].iloc[9] == 9999.0
    assert [a.count for a in report.actions if a.step == "outliers"] == [1]


# -- order-sensitive engine parity ------------------------------------------


@pytest.mark.parametrize("engine", ["polars", "duckdb"])
def test_engines_agree_without_sorting_the_answer_first(engine):
    """Close the gap left by test_action_parity, which sorts before comparing.

    That normalisation makes row- and column-order divergence between engines
    invisible. Here the frames are compared as returned.
    """
    pytest.importorskip(engine)
    df = _frame()
    reference = _clean(df, strategy="conservative", fix_dtypes=False)
    native = _clean(df, strategy="conservative", fix_dtypes=False, engine=engine)
    assert list(native.columns) == list(reference.columns)
    assert native["row_key"].tolist() == reference["row_key"].tolist()
