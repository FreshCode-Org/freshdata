"""Currency parsing: the decisions `test_currency_locale.py` does not pin.

Mutation testing over :mod:`freshdata.semantic.experts` found that the currency
parser still passes its suite when several of its locale decisions are inverted.
The existing suite checks the *values* that well-formed amounts parse to; it
does not check the boundaries of the grouping validator, the "repeated
separator can only be grouping" rule, or the ``ambiguous`` flag that decides
whether a cell is auto-repaired or routed to a human. Each group below pins one
of those, and each fails if the corresponding decision is flipped.

What is at stake in each group:

* **Grouping boundaries** -- ``_valid_grouping`` is the only thing standing
  between ``$1.2.3`` and a silent reading of 123. If its first-group length
  check is off by one, ordinary ``$123,456.78`` stops parsing; if its
  reject-branches are inverted, malformed money is coerced to a plausible
  number instead of being left alone. Both directions are pinned here.
* **Digit gate** -- ``_split_amount`` refuses a body with no digits *before*
  handing it to ``float()``. Without that gate ``float()`` happily accepts
  ``"inf"`` and ``"nan"``, so a junk cell would become an infinite monetary
  amount rather than ``None``.
* **The ambiguity flag** -- ``"1,000"`` with no currency to appeal to is either
  one thousand or 1.0 with a decimal comma, and nothing in the string settles
  it. ``_split_amount`` returns ``ambiguous=True`` so the caller can route the
  cell to review. If that flag is flipped to ``False`` the value is silently
  auto-applied at high confidence, which is exactly the thousand-fold error
  FD2-002 was about -- only this time with no signal that a guess was made.
  Nothing else in the suite asserts on the flag's ``True`` case.

These call the private ``_split_amount`` / ``_valid_grouping`` deliberately:
``_split_amount``'s no-currency branch is not reachable through
``parse_currency_parts`` (see the last test), so testing it at the public API
alone cannot pin it.
"""

from __future__ import annotations

import pytest

from freshdata.semantic.experts import (
    _split_amount,
    _valid_grouping,
    parse_currency,
    parse_currency_parts,
)

# -- the ambiguity flag ------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # One thousand under US convention, 1.0 under European convention.
        ("1,000", (1000.0, True)),
        ("1.000", (1.0, True)),
        ("1,200", (1200.0, True)),
        ("1.200", (1.2, True)),
    ],
)
def test_a_grouped_amount_without_a_currency_code_is_flagged_ambiguous(body, expected):
    """No currency and a 3-digit tail: the reading is a guess, and says so.

    The value returned is the dot-as-decimal reading, but the second element
    must stay ``True`` so the caller routes the cell to a human instead of
    auto-applying a reading that is a coin flip.
    """
    assert _split_amount(body, None) == expected


def test_a_currency_code_removes_the_ambiguity_flag():
    """Once a currency is known the reading is decided, not guessed."""
    assert _split_amount("1,000", "USD") == (1000.0, False)
    assert _split_amount("1.000", "USD") == (1.0, False)
    # CHF writes the decimal comma, so its readings are the mirror image.
    assert _split_amount("1,000", "CHF") == (1.0, False)
    assert _split_amount("1.000", "CHF") == (1000.0, False)


def test_a_known_currency_reads_a_three_digit_tail_by_its_own_convention():
    """``$1.000`` is one dollar, not a thousand: USD writes the decimal dot."""
    assert parse_currency_parts("$1.000") == (1.0, False)
    assert parse_currency_parts("$1.250") == (1.25, False)
    assert parse_currency_parts("EUR 1.000") == (1000.0, False)


# -- grouping boundaries -----------------------------------------------------


@pytest.mark.parametrize(
    ("part", "sep"),
    [("123,456", ","), ("999,999,999", ","), ("123.456", ".")],
)
def test_a_three_digit_first_group_is_still_valid_grouping(part, sep):
    """The first group may be *up to* three digits, and three is allowed."""
    assert _valid_grouping(part, sep) is True


def test_an_amount_whose_first_group_is_exactly_three_digits_parses():
    """The commonest shape of all -- rejecting it would break ordinary money."""
    assert parse_currency("$123,456.78") == 123456.78
    assert parse_currency("€123.456,78") == 123456.78


@pytest.mark.parametrize("part", ["1234,567", "12345,678"])
def test_a_first_group_longer_than_three_digits_is_not_grouping(part):
    assert _valid_grouping(part, ",") is False


def test_an_over_long_first_group_is_rejected_rather_than_coerced():
    """``$1234,567.00`` is not 1234567.00 under either convention.

    Accepting it would mean inventing a grouping that the writer did not use,
    so the parser must decline and leave the cell to ordinary dtype repair.
    """
    assert parse_currency("$1234,567.00") is None


def test_a_thousands_group_containing_a_non_digit_is_rejected():
    """``float()`` accepts underscores between digits; grouping must not.

    ``float("1_00.25")`` is 100.25, so without the ``isdigit`` check on every
    group after the first, ``$1,_00.25`` would parse to 100.25 instead of being
    reported unparseable.
    """
    assert _valid_grouping("1,_00", ",") is False
    assert parse_currency("$1,_00.25") is None


# -- a repeated separator can only be grouping -------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$1.234.567", 1234567.0),
        ("$1.234.567,89", 1234567.89),
        ("€1,234,567", 1234567.0),
    ],
)
def test_a_separator_used_more_than_once_is_read_as_grouping(text, expected):
    """A decimal separator appears at most once, so twice means grouping.

    This holds even when the separator is the one the currency normally uses as
    a decimal point: ``$1.234.567`` is a million, not a malformed 1.234.
    """
    assert parse_currency(text) == expected


# -- the digit gate before float() -------------------------------------------


@pytest.mark.parametrize("body", ["", "inf", "nan", "-inf", "infinity", ".", ",", "-"])
def test_a_body_with_no_digits_is_rejected_before_float_sees_it(body):
    """``float("inf")`` succeeds; an amount column must not inherit that.

    Every one of these must come back ``(None, False)`` -- no value, and no
    claim that an ambiguity was resolved.
    """
    assert _split_amount(body, "USD") == (None, False)
    assert _split_amount(body, None) == (None, False)


def test_a_marker_with_no_number_reports_no_value_and_no_ambiguity():
    assert parse_currency_parts("$") == (None, False)
    assert parse_currency_parts("EUR") == (None, False)


def test_malformed_grouping_reports_no_value_and_no_ambiguity():
    """Rejection is not ambiguity: there is no reading to route for review."""
    assert parse_currency_parts("$1.2.3") == (None, False)
    assert parse_currency_parts("$1,20.50") == (None, False)
    assert parse_currency_parts("$1234,567.00") == (None, False)


def test_a_string_without_a_currency_marker_reports_no_ambiguity():
    """Not currency at all, so there is nothing for a human to adjudicate."""
    assert parse_currency_parts("1,200") == (None, False)
    assert parse_currency_parts("1.200") == (None, False)


# -- documented, not endorsed ------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("CAD 1.200", 1.2), ("AUD 1.200", 1.2), ("CNY 1.200", 1.2), ("$1.200", 1.2)],
)
def test_an_ambiguous_amount_reaching_the_public_parser_is_never_flagged(text, expected):
    """Current behaviour: ``ambiguous=True`` cannot surface through the API.

    ``parse_currency_parts`` requires a currency marker, and every marker it
    accepts is one ``detect_currency`` also resolves, so ``_split_amount`` is
    never called with ``code=None`` from here. The "report it ambiguous rather
    than guess" branch is therefore unreachable in production, and a currency
    outside ``_COMMA_DECIMAL_CURRENCIES`` is guessed as dot-decimal instead --
    contrary to what the module comment on that table says. Pinned as the
    behaviour that exists today; see the note filed with this suite.
    """
    assert parse_currency_parts(text) == (expected, False)
