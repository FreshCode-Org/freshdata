"""An unrecognised field type must not silently lose protection (FD2-004).

``config_for_field`` matched the declared ``semantic_type`` exactly -- case
sensitive and untrimmed -- and fell through to the caller's base config on no
match. The failure mode ran backwards for a safety-oriented library: a
recognised structural type was protected from lossy transformation, while an
unrecognised one received it in full. So ``"Ticker"`` was cleaned more
aggressively than ``"ticker"``, and a column declared ``"password"`` was
case-folded and stripped of punctuation.

Every lossy option is opt-in, so a default ``fd.clean`` was never affected.
"""

from __future__ import annotations

import warnings

import pytest

from freshdata.textclean import TextCleanConfig, clean_text_value, config_for_field

LOSSY = TextCleanConfig(case="lower", remove_punctuation=True)
SECRET = "Sekr3t-P@ss!!!"


@pytest.mark.parametrize("declared", ["ticker", "Ticker", "TICKER", " ticker ", "\tticker\n"])
def test_case_and_whitespace_variants_resolve_to_the_same_protection(declared):
    """'Ticker' names the same field type as 'ticker'."""
    assert config_for_field(declared, LOSSY).case is None
    assert clean_text_value(SECRET, config_for_field(declared, LOSSY)).cleaned == SECRET


@pytest.mark.parametrize("declared", ["email", "EMAIL", "Person_Name", "IDENTIFIER", "Free_Text"])
def test_other_types_normalise_too(declared):
    """The normalisation is not special-cased to one type."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config_for_field(declared, LOSSY)  # must not warn: all are known


def test_a_structural_type_still_refuses_lossy_options():
    assert clean_text_value(SECRET, config_for_field("identifier", LOSSY)).cleaned == SECRET


def test_free_text_still_accepts_them():
    """The protection must not become blanket: free text really is free text."""
    assert clean_text_value(SECRET, config_for_field("free_text", LOSSY)).cleaned == "sekr3tpss"


@pytest.mark.parametrize("declared", ["password", "api_key", "secret", "identifer", "e-mail"])
def test_an_unknown_type_warns_when_a_lossy_option_is_active(declared):
    """Silence let a misspelling quietly downgrade a column.

    fieldcheck already warns for an unknown semantic_type; this is the same
    courtesy on the cleaning side.
    """
    with pytest.warns(UserWarning, match="unknown semantic_type"):
        config_for_field(declared, LOSSY)


def test_the_warning_names_the_offending_options_and_the_known_types():
    with pytest.warns(UserWarning) as caught:
        config_for_field("password", LOSSY)
    message = str(caught[0].message)
    assert "password" in message
    assert "case='lower'" in message and "remove_punctuation" in message
    assert "ticker" in message  # the known-type list is included


def test_no_warning_on_the_default_path():
    """A default config has no lossy option, so an unknown type is harmless."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config_for_field("password", TextCleanConfig())
        config_for_field(None, TextCleanConfig())


def test_none_is_not_reported_as_an_unknown_type():
    """No declaration is not a misspelling; it must stay quiet."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert config_for_field(None, LOSSY).case == "lower"


def test_a_non_string_type_does_not_crash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert config_for_field(42, LOSSY).case == "lower"  # type: ignore[arg-type]
