"""Tier-0 parser: every intent family, threshold parsing, unparsed handling."""

import pytest

from freshdata.context import parse_context, split_sentences
from freshdata.context.lexicon import parse_confidence


def _single(text):
    result = parse_context(text)
    assert len(result.candidates) == 1, (text, result)
    assert not result.unparsed
    return result.candidates[0]


# -- sentence splitting -------------------------------------------------------


def test_split_sentences_newlines_bullets_and_punctuation():
    text = """
    - CustomerID is unique.
    2) Emails must be valid!  Never modify revenue.
    """
    assert split_sentences(text) == (
        "CustomerID is unique",
        "Emails must be valid",
        "Never modify revenue",
    )


def test_split_never_breaks_decimals_or_percentages():
    (sentence,) = split_sentences("Missing Age should be estimated only if confidence above 0.95")
    assert "0.95" in sentence


# -- one test per intent family ----------------------------------------------


def test_domain():
    c = _single("This is an ecommerce customer dataset.")
    assert c.intent == "domain"
    assert c.params == {"name": "ecommerce_customer"}


def test_unique():
    c = _single("CustomerID is unique.")
    assert c.intent == "unique"
    assert c.column_refs == ("CustomerID",)


def test_valid_format():
    c = _single("Emails must be valid.")
    assert c.intent == "valid_format"
    assert c.column_refs == ("Emails",)
    assert c.params == {"format": "email"}


def test_valid_format_explicit_word():
    c = _single("contact_url must be a valid url.")
    assert c.intent == "valid_format"
    assert c.params == {"format": "url"}


def test_locale_format():
    c = _single("Phone numbers are Indian.")
    assert c.intent == "locale_format"
    assert c.column_refs == ("Phone numbers",)
    assert c.params == {"format": "phone", "region": "IN"}


def test_protect_variants():
    for text in (
        "Never modify revenue values.",
        "Do not touch revenue.",
        "Don't change revenue.",
        "Leave revenue unchanged.",
        "Keep revenue untouched.",
        "revenue is read-only.",
        "Protect revenue.",
    ):
        c = _single(text)
        assert c.intent == "protected", text
        assert c.column_refs == ("revenue",), text


def test_impute_if_with_confidence():
    c = _single("Missing Age should be estimated only if confidence >95%.")
    assert c.intent == "impute_if"
    assert c.column_refs == ("Age",)
    assert c.params == {"min_confidence": 0.95}


def test_impute_if_without_condition():
    c = _single("Missing Age should be estimated.")
    assert c.intent == "impute_if"
    assert c.params == {}


def test_impute_if_unrepresentable_condition_is_unparsed():
    result = parse_context("Missing Age should be estimated only if the moon is full.")
    assert not result.candidates
    assert len(result.unparsed) == 1


def test_allowed_values():
    c = _single("Allowed status values are active, inactive, pending.")
    assert c.intent == "allowed_values"
    assert c.column_refs == ("status",)
    assert c.params == {"values": ["active", "inactive", "pending"]}


def test_allowed_values_one_of():
    c = _single("status must be one of: active, inactive or pending.")
    assert c.params == {"values": ["active", "inactive", "pending"]}


def test_range():
    c = _single("Age must be between 18 and 100.")
    assert c.intent == "range"
    assert c.column_refs == ("Age",)
    assert c.params == {"lo": 18, "hi": 100}


def test_range_one_sided():
    assert _single("quantity must be at least 1.").params == {"lo": 1, "hi": None}
    assert _single("discount must not exceed 0.5").params == {"lo": None, "hi": 0.5}


def test_dedup_key():
    c = _single("Deduplicate by email and phone.")
    assert c.intent == "dedup_key"
    assert c.column_refs == ("email", "phone")


def test_drop_if():
    c = _single("Drop rows where age is missing.")
    assert c.intent == "drop_if"
    assert c.column_refs == ("age",)
    assert c.params == {"condition": "missing"}
    assert _single("Remove rows with empty email.").params == {"condition": "missing"}


def test_rename():
    c = _single("Rename cust to customer_id.")
    assert c.intent == "rename"
    assert c.column_refs == ("cust",)
    assert c.params == {"new_name": "customer_id"}


def test_map_single_pair():
    c = _single("Replace 'M' with 'Male' in gender.")
    assert c.intent == "map"
    assert c.column_refs == ("gender",)
    assert c.params == {"mapping": {"M": "Male"}}


def test_map_pairs():
    c = _single("Map country values: IND -> India, USA -> United-States")
    assert c.params == {"mapping": {"IND": "India", "USA": "United-States"}}


# -- confidence normalization --------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [">95%", "confidence > 95%", "confidence above 0.95", "only if confidence >= 95 percent"],
)
def test_confidence_forms_normalize_to_fraction(phrase):
    assert parse_confidence(phrase) == 0.95


def test_confidence_absent():
    assert parse_confidence("whenever it feels right") is None


# -- unparsed sentences are surfaced, never guessed ---------------------------


def test_unparsed_sentences_are_returned():
    result = parse_context(
        "CustomerID is unique. The vibes in this dataset are immaculate."
    )
    assert len(result.candidates) == 1
    assert [u.sentence for u in result.unparsed] == [
        "The vibes in this dataset are immaculate"
    ]


def test_parser_is_deterministic():
    text = "Emails must be valid. Age must be between 18 and 100. Garbage line."
    first = parse_context(text)
    second = parse_context(text)
    assert first == second
