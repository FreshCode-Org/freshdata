"""Semantic-value corruptors: emails, Indian phones, categories, dates.

Ambiguous corruptions (two plausible repairs, dangerous near-pairs, dayfirst
puzzles) are always labeled ``should_auto_apply=False`` — a cleaner that
auto-applies them fails CleanBench by construction.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Corruptor

_SPELLED = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "20": "twenty", "25": "twenty five", "30": "thirty", "40": "forty",
    "50": "fifty", "100": "one hundred",
}
_MONTHS = ("January", "February", "March", "April", "May", "June", "July",
           "August", "September", "October", "November", "December")


# -- emails --------------------------------------------------------------------

def _one_at(v: str) -> bool:
    return v.count("@") == 1 and "." in v.partition("@")[2]


def _email_at_whitespace(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _one_at(v):
        return None
    local, _, domain = v.partition("@")
    return f"{local}{rng.choice((' @', ' @ ', '@ '))}{domain}"


def _email_double_at(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    return v.replace("@", "@@", 1) if _one_at(v) else None


def _email_casing(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _one_at(v) or v == v.upper():
        return None
    return rng.choice((v.upper(), v.title()))


def _email_punct_noise(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not _one_at(v):
        return None
    local, _, domain = v.partition("@")
    return rng.choice((
        f"{local}.@{domain}",          # trailing dot in local part
        f"{local}@{domain},",          # trailing comma
        f"{local}@{domain}.",          # trailing dot
        f"{local.replace('.', '..', 1)}@{domain}" if "." in local else f"{local},@{domain}",
    ))


# -- Indian phones ---------------------------------------------------------------

def _canonical_in_phone(v: str) -> str | None:
    digits = "".join(ch for ch in v if ch.isdigit())
    if v.startswith("+91") and len(digits) == 12 and digits.startswith("91"):
        return digits[2:]
    return None


def _phone_spacing(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    ten = _canonical_in_phone(v)
    if ten is None:
        return None
    return rng.choice((
        f"{ten[:5]} {ten[5:]}", f"+91 {ten[:5]} {ten[5:]}", f"{ten[:4]} {ten[4:7]} {ten[7:]}",
    ))


def _phone_zero_prefix(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    ten = _canonical_in_phone(v)
    return f"0{ten}" if ten else None


def _phone_plus91_format(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    ten = _canonical_in_phone(v)
    if ten is None:
        return None
    return rng.choice((f"(+91) {ten}", f"+91-{ten}", f"91{ten}", f"+91 {ten}"))


def _phone_hyphenation(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    ten = _canonical_in_phone(v)
    if ten is None:
        return None
    return rng.choice((
        f"{ten[:5]}-{ten[5:]}", f"+91-{ten[:5]}-{ten[5:]}", f"{ten[:3]}-{ten[3:6]}-{ten[6:]}",
    ))


def _phone_unsafe(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    """Digit loss/gain: the value is wrong and there is no safe repair."""
    ten = _canonical_in_phone(v)
    if ten is None:
        return None
    cut = rng.randrange(10)
    return rng.choice((
        ten[:cut] + ten[cut + 1:],                      # dropped digit
        ten[:cut] + str(rng.randrange(10)) + ten[cut:],  # inserted digit
    ))


# -- category / reference values ---------------------------------------------------

def _allowed_case(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if not v.isalpha():
        return None
    choices = [c for c in (v.upper(), v.title(), v.swapcase()) if c != v]
    return rng.choice(choices) if choices else None


def _allowed_whitespace(v: str, rng: random.Random, params: dict[str, Any]) -> str:
    return rng.choice((f" {v}", f"{v} ", f"  {v}  ", f"{v}\t"))


def _allowed_separator(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if len(v) < 4:
        return None
    cut = rng.randrange(1, len(v) - 1)
    return v[:cut] + rng.choice("-_ ") + v[cut:]


def _edit_distance_typo(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if len(v) < 4 or not v.isalpha():
        return None
    pos = rng.randrange(1, len(v) - 1)
    op = rng.choice(("swap", "drop", "double"))
    if op == "swap":
        return v[:pos] + v[pos + 1] + v[pos] + v[pos + 2:] if pos + 1 < len(v) else None
    if op == "drop":
        return v[:pos] + v[pos + 1:]
    return v[:pos] + v[pos] + v[pos:]


def _ambiguous_category(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    """Values one edit from *two* allowed values (e.g. ``nactive``)."""
    pairs = params.get("ambiguous_map", {"active": "nactive", "inactive": "nactive"})
    return pairs.get(v.strip().lower())


def _country_code_ambiguity(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    codes = params.get(
        "country_codes", {"india": "IN", "australia": "AU", "austria": "AT", "canada": "CA"})
    return codes.get(v.strip().lower())


def _short_code_ambiguity(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    codes = params.get("short_codes", {"california": "CA", "canada": "CA", "maharashtra": "MH"})
    return codes.get(v.strip().lower())


def _close_category_pair(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    """Dangerous near-pairs: Austria/Australia, May/month-vs-name, ID codes."""
    pairs = params.get("close_pairs", {
        "austria": "Austrlia", "australia": "Austrlia",
        "delivered": "deliverd", "returned": "returend",
    })
    return pairs.get(v.strip().lower())


def _spelled_number(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    return _SPELLED.get(v.strip())


# -- dates -------------------------------------------------------------------------

def _iso_parts(v: str) -> tuple[int, int, int] | None:
    parts = v.split("-")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    year, month, day = (int(p) for p in parts)
    if not (1 <= month <= 12 and 1 <= day <= 28):
        return None
    return year, month, day


def _date_format_shuffle(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    parsed = _iso_parts(v)
    if parsed is None:
        return None
    year, month, day = parsed
    return rng.choice((
        f"{_MONTHS[month - 1]} {day}, {year}",
        f"{day} {_MONTHS[month - 1]} {year}",
        f"{year}/{month:02d}/{day:02d}",
    ))


def _date_dayfirst_ambiguity(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    parsed = _iso_parts(v)
    if parsed is None:
        return None
    year, month, day = parsed
    if day > 12 or day == month:
        return None  # unambiguous or identity — not a dayfirst puzzle
    return rng.choice((f"{month:02d}/{day:02d}/{year}", f"{day:02d}/{month:02d}/{year}"))


def _relative_date_phrase(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if _iso_parts(v) is None:
        return None
    return rng.choice(("last Tuesday", "yesterday", "two weeks ago", "next Monday"))


def _month_abbreviation(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    parsed = _iso_parts(v)
    if parsed is None:
        return None
    year, month, day = parsed
    abbr = _MONTHS[month - 1][:3]
    return rng.choice((f"{day}-{abbr}-{year}", f"{abbr} {day} {year}", f"{day} {abbr}. {year}"))


def _invalid_date_phrase(v: str, rng: random.Random, params: dict[str, Any]) -> str | None:
    if _iso_parts(v) is None:
        return None
    return rng.choice(("2025-13-45", "31/02/2024", "0000-00-00", "the 32nd of Neverember"))


# --------------------------------------------------------------------------- #

spelled_number_replacement = Corruptor(
    name="spelled_number_replacement", family="semantic_value",
    fn=_spelled_number, risk="medium")
email_at_whitespace = Corruptor(
    name="email_at_whitespace", family="semantic_value", fn=_email_at_whitespace)
email_double_at = Corruptor(
    name="email_double_at", family="semantic_value", fn=_email_double_at)
email_casing = Corruptor(
    name="email_casing", family="semantic_value", fn=_email_casing)
email_punct_noise = Corruptor(
    name="email_punct_noise", family="semantic_value", fn=_email_punct_noise,
    risk="medium")
phone_in_spacing = Corruptor(
    name="phone_in_spacing", family="semantic_value", fn=_phone_spacing)
phone_in_zero_prefix = Corruptor(
    name="phone_in_zero_prefix", family="semantic_value", fn=_phone_zero_prefix)
phone_in_plus91_format = Corruptor(
    name="phone_in_plus91_format", family="semantic_value", fn=_phone_plus91_format)
phone_hyphenation = Corruptor(
    name="phone_hyphenation", family="semantic_value", fn=_phone_hyphenation)
phone_unsafe_mutation = Corruptor(
    name="phone_unsafe_mutation", family="semantic_value", fn=_phone_unsafe,
    risk="high", should_repair=False, should_auto_apply=False, ambiguous=True)

allowed_value_case = Corruptor(
    name="allowed_value_case", family="reference_value", fn=_allowed_case)
allowed_value_whitespace = Corruptor(
    name="allowed_value_whitespace", family="reference_value", fn=_allowed_whitespace)
allowed_value_separator = Corruptor(
    name="allowed_value_separator", family="reference_value", fn=_allowed_separator)
edit_distance_typo = Corruptor(
    name="edit_distance_typo", family="reference_value", fn=_edit_distance_typo,
    risk="medium")
ambiguous_category = Corruptor(
    name="ambiguous_category", family="reference_value", fn=_ambiguous_category,
    risk="high", should_auto_apply=False, ambiguous=True)
country_code_ambiguity = Corruptor(
    name="country_code_ambiguity", family="reference_value", fn=_country_code_ambiguity,
    risk="high", should_auto_apply=False, ambiguous=True)
short_code_ambiguity = Corruptor(
    name="short_code_ambiguity", family="reference_value", fn=_short_code_ambiguity,
    risk="high", should_auto_apply=False, ambiguous=True)
close_category_pair = Corruptor(
    name="close_category_pair", family="reference_value", fn=_close_category_pair,
    risk="high", should_auto_apply=False, ambiguous=True)

date_format_shuffle = Corruptor(
    name="date_format_shuffle", family="date_time", fn=_date_format_shuffle)
date_dayfirst_ambiguity = Corruptor(
    name="date_dayfirst_ambiguity", family="date_time", fn=_date_dayfirst_ambiguity,
    risk="high", should_auto_apply=False, ambiguous=True)
relative_date_phrase = Corruptor(
    name="relative_date_phrase", family="date_time", fn=_relative_date_phrase,
    risk="high", should_repair=False, should_auto_apply=False, ambiguous=True)
month_name_abbreviation = Corruptor(
    name="month_name_abbreviation", family="date_time", fn=_month_abbreviation)
invalid_date_phrase = Corruptor(
    name="invalid_date_phrase", family="date_time", fn=_invalid_date_phrase,
    risk="medium", should_auto_apply=False, params={"target": "missing"})

SEMANTIC_VALUE_CORRUPTORS = (
    spelled_number_replacement,
    email_at_whitespace,
    email_double_at,
    email_casing,
    email_punct_noise,
    phone_in_spacing,
    phone_in_zero_prefix,
    phone_in_plus91_format,
    phone_hyphenation,
    phone_unsafe_mutation,
    allowed_value_case,
    allowed_value_whitespace,
    allowed_value_separator,
    edit_distance_typo,
    ambiguous_category,
    country_code_ambiguity,
    short_code_ambiguity,
    close_category_pair,
    date_format_shuffle,
    date_dayfirst_ambiguity,
    relative_date_phrase,
    month_name_abbreviation,
    invalid_date_phrase,
)
