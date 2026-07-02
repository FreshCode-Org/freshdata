"""The deterministic tier-0 lexicon: aliases, regions, formats, confidence.

Everything the parser and resolver know about English lives here as plain data
plus a few pure helpers — no models, no inference, no I/O. Extending coverage
means adding entries and (golden-)testing them, never loosening the matching.
"""

from __future__ import annotations

import re

from .normalize import singular_ref, snake_ref

# ---------------------------------------------------------------------------
# Column alias groups: a canonical concept -> names it commonly appears under.
# Both the user's reference and the schema column are snake_cased+singularized
# before lookup, so entries are written in that normal form.
# ---------------------------------------------------------------------------

ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "customer_id": ("customer_id", "cust_id", "custid", "client_id", "customer", "cust"),
    "email": ("email", "email_addr", "email_address", "e_mail", "mail"),
    "phone": (
        "phone",
        "phone_number",
        "phone_no",
        "mobile",
        "mobile_no",
        "mobile_number",
        "contact_no",
        "contact_number",
        "telephone",
        "tel",
        "msisdn",
    ),
    "revenue": (
        "revenue",
        "monthly_revenue",
        "annual_revenue",
        "sales_amount",
        "amount",
        "amount_inr",
        "sales",
        "turnover",
    ),
    "age": ("age",),
    "date": ("date", "signup_date", "created_at", "updated_at", "order_date", "timestamp"),
    "status": ("status", "state",),
    "name": ("name", "full_name", "first_name", "last_name", "customer_name"),
    "address": ("address", "addr", "street_address", "billing_address"),
    "city": ("city", "town"),
    "country": ("country", "country_code", "nation"),
    "zip": ("zip", "zip_code", "zipcode", "postal_code", "pincode", "pin_code", "postcode"),
    "gender": ("gender", "sex"),
    "price": ("price", "unit_price", "list_price"),
    "quantity": ("quantity", "qty", "count", "units"),
    "salary": ("salary", "wage", "pay", "compensation"),
}

#: Reverse index: normal-form name -> canonical alias group key.
_ALIAS_INDEX: dict[str, str] = {}
for _canonical, _members in ALIAS_GROUPS.items():
    _ALIAS_INDEX.setdefault(_canonical, _canonical)
    for _m in _members:
        _ALIAS_INDEX.setdefault(_m, _canonical)


def alias_group(ref: str) -> str | None:
    """The canonical alias-group key for a reference, or ``None``.

    Lookup is on the snake_cased form and its singularized form, so
    ``"Phone numbers"`` and ``"phone_number"`` land in the same group.
    """
    snake = snake_ref(ref)
    return _ALIAS_INDEX.get(snake) or _ALIAS_INDEX.get(singular_ref(snake))


# ---------------------------------------------------------------------------
# Locale words -> ISO-3166 alpha-2 region codes (for LOCALE_FORMAT).
# ---------------------------------------------------------------------------

REGION_WORDS: dict[str, str] = {
    "indian": "IN",
    "india": "IN",
    "american": "US",
    "us": "US",
    "usa": "US",
    "united states": "US",
    "british": "GB",
    "uk": "GB",
    "united kingdom": "GB",
    "german": "DE",
    "germany": "DE",
    "french": "FR",
    "france": "FR",
    "australian": "AU",
    "australia": "AU",
    "canadian": "CA",
    "canada": "CA",
    "brazilian": "BR",
    "brazil": "BR",
    "japanese": "JP",
    "japan": "JP",
    "chinese": "CN",
    "china": "CN",
    "singaporean": "SG",
    "singapore": "SG",
}

#: Regex alternation over region words, longest first (so "united states" wins "us").
REGION_PATTERN = "|".join(
    sorted((re.escape(w) for w in REGION_WORDS), key=len, reverse=True)
)


def region_code(word: str) -> str | None:
    """ISO region code for a locale word (``"Indian"`` -> ``"IN"``)."""
    return REGION_WORDS.get(word.strip().lower())


# ---------------------------------------------------------------------------
# Value formats recognised by VALID_FORMAT / LOCALE_FORMAT.
# ---------------------------------------------------------------------------

FORMAT_WORDS: dict[str, str] = {
    "email": "email",
    "emails": "email",
    "email address": "email",
    "email addresses": "email",
    "e-mail": "email",
    "e-mails": "email",
    "phone": "phone",
    "phones": "phone",
    "phone number": "phone",
    "phone numbers": "phone",
    "mobile": "phone",
    "mobiles": "phone",
    "mobile number": "phone",
    "mobile numbers": "phone",
    "telephone": "phone",
    "url": "url",
    "urls": "url",
    "link": "url",
    "links": "url",
    "website": "url",
    "websites": "url",
    "date": "date",
    "dates": "date",
    "uuid": "uuid",
    "uuids": "uuid",
    "ip address": "ip",
    "ip addresses": "ip",
    "zip code": "zip",
    "zip codes": "zip",
    "postal code": "zip",
    "postal codes": "zip",
    "pincode": "zip",
    "pincodes": "zip",
}

#: Alternation over format words, longest first.
FORMAT_PATTERN = "|".join(
    sorted((re.escape(w) for w in FORMAT_WORDS), key=len, reverse=True)
)


def format_for(word: str) -> str | None:
    """Canonical format name for a surface word (``"Emails"`` -> ``"email"``)."""
    return FORMAT_WORDS.get(word.strip().lower())


# ---------------------------------------------------------------------------
# Confidence phrases -> fraction in [0, 1].
# ---------------------------------------------------------------------------

_CONFIDENCE = re.compile(
    r"""
    (?:confidence|certainty|probability|sure(?:ness)?)?   # optional subject
    \s*
    (?:is\s+|of\s+)?
    (?:>=|>|≥|above|over|at\ least|exceeds|greater\ than|more\ than)?
    \s*
    (?P<value>\d+(?:\.\d+)?)
    \s*
    (?P<pct>%|percent|per\ cent)?
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_confidence(text: str) -> float | None:
    """Extract a confidence threshold from prose, normalized to [0, 1].

    >>> parse_confidence(">95%"), parse_confidence("confidence above 0.95")
    (0.95, 0.95)
    >>> parse_confidence("only if confidence >= 95 percent")
    0.95
    """
    match = _CONFIDENCE.search(text)
    if match is None:
        return None
    value = float(match.group("value"))
    if match.group("pct") or value > 1.0:
        value = value / 100.0
    if not 0.0 <= value <= 1.0:
        return None
    return round(value, 6)
