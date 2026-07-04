"""Deterministic synthetic PII-shaped data.

All identity-shaped values are assembled from invented token lists — none of
the names, addresses, or numbers are derived from real people, customer data,
or production logs. Every generator takes an explicit ``seed`` and produces
byte-identical output for the same seed, and every record carries
``synthetic=True`` metadata (also mirrored in ``DataFrame.attrs``).
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

SYNTHETIC_SOURCE_ID = "synthetic_core_v1"

# Invented tokens only. Common Indian-flavored given names are generic
# dictionary words in this context; surnames and streets are deliberately
# fictional compounds so no full name can collide with a real identity record.
_GIVEN = (
    "asha", "ravi", "neha", "vikram", "priya", "arjun", "meera", "kiran",
    "divya", "rohan", "sneha", "amit", "tara", "dev", "isha", "kabir",
)
_SURNAME = (
    "Voskette", "Ramblewood", "Quenderi", "Talvane", "Mirostra", "Kelvani",
    "Purvatti", "Sandrelo", "Vintara", "Okelmere",
)
_STREET = (
    "Lantern Row", "Copperleaf Marg", "Quarry Lane", "Palmshade Street",
    "Riverbend Road", "Old Mill Avenue", "Jacaranda Nagar", "Harborview Block",
)
_CITY_STATE = (
    ("Mumbai", "Maharashtra"), ("Pune", "Maharashtra"), ("Bengaluru", "Karnataka"),
    ("Chennai", "Tamil Nadu"), ("Hyderabad", "Telangana"), ("Delhi", "Delhi"),
    ("Kolkata", "West Bengal"), ("Jaipur", "Rajasthan"),
)
_COUNTRY = ("India", "India", "India", "Australia", "Austria", "Canada", "Singapore")
_EMAIL_DOMAINS = ("example.com", "example.org", "example.net", "mail.example.in")
_STATUS = ("active", "inactive", "pending", "churned")
_UNITS = ("kg", "g", "pcs", "ltr", "m")
_CATEGORY_CODES = ("EL-100", "EL-200", "AP-300", "GR-110", "HM-220", "TY-500")


def _rng(seed: int) -> random.Random:
    return random.Random(("freshdata-synthetic", seed).__repr__())


def _mark_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    df.attrs["synthetic"] = True
    df.attrs["source_id"] = SYNTHETIC_SOURCE_ID
    return df


def make_customers(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Customer-shaped frame: ids, names, emails, Indian phones, addresses."""
    rng = _rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        given = rng.choice(_GIVEN)
        surname = rng.choice(_SURNAME)
        city, state = rng.choice(_CITY_STATE)
        # Indian mobile numbers start 6-9; keep the canonical +91 form as truth.
        ten_digit = f"{rng.randint(6, 9)}{rng.randrange(10**8, 10**9):09d}"
        rows.append({
            "cust_id": f"C{100000 + i}",
            "full_name": f"{given.title()} {surname}",
            "email": f"{given}.{surname.lower()}{i}@{rng.choice(_EMAIL_DOMAINS)}",
            "phone": f"+91{ten_digit}",
            "address": f"{rng.randint(1, 240)} {rng.choice(_STREET)}",
            "city": city,
            "state": state,
            "country": rng.choice(_COUNTRY),
            "postal_code": f"{rng.randint(110001, 899999)}",
            "national_id_like": f"ZZ{rng.randrange(10**7, 10**8)}X",
            "status": rng.choice(_STATUS),
            "signup_date": (
                f"{rng.randint(2019, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            ),
            "website": f"https://www.{surname.lower()}{i}.example.com",
            "newsletter_opt_in": rng.choice(("true", "false")),
            "synthetic": True,
        })
    return _mark_synthetic(pd.DataFrame(rows))


def make_transactions(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """Transaction-shaped frame: ids, currency strings, quantities, revenue."""
    rng = _rng(seed)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        amount = round(rng.uniform(49.0, 99999.0), 2)
        qty = rng.randint(1, 40)
        rows.append({
            "txn_id": f"T{9000000 + i}",
            "cust_id": f"C{100000 + rng.randrange(200)}",
            "order_status": rng.choice(
                ("placed", "shipped", "delivered", "returned", "cancelled")),
            "category_code": rng.choice(_CATEGORY_CODES),
            "quantity": f"{qty} {rng.choice(_UNITS)}",
            "unit_price_inr": f"₹{amount:,.2f}",
            "monthly_revenue": f"{round(amount * qty, 2)}",
            "order_date": (
                f"{rng.randint(2022, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
            ),
            "notes": rng.choice((
                "priority shipping", "gift wrap requested", "repeat customer",
                "call before delivery", "left at door",
            )),
            "synthetic": True,
        })
    return _mark_synthetic(pd.DataFrame(rows))


def make_context_sentences(seed: int = 2) -> list[dict[str, Any]]:
    """Labeled context sentences (intent ground truth) with author tags.

    ``author`` supports author-disjoint eval splits; the ``hinglish`` authors
    provide the Indian-English / Hinglish paraphrase set required by Phase 5.
    """
    rng = _rng(seed)
    templates: list[tuple[str, str, dict[str, Any], str]] = [
        ("This is an ecommerce customer dataset.", "DOMAIN",
         {"domain": "ecommerce_customer"}, "t0"),
        ("This is a retail transactions dataset.", "DOMAIN",
         {"domain": "retail_transactions"}, "t1"),
        ("This dataset is an ecommerce orders dataset.", "DOMAIN",
         {"domain": "ecommerce_orders"}, "t2"),
        ("This is a customer support dataset.", "DOMAIN", {"domain": "customer_support"}, "t3"),
        # UNIQUE — shared vocabulary: unique / duplicate / once / repeat
        ("{col} is unique.", "UNIQUE", {}, "t0"),
        ("{col} must be unique.", "UNIQUE", {}, "t1"),
        ("{col} values should be unique with no duplicates.", "UNIQUE", {}, "t3"),
        ("No duplicate values in {col}.", "UNIQUE", {}, "t4"),
        ("{col} should not repeat.", "UNIQUE", {}, "t0"),
        ("Each {col} appears once, duplicates are not allowed.", "UNIQUE", {}, "t1"),
        ("{col} appears only once per table.", "UNIQUE", {}, "t3"),
        ("Every value in {col} occurs only once.", "UNIQUE", {}, "t4"),
        ("Every {col} should appear only once.", "UNIQUE", {}, "t2"),
        ("{col} must be unique, one per row.", "UNIQUE", {}, "t2"),
        ("{col} har row me unique hona chahiye.", "UNIQUE", {}, "hinglish_a"),
        ("{col} repeat nahi hona chahiye, unique rakho.", "UNIQUE", {}, "hinglish_b"),
        # VALID_FORMAT — shared vocabulary: valid / email(s)
        ("{col} must be valid emails.", "VALID_FORMAT", {"format": "email"}, "t0"),
        ("Emails in {col} must be valid.", "VALID_FORMAT", {"format": "email"}, "t1"),
        ("{col} should contain valid email addresses.", "VALID_FORMAT", {"format": "email"}, "t3"),
        ("Make sure {col} emails are valid.", "VALID_FORMAT", {"format": "email"}, "t4"),
        ("{col} emails must all be valid addresses.", "VALID_FORMAT", {"format": "email"}, "t2"),
        ("{col} me valid email hona chahiye.", "VALID_FORMAT", {"format": "email"}, "hinglish_a"),
        ("{col} ke emails valid hone chahiye.", "VALID_FORMAT", {"format": "email"}, "hinglish_b"),
        # LOCALE_FORMAT — shared vocabulary: Indian / phone numbers
        ("{col} are Indian phone numbers.", "LOCALE_FORMAT",
         {"locale": "IN", "format": "phone"}, "t0"),
        ("Phone numbers are Indian.", "LOCALE_FORMAT", {"locale": "IN", "format": "phone"}, "t1"),
        ("{col} holds Indian phone numbers.", "LOCALE_FORMAT",
         {"locale": "IN", "format": "phone"}, "t3"),
        ("Treat {col} as Indian phone numbers.", "LOCALE_FORMAT",
         {"locale": "IN", "format": "phone"}, "t4"),
        ("The phone numbers in {col} are Indian numbers.", "LOCALE_FORMAT",
         {"locale": "IN", "format": "phone"}, "t2"),
        ("{col} me Indian phone numbers hain.", "LOCALE_FORMAT",
         {"locale": "IN", "format": "phone"}, "hinglish_a"),
        ("{col} ke phone numbers Indian hain.", "LOCALE_FORMAT",
         {"locale": "IN", "format": "phone"}, "hinglish_b"),
        # PROTECT — shared vocabulary: never modify / protected / do not change
        ("Never modify {col} values.", "PROTECT", {}, "t0"),
        ("Do not touch {col}.", "PROTECT", {}, "t1"),
        ("Do not change {col}.", "PROTECT", {}, "t3"),
        ("{col} must never be modified.", "PROTECT", {}, "t4"),
        ("Leave {col} untouched, it is protected.", "PROTECT", {}, "t0"),
        ("{col} is protected.", "PROTECT", {}, "t2"),
        ("{col} is protected, never change it.", "PROTECT", {}, "t2"),
        ("{col} kabhi mat badalna.", "PROTECT", {}, "hinglish_a"),
        ("{col} ko modify mat karna, protected hai.", "PROTECT", {}, "hinglish_a"),
        ("{col} ko change mat karo, bilkul nahi.", "PROTECT", {}, "hinglish_b"),
        ("{col} protected hai, mat badalna.", "PROTECT", {}, "hinglish_b"),
        # IMPUTE_IF — shared vocabulary: missing / estimated / confidence %
        ("Missing {col} should be estimated only if confidence >95%.", "IMPUTE_IF",
         {"threshold": 0.95}, "t0"),
        ("Fill {col} gaps only when at least 95% sure.", "IMPUTE_IF", {"threshold": 0.95}, "t1"),
        ("Estimate missing {col} only above 95% confidence.", "IMPUTE_IF",
         {"threshold": 0.95}, "t3"),
        ("Impute {col} only with confidence over 90%.", "IMPUTE_IF", {"threshold": 0.90}, "t4"),
        ("Missing {col} may be estimated only at 95% confidence or more.", "IMPUTE_IF",
         {"threshold": 0.95}, "t2"),
        ("{col} missing ho to sirf 95% confidence par estimate karo.", "IMPUTE_IF",
         {"threshold": 0.95}, "hinglish_a"),
        ("{col} khali ho to sirf 95% sure hone par bharna.", "IMPUTE_IF", {"threshold": 0.95},
         "hinglish_a"),
        ("{col} khali ho to 95% pakka hone par hi bharna.", "IMPUTE_IF", {"threshold": 0.95},
         "hinglish_b"),
        # ALLOWED_VALUES — shared vocabulary: allowed / values / only be
        ("Allowed status values are active, inactive, pending.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "t0"),
        ("{col} can only be active, inactive or pending.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "t1"),
        ("Allowed values for {col} are active, inactive, pending.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "t3"),
        ("Valid {col} values: active, inactive, pending.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "t4"),
        ("The only allowed {col} values are active, inactive and pending.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "t2"),
        ("Status sirf active, inactive ya pending ho sakta hai.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "hinglish_a"),
        ("{col} sirf active, inactive ya pending allowed hai.", "ALLOWED_VALUES",
         {"values": ["active", "inactive", "pending"]}, "hinglish_b"),
        # RANGE — shared vocabulary: between X and Y / within
        ("{col} must be between 18 and 99.", "RANGE", {"min": 18, "max": 99}, "t0"),
        ("Keep {col} within 0 to 100.", "RANGE", {"min": 0, "max": 100}, "t1"),
        ("{col} should stay between 1 and 40.", "RANGE", {"min": 1, "max": 40}, "t3"),
        ("Values of {col} must lie between 0 and 999.", "RANGE", {"min": 0, "max": 999}, "t4"),
        ("{col} has to be between 18 and 65.", "RANGE", {"min": 18, "max": 65}, "t2"),
        ("{col} 5 aur 95 ke beech hona chahiye.", "RANGE", {"min": 5, "max": 95}, "hinglish_a"),
        ("{col} between 10 aur 50 ke beech hona chahiye.", "RANGE", {"min": 10, "max": 50},
         "hinglish_b"),
        # DEDUP_KEY — shared vocabulary: deduplicate / dedup key
        ("Deduplicate rows by {col}.", "DEDUP_KEY", {}, "t0"),
        ("Use {col} as the dedup key.", "DEDUP_KEY", {}, "t1"),
        ("Deduplicate on {col}.", "DEDUP_KEY", {}, "t3"),
        ("Rows should be deduplicated using {col} as the key.", "DEDUP_KEY", {}, "t2"),
        ("{col} se deduplicate karo.", "DEDUP_KEY", {}, "hinglish_b"),
        # DROP_IF — shared vocabulary: drop / remove rows / empty / missing
        ("Drop rows if {col} is empty.", "DROP_IF", {"condition": "empty"}, "t0"),
        ("Remove records where {col} is missing.", "DROP_IF", {"condition": "empty"}, "t1"),
        ("Drop any row with empty {col}.", "DROP_IF", {"condition": "empty"}, "t3"),
        ("Missing {col} rows should be dropped.", "DROP_IF", {"condition": "empty"}, "t4"),
        ("Rows should be dropped when {col} is missing.", "DROP_IF", {"condition": "empty"}, "t0"),
        ("Rows with missing {col} should be dropped.", "DROP_IF", {"condition": "empty"}, "t2"),
        ("{col} empty ho to row drop kar do.", "DROP_IF", {"condition": "empty"}, "hinglish_b"),
        # RENAME — shared vocabulary: rename ... to
        ("Rename {col} to customer_id.", "RENAME", {"to": "customer_id"}, "t0"),
        ("Please rename {col} to customer_id.", "RENAME", {"to": "customer_id"}, "t1"),
        ("Rename column {col} to customer_id.", "RENAME", {"to": "customer_id"}, "t3"),
        ("{col} ko customer_id rename kar do.", "RENAME", {"to": "customer_id"}, "hinglish_a"),
        ("Rename the {col} column to customer_id.", "RENAME", {"to": "customer_id"}, "t2"),
        ("{col} ko rename karke customer_id kar do.", "RENAME", {"to": "customer_id"},
         "hinglish_b"),
        # MAP — shared vocabulary: map X to Y
        ("Map M to male and F to female in {col}.", "MAP",
         {"mapping": {"M": "male", "F": "female"}}, "t0"),
        ("In {col}, map M to male and F to female.", "MAP",
         {"mapping": {"M": "male", "F": "female"}}, "t1"),
        ("Map the code M to male and the code F to female in {col}.", "MAP",
         {"mapping": {"M": "male", "F": "female"}}, "t2"),
        ("{col} me M ko male map karo aur F ko female.", "MAP",
         {"mapping": {"M": "male", "F": "female"}}, "hinglish_a"),
        ("{col} me M ko male aur F ko female map karo.", "MAP",
         {"mapping": {"M": "male", "F": "female"}}, "hinglish_b"),
        # UNKNOWN — no actionable constraint
        ("The vibes of this table are immaculate.", "UNKNOWN", {}, "t0"),
        ("Please make everything nicer somehow.", "UNKNOWN", {}, "t1"),
        ("Great data, love it.", "UNKNOWN", {}, "t3"),
        ("Have a look and see what you think.", "UNKNOWN", {}, "t4"),
        ("Honestly this table just needs some love.", "UNKNOWN", {}, "t2"),
        ("Kuch accha sa kar do is table ke saath.", "UNKNOWN", {}, "hinglish_b"),
        ("Table ke saath kuch accha kar do.", "UNKNOWN", {}, "hinglish_a"),
        ("Data ko thoda accha bana do yaar.", "UNKNOWN", {}, "hinglish_a"),
    ]
    columns = ("CustomerID", "cust_id", "email_addr", "mobile", "status",
               "monthly_revenue", "age", "order_date", "postal_code")
    out: list[dict[str, Any]] = []
    for text, intent, slots, author in templates:
        picks = rng.sample(columns, k=3) if "{col}" in text else [None]
        for col in picks:
            sentence = text.format(col=col) if col is not None else text
            out.append({
                "sentence": sentence,
                "intent": intent,
                "slots": dict(slots, column=col) if col is not None else dict(slots),
                "author": author,
                "synthetic": True,
                "source_id": SYNTHETIC_SOURCE_ID,
            })
    return out


def seed_tables(seed: int = 0) -> dict[str, pd.DataFrame]:
    """All synthetic seed tables keyed by name."""
    return {
        "customers": make_customers(seed=seed),
        "transactions": make_transactions(seed=seed + 1),
    }
